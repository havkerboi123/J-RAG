import asyncio
import os

import httpx

# "clef" (27B, more accurate) or "clef-flash" (9B, faster, ~40% of the price).
CLEF_MODEL = "clef"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class ClefClient:
    """Async client for Cloudflare's Clef on Workers AI. Same request format as Jev, plus optional images."""

    def __init__(self, model: str = CLEF_MODEL, max_concurrency: int = 16, max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self.input_tokens = 0
        self._url = (
            f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}"
            f"/ai/run/@cf/cloudflare/{model}"
        )
        self._sem = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.environ['CLOUDFLARE_AUTH_TOKEN']}"},
            timeout=60,
        )

    async def ask(self, state: dict | str, questions: dict, images: list[str] | None = None) -> dict:
        """`images` are data URLs (data:image/jpeg;base64,...), at most 4 per call."""
        body = {"model": self.model, "state": state, "questions": questions}
        if images:
            body["images"] = images
        for attempt in range(self.max_retries + 1):
            try:
                async with self._sem:
                    resp = await self._http.post(self._url, json=body)
            except httpx.TransportError:
                # Dropped connection or timeout: retry rather than lose the whole run.
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                break
            await asyncio.sleep(float(resp.headers.get("retry-after", 2**attempt)))

        data = resp.json()
        if not resp.is_success or not data.get("success"):
            raise RuntimeError(f"Clef request failed ({resp.status_code}): {data.get('errors')}")

        # Workers AI wraps the model output in Cloudflare's {"result": ..., "success": ...} envelope.
        result = data["result"]
        self.calls += 1
        self.input_tokens += result["usage"]["input_tokens"]
        return result["answers"]

    async def aclose(self) -> None:
        await self._http.aclose()
