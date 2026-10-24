import asyncio
import os

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class JevClient:
    """Minimal async client for TypeSafe's System One endpoint, with retries and a concurrency cap."""

    def __init__(self, model: str = JEV_MODEL, max_concurrency: int = 16, max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self.input_tokens = 0
        self._sem = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
            timeout=30,
        )

    async def ask(self, state: dict | str, questions: dict) -> dict:
        body = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.max_retries + 1):
            async with self._sem:
                resp = await self._http.post(JEV_URL, json=body)
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                break
            await asyncio.sleep(float(resp.headers.get("retry-after", 2**attempt)))
        resp.raise_for_status()

        data = resp.json()
        self.calls += 1
        self.input_tokens += data["usage"]["input_tokens"]
        return data["answers"]

    async def aclose(self) -> None:
        await self._http.aclose()
