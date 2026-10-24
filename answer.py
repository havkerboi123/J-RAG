from openai import AsyncOpenAI

from corpus import Passage

MODEL = "gpt-5-mini"

INSTRUCTIONS = (
    "Answer the user's question using only the numbered passages provided. "
    "Cite the passages you rely on inline, like [2]. "
    "If the passages don't contain enough information to answer, say so plainly instead of guessing."
)


async def answer(client: AsyncOpenAI, question: str, passages: list[Passage]) -> str:
    context = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, 1))
    response = await client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=f"<passages>\n{context}\n</passages>\n\nQuestion: {question}",
    )
    return response.output_text
