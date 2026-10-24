from openai import AsyncOpenAI

from corpus import Passage

MODEL = "gpt-5-mini"

INSTRUCTIONS = (
    "Answer the user's question using only the numbered sources provided. Sources are either text passages "
    "or images of document pages; a page can mix running text, tables, charts and figures, so read the values "
    "in tables and charts as well as the text. Consecutive pages of the same report are included together, "
    "so a table or passage may continue from one page to the next. "
    "Cite the sources you rely on inline, like [2]. "
    "If the sources don't contain enough information to answer, say so plainly instead of guessing."
)


async def answer(client: AsyncOpenAI, question: str, passages: list[Passage]) -> str:
    # Page images go to the model as images (the same view Clef ranked them on); text-only passages as text.
    content = []
    for i, p in enumerate(passages, 1):
        if p.image:
            where = f": {p.doc_id}, page {p.page}" if p.doc_id else ""
            content.append({"type": "input_text", "text": f"[{i}] (page image{where})"})
            content.append({"type": "input_image", "image_url": p.image, "detail": "high"})
        else:
            content.append({"type": "input_text", "text": f"[{i}] {p.text}"})
    content.append({"type": "input_text", "text": f"Question: {question}"})

    response = await client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": content}],
    )
    return response.output_text
