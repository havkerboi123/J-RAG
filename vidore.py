"""ViDoRe V3 HR: EU HR reports, 1,110 pages with OCR'd markdown, 318 English questions."""

import base64
import io
import random

from datasets import load_dataset

from corpus import Corpus, Passage

DATASET = "vidore/vidore_v3_hr"
# Clef downscales anything over ~1 megapixel anyway; shrinking first keeps requests small.
MAX_IMAGE_SIDE = 1024


def to_data_url(image) -> str:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_vidore_corpus(limit: int | None = None, seed: int = 0, images: bool = False) -> Corpus:
    # Without `images`, the image column is never read, so pages are downloaded but not decoded.
    page_columns = ["corpus_id", "doc_id", "page_number_in_doc", "markdown"] + (["image"] if images else [])
    pages = load_dataset(DATASET, data_dir="corpus", split="test").select_columns(page_columns)
    queries = load_dataset(DATASET, data_dir="queries", split="test").select_columns(
        ["query_id", "query", "answer", "language"]
    )
    qrels = load_dataset(DATASET, data_dir="qrels", split="test").select_columns(["query_id", "corpus_id"])

    # Each question is stored in 6 languages; keep the English originals.
    rows = [q for q in queries if q["language"] == "english"]
    gold: dict[int, set[str]] = {q["query_id"]: set() for q in rows}
    for r in qrels:
        if r["query_id"] in gold:
            gold[r["query_id"]].add(f"page{r['corpus_id']}")

    rows = [q for q in rows if gold[q["query_id"]]]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)

    passages = []
    for p in pages:
        text = (p["markdown"] or "").strip()
        image = to_data_url(p["image"]) if images else None
        # Without an image, a page with no OCR text (e.g. a full-page chart) has nothing to judge.
        if text or image:
            passages.append(
                Passage(
                    id=f"page{p['corpus_id']}",
                    text=text,
                    image=image,
                    doc_id=p["doc_id"],
                    page=p["page_number_in_doc"],
                )
            )
    # Reading order, so consecutive passages are consecutive pages of the same report.
    passages.sort(key=lambda p: (p.doc_id, p.page))

    return Corpus(
        passages=passages,
        questions=[q["query"] for q in rows],
        references=[q["answer"] for q in rows],
        gold=[gold[q["query_id"]] for q in rows],
    )
