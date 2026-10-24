"""Retrieval with a System One decision model (Clef or Jev), no embeddings.

With `window=1`, every passage gets its own call. Passages that have a page image are judged
from the image alone; the rest from their text.

With `window>1` (page images only), each call shows Clef that many consecutive pages of one
report, overlapping by one page, and asks the questions separately for each page. Pages get
their neighbours as context but keep their own score; a page seen in two windows keeps its best.

Either way, calls run in parallel and pages are ranked by score. The best `top_k` go to the LLM,
each with `neighbours` pages either side from the same report, so an answer split across pages
reaches the LLM whole without paying Clef for windows.
"""

import asyncio
import itertools
from dataclasses import dataclass

from clef import ClefClient
from corpus import Passage
from jev import JevClient

Ranker = ClefClient | JevClient
ENGINES = ["clef", "jev"]

# How much each answer counts toward a passage's ranking score. A passage that answers the
# question alone scores ~1.0, one holding a piece of the answer (e.g. one year of a comparison)
# ~0.6, and one that is merely on-topic ~0.2.
WEIGHTS = {"sufficient": 0.4, "partial": 0.4, "related": 0.2}

# Text-only passages (amnesty_qa, or any corpus run through Jev).
TEXT_QUESTIONS = {
    "sufficient": {
        "type": "noul",
        "instructions": "Does this passage contain enough information to answer the user's question?",
    },
    "partial": {
        "type": "noul",
        "instructions": (
            "Does this passage contain at least part of the information needed to answer the user's question, "
            "such as one of several facts the answer depends on?"
        ),
    },
    "related": {
        "type": "noul",
        "instructions": "Does this passage contain information related to the user's question?",
    },
}

# Document pages sent to Clef as an image only. Explained once in the state rather than in
# every question, since question text is billed per question.
PAGE_FORMAT = (
    "The attached image is one page from a document. The page can mix running text, tables, charts, "
    "graphs, figures and photos. Read all of it, including the values shown in tables and charts."
)

PAGE_QUESTIONS = {
    "sufficient": {
        "type": "noul",
        "instructions": "Does this page (its text, tables, charts or figures) contain enough information to answer `question`?",
    },
    "partial": {
        "type": "noul",
        "instructions": (
            "Does this page (its text, tables, charts or figures) contain at least part of the information needed "
            "to answer `question`, such as one of several facts or figures the answer depends on?"
        ),
    },
    "related": {
        "type": "noul",
        "instructions": "Does this page (its text, tables, charts or figures) contain information related to `question`?",
    },
}


# Consecutive pages shown together in one Clef call.
WINDOW_FORMAT = (
    "The attached images are consecutive pages from one document, in reading order: image 1 comes first. "
    "`pages` says which page each image is. Neighbouring pages are included, overlapping with other windows, "
    "so you can see the context around each page: what a section is about, a table's header on the page before, "
    "or text that continues onto the next page. Use them to understand each page, but judge each page on what "
    "it contains itself. Pages can mix running text, tables, charts, graphs, figures and "
    "photos, and a table or section can continue from one page to the next. Read all of it, including the "
    "values shown in tables and charts."
)
# Each image is judged on its own content; its neighbours are only there for context.
WINDOW_INSTRUCTIONS = {
    "sufficient": "Does image {i} itself (not its neighbours) contain enough information to answer `question`?",
    "partial": (
        "Does image {i} itself (not its neighbours) contain at least part of the information needed to answer "
        "`question`, such as one of several facts or figures the answer depends on?"
    ),
    "related": "Does image {i} itself (not its neighbours) contain information related to `question`?",
}
MAX_CLEF_IMAGES = 4


def make_ranker(engine: str) -> Ranker:
    return ClefClient() if engine == "clef" else JevClient()


@dataclass
class RetrievalConfig:
    top_k: int = 5
    window: int = 1  # pages per Clef call; 1 scores each page on its own
    neighbours: int = 1  # pages either side of each top page also sent to the LLM (same report only)


def make_windows(passages: list[Passage], size: int) -> list[list[Passage]]:
    """Consecutive pages of the same report, overlapping by one page: 1-3, 3-5, 5-7, ..."""
    windows = []
    for _, doc_pages in itertools.groupby(passages, key=lambda p: p.doc_id):
        doc_pages = list(doc_pages)
        for start in range(0, len(doc_pages), size - 1):
            windows.append(doc_pages[start : start + size])
            if start + size >= len(doc_pages):
                break
    return windows


def with_neighbours(top: list[Passage], passages: list[Passage], n: int) -> list[Passage]:
    """Each top page with up to `n` pages either side from the same report, in reading order, no repeats.

    `passages` must be in reading order. Passages without a report (plain text corpora) get no neighbours.
    """
    position = {p.id: i for i, p in enumerate(passages)}
    context: list[Passage] = []
    seen: set[str] = set()
    for page in top:
        i = position[page.id]
        block = [
            passages[j]
            for j in range(max(i - n, 0), min(i + n + 1, len(passages)))
            if page.doc_id and passages[j].doc_id == page.doc_id
        ] or [page]
        for p in block:
            if p.id not in seen:
                seen.add(p.id)
                context.append(p)
    if not all(p.doc_id for p in context):
        return context  # plain text passages: keep rank order
    # Group by report (best-ranked report first) and keep pages in reading order within each.
    report_rank = {doc: r for r, doc in reversed(list(enumerate(p.doc_id for p in context)))}
    return sorted(context, key=lambda p: (report_rank[p.doc_id], position[p.id]))


def calls_per_question(passages: list[Passage], config: RetrievalConfig) -> int:
    return len(passages) if config.window == 1 else len(make_windows(passages, config.window))


@dataclass
class ScoredPassage:
    passage: Passage
    sufficient: float
    partial: float
    related: float

    @property
    def score(self) -> float:
        return sum(weight * getattr(self, name) for name, weight in WEIGHTS.items())


@dataclass
class Retrieval:
    top: list[Passage]  # the best `top_k` pages
    passages: list[Passage]  # what the LLM gets: `top` plus their neighbours
    ranked: list[ScoredPassage]  # every passage, best first

    @property
    def sufficient(self) -> float:
        """Highest confidence that one of the top pages answers the question."""
        return max((s.sufficient for s in self.ranked[: len(self.top)]), default=0.0)


async def score_passage(ranker: Ranker, question: str, passage: Passage) -> ScoredPassage:
    if passage.image:
        state = {"page_format": PAGE_FORMAT, "question": question}
        answers = await ranker.ask(state, PAGE_QUESTIONS, images=[passage.image])
    else:
        answers = await ranker.ask({"question": question, "passage": passage.text}, TEXT_QUESTIONS)
    return ScoredPassage(
        passage,
        sufficient=answers["sufficient"]["noul"],
        partial=answers["partial"]["noul"],
        related=answers["related"]["noul"],
    )


async def score_window(ranker: Ranker, question: str, pages: list[Passage]) -> list[ScoredPassage]:
    state = {
        "page_format": WINDOW_FORMAT,
        "question": question,
        "pages": [f"image {i}: {p.doc_id}, page {p.page}" for i, p in enumerate(pages, 1)],
    }
    questions = {
        f"image{i}_{name}": {"type": "noul", "instructions": template.format(i=i)}
        for i in range(1, len(pages) + 1)
        for name, template in WINDOW_INSTRUCTIONS.items()
    }
    answers = await ranker.ask(state, questions, images=[p.image for p in pages])
    return [
        ScoredPassage(
            p,
            sufficient=answers[f"image{i}_sufficient"]["noul"],
            partial=answers[f"image{i}_partial"]["noul"],
            related=answers[f"image{i}_related"]["noul"],
        )
        for i, p in enumerate(pages, 1)
    ]


async def retrieve(
    ranker: Ranker,
    question: str,
    passages: list[Passage],
    config: RetrievalConfig = RetrievalConfig(),
) -> Retrieval:
    if config.window == 1:
        scored = await asyncio.gather(*(score_passage(ranker, question, p) for p in passages))
    else:
        if not 1 < config.window <= MAX_CLEF_IMAGES:
            raise ValueError(f"window must be 1-{MAX_CLEF_IMAGES} (Clef takes at most {MAX_CLEF_IMAGES} images)")
        if not all(p.image for p in passages):
            raise ValueError("windows need page images: use --engine clef on a page dataset, or --window 1")
        windows = make_windows(passages, config.window)
        per_window = await asyncio.gather(*(score_window(ranker, question, w) for w in windows))
        # A page in two overlapping windows keeps whichever score is higher.
        best: dict[str, ScoredPassage] = {}
        for s in itertools.chain.from_iterable(per_window):
            if s.passage.id not in best or s.score > best[s.passage.id].score:
                best[s.passage.id] = s
        scored = list(best.values())
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    top = [s.passage for s in ranked[: config.top_k]]
    return Retrieval(top=top, passages=with_neighbours(top, passages, config.neighbours), ranked=ranked)
