"""Retrieval with a System One decision model (Clef or Jev), no embeddings.

Every passage gets its own call, all in parallel. Passages that have a page image are judged
from the image alone; the rest from their text. Passages are ranked by score and the best
`top_k` go to the LLM.
"""

import asyncio
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


def make_ranker(engine: str) -> Ranker:
    return ClefClient() if engine == "clef" else JevClient()


@dataclass
class RetrievalConfig:
    top_k: int = 5


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
    passages: list[Passage]
    ranked: list[ScoredPassage]  # every passage, best first

    @property
    def sufficient(self) -> float:
        """Highest confidence that a passage sent to the LLM answers the question."""
        return max((s.sufficient for s in self.ranked[: len(self.passages)]), default=0.0)


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


async def retrieve(
    ranker: Ranker,
    question: str,
    passages: list[Passage],
    config: RetrievalConfig = RetrievalConfig(),
) -> Retrieval:
    scored = await asyncio.gather(*(score_passage(ranker, question, p) for p in passages))
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    return Retrieval([s.passage for s in ranked[: config.top_k]], ranked)
