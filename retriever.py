"""Retrieval with Jev, no embeddings.

Every passage gets its own Jev call, all in parallel. Passages are ranked by score
and the best `top_k` go to the LLM.
"""

import asyncio
from dataclasses import dataclass

from corpus import Passage
from jev import JevClient

# How much each Jev answer counts toward a passage's ranking score. A passage that answers the
# question alone scores ~1.0, one holding a piece of the answer (e.g. one year of a comparison)
# ~0.6, and one that is merely on-topic ~0.2.
WEIGHTS = {"sufficient": 0.4, "partial": 0.4, "related": 0.2}

JEV_QUESTIONS = {
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
        """Jev's highest confidence that a passage sent to the LLM answers the question."""
        return max((s.sufficient for s in self.ranked[: len(self.passages)]), default=0.0)


async def score_passage(jev: JevClient, question: str, passage: Passage) -> ScoredPassage:
    answers = await jev.ask({"question": question, "passage": passage.text}, JEV_QUESTIONS)
    return ScoredPassage(
        passage,
        sufficient=answers["sufficient"]["noul"],
        partial=answers["partial"]["noul"],
        related=answers["related"]["noul"],
    )


async def retrieve(
    jev: JevClient,
    question: str,
    passages: list[Passage],
    config: RetrievalConfig = RetrievalConfig(),
) -> Retrieval:
    scored = await asyncio.gather(*(score_passage(jev, question, p) for p in passages))
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    return Retrieval([s.passage for s in ranked[: config.top_k]], ranked)
