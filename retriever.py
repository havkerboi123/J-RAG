"""Retrieval with Jev, no embeddings.

Every passage gets its own Jev call, all in parallel. Passages are ranked by score
and the best `top_k` go to the LLM.
"""

import asyncio
from dataclasses import dataclass

from corpus import Passage
from jev import JevClient

# How much "answers the question" counts against "has related info" when ranking.
SUFFICIENT_WEIGHT = 0.6

JEV_QUESTIONS = {
    "sufficient": {
        "type": "noul",
        "instructions": "Does this passage contain enough information to answer the user's question?",
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
    related: float

    @property
    def score(self) -> float:
        return SUFFICIENT_WEIGHT * self.sufficient + (1 - SUFFICIENT_WEIGHT) * self.related


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
    return ScoredPassage(passage, answers["sufficient"]["noul"], answers["related"]["noul"])


async def retrieve(
    jev: JevClient,
    question: str,
    passages: list[Passage],
    config: RetrievalConfig = RetrievalConfig(),
) -> Retrieval:
    scored = await asyncio.gather(*(score_passage(jev, question, p) for p in passages))
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)
    return Retrieval([s.passage for s in ranked[: config.top_k]], ranked)
