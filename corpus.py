from dataclasses import dataclass

from load_dataset import load_eval_dataset


@dataclass(frozen=True)
class Passage:
    id: str
    text: str


@dataclass
class Corpus:
    passages: list[Passage]
    questions: list[str]
    references: list[str]
    # Ids of the passages each question's contexts came from, used to score retrieval.
    gold: list[set[str]]


def build_corpus() -> Corpus:
    samples = load_eval_dataset().samples

    ids_by_text: dict[str, str] = {}
    gold = []
    for sample in samples:
        gold.append(
            {ids_by_text.setdefault(text, f"p{len(ids_by_text)}") for text in sample.retrieved_contexts}
        )

    return Corpus(
        passages=[Passage(id=pid, text=text) for text, pid in ids_by_text.items()],
        questions=[s.user_input for s in samples],
        references=[s.reference for s in samples],
        gold=gold,
    )
