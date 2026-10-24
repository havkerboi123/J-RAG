from dataclasses import dataclass

from load_dataset import load_eval_dataset


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    image: str | None = None  # page image as a data URL; when set, Clef judges the page from this alone


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


DATASETS = ["amnesty", "vidore-hr"]


def load_corpus(dataset: str, limit: int | None = None, images: bool = False) -> Corpus:
    """Load a corpus by name, keeping at most `limit` questions. `images` attaches page images where the dataset has them."""
    if dataset == "vidore-hr":
        from vidore import build_vidore_corpus  # imported here: vidore.py imports this module

        return build_vidore_corpus(limit, images=images)

    corpus = build_corpus()
    if limit is not None:
        corpus.questions = corpus.questions[:limit]
        corpus.references = corpus.references[:limit]
        corpus.gold = corpus.gold[:limit]
    return corpus
