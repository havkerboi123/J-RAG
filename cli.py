"""
Ask a question:        python cli.py ask "What are the global implications of ...?"
Score retrieval only:  python cli.py eval
3-page Clef windows:   python cli.py --window 3 eval
On amnesty_qa:         python cli.py --dataset amnesty --window 1 eval
"""

import argparse
import asyncio
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from answer import answer
from corpus import DATASETS, load_corpus
from retriever import ENGINES, RetrievalConfig, calls_per_question, make_ranker, retrieve

# Show the top this-many passages in `ask` output, so you can see what just missed the cut.
SHOW_RANKED = 10


async def run_ask(question: str, config: RetrievalConfig, dataset: str, engine: str) -> None:
    corpus = load_corpus(dataset, images=engine == "clef")
    ranker = make_ranker(engine)
    try:
        start = time.perf_counter()
        result = await retrieve(ranker, question, corpus.passages, config)
        retrieval_s = time.perf_counter() - start
    finally:
        await ranker.aclose()

    print(f"Top {SHOW_RANKED} of {len(result.ranked)} passages (* = sent to LLM):")
    for rank, s in enumerate(result.ranked[:SHOW_RANKED], 1):
        mark = "*" if rank <= config.top_k else " "
        print(f" {mark}{rank:>2}. {s.passage.id:<4} score={s.score:.2f} sufficient={s.sufficient:.2f} partial={s.partial:.2f} related={s.related:.2f}  {s.passage.text[:60]!r}")
    print(f"Sent to LLM (top {config.top_k} + {config.neighbours} neighbour(s) each side): "
          f"{[p.id for p in result.passages]}")
    print(f"Retrieval: {ranker.calls} {engine} calls, {ranker.input_tokens} input tokens, {retrieval_s:.2f}s\n")

    async with AsyncOpenAI() as client:
        print(await answer(client, question, result.passages))


async def run_eval(config: RetrievalConfig, dataset: str, limit: int | None, engine: str) -> None:
    """Retrieval-only check: do the passages handed to the LLM include the question's own source passages?"""
    corpus = load_corpus(dataset, limit, images=engine == "clef")
    per_question = calls_per_question(corpus.passages, config)
    print(f"{len(corpus.questions)} questions x {per_question} calls ({len(corpus.passages)} passages, "
          f"window {config.window}) = {len(corpus.questions) * per_question:,} {engine} calls")
    ranker = make_ranker(engine)
    try:
        start = time.perf_counter()
        results = await asyncio.gather(*(retrieve(ranker, q, corpus.passages, config) for q in corpus.questions))
        elapsed = time.perf_counter() - start
    finally:
        await ranker.aclose()

    n = len(results)
    print(f"Config: {config}, {len(corpus.passages)} passages, {n} questions")
    # Every passage is scored anyway, so recall at other cutoffs comes for free.
    for k in sorted({config.top_k, 3, 5, 10, 15}):
        hits = recall = 0.0
        for result, gold in zip(results, corpus.gold):
            top = {s.passage.id for s in result.ranked[:k]}
            hits += bool(top & gold)
            recall += len(top & gold) / len(gold)
        sent = "  <- sent to LLM" if k == config.top_k else ""
        print(f"  top {k:>2}: hit rate {hits / n:>4.0%}, recall {recall / n:>4.0%}{sent}")
    print(f"{ranker.calls} {engine} calls, {ranker.input_tokens} input tokens, {elapsed:.1f}s")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Per-passage Clef/Jev retrieval + GPT-5 mini answering")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", choices=DATASETS, default="vidore-hr")
    parser.add_argument("--limit", type=int, default=20, help="max questions to evaluate")
    parser.add_argument("--engine", choices=ENGINES, default="clef")
    parser.add_argument("--window", type=int, default=1, help="consecutive pages per Clef call (1 = one page per call)")
    parser.add_argument("--neighbours", type=int, default=1, help="pages either side of each top page also sent to the LLM")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("eval")
    args = parser.parse_args()

    config = RetrievalConfig(args.top_k, args.window, args.neighbours)
    if args.command == "ask":
        asyncio.run(run_ask(args.question, config, args.dataset, args.engine))
    else:
        asyncio.run(run_eval(config, args.dataset, args.limit, args.engine))


if __name__ == "__main__":
    main()
