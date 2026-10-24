"""
Ask a question:        python cli.py ask "What are the global implications of ...?"
Score retrieval only:  python cli.py eval
"""

import argparse
import asyncio
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from answer import answer
from corpus import build_corpus
from jev import JevClient
from retriever import RetrievalConfig, retrieve

# Show the top this-many passages in `ask` output, so you can see what just missed the cut.
SHOW_RANKED = 10


async def run_ask(question: str, config: RetrievalConfig) -> None:
    corpus = build_corpus()
    jev = JevClient()
    try:
        start = time.perf_counter()
        result = await retrieve(jev, question, corpus.passages, config)
        retrieval_s = time.perf_counter() - start
    finally:
        await jev.aclose()

    print(f"Top {SHOW_RANKED} of {len(result.ranked)} passages (* = sent to LLM):")
    for rank, s in enumerate(result.ranked[:SHOW_RANKED], 1):
        mark = "*" if rank <= config.top_k else " "
        print(f" {mark}{rank:>2}. {s.passage.id:<4} score={s.score:.2f} sufficient={s.sufficient:.2f} partial={s.partial:.2f} related={s.related:.2f}  {s.passage.text[:60]!r}")
    print(f"Retrieval: {jev.calls} Jev calls, {jev.input_tokens} input tokens, {retrieval_s:.2f}s\n")

    async with AsyncOpenAI() as client:
        print(await answer(client, question, result.passages))


async def run_eval(config: RetrievalConfig) -> None:
    """Retrieval-only check: do the passages handed to the LLM include the question's own source passages?"""
    corpus = build_corpus()
    jev = JevClient()
    try:
        start = time.perf_counter()
        results = await asyncio.gather(*(retrieve(jev, q, corpus.passages, config) for q in corpus.questions))
        elapsed = time.perf_counter() - start
    finally:
        await jev.aclose()

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
    print(f"{jev.calls} Jev calls, {jev.input_tokens} input tokens, {elapsed:.1f}s")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Per-passage Jev retrieval + GPT-5 mini answering")
    parser.add_argument("--top-k", type=int, default=5)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("eval")
    args = parser.parse_args()

    config = RetrievalConfig(args.top_k)
    if args.command == "ask":
        asyncio.run(run_ask(args.question, config))
    else:
        asyncio.run(run_eval(config))


if __name__ == "__main__":
    main()
