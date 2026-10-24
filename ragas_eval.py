"""
End-to-end eval: Jev retrieval -> GPT-5 mini answer -> ragas scores against the dataset's reference answers.

    python ragas_eval.py
    python ragas_eval.py --top-k 10
"""

import argparse
import asyncio
import json
import statistics
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, FactualCorrectness, Faithfulness

from answer import answer
from corpus import build_corpus
from jev import JevClient
from retriever import RetrievalConfig, retrieve

JUDGE_MODEL = "gpt-5-mini"
JUDGE_MAX_TOKENS = 16000
RESULTS_PATH = "ragas_results.json"
MAX_CONCURRENCY = 8
METRICS = ["gold_recall", "context_recall", "faithfulness", "factual_correctness"]


async def score(name: str, coro) -> float | None:
    """Run one ragas metric; a failure costs that cell, not the whole run."""
    try:
        return float((await coro).value)
    except Exception as e:
        print(f"  {name} failed: {e!r}")
        return None


async def run(config: RetrievalConfig) -> None:
    corpus = build_corpus()
    openai_client = AsyncOpenAI()
    # GPT-5 mini spends output tokens on reasoning, so ragas's default 1024 cap cuts off its verdicts.
    judge = llm_factory(JUDGE_MODEL, client=openai_client, max_tokens=JUDGE_MAX_TOKENS)
    faithfulness = Faithfulness(llm=judge)
    context_recall = ContextRecall(llm=judge)
    factual_correctness = FactualCorrectness(llm=judge)
    jev = JevClient()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def evaluate_one(i: int) -> dict:
        question, reference, gold = corpus.questions[i], corpus.references[i], corpus.gold[i]
        async with sem:
            result = await retrieve(jev, question, corpus.passages, config)
            response = await answer(openai_client, question, result.passages)
            contexts = [p.text for p in result.passages]
            faith, recall, correct = await asyncio.gather(
                score("faithfulness", faithfulness.ascore(user_input=question, response=response, retrieved_contexts=contexts)),
                score("context_recall", context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference)),
                score("factual_correctness", factual_correctness.ascore(response=response, reference=reference)),
            )
        selected = {p.id for p in result.passages}
        return {
            "question": question,
            "response": response,
            "reference": reference,
            "passage_ids": [p.id for p in result.passages],
            "jev_sufficient": result.sufficient,
            "gold_recall": len(selected & gold) / len(gold),
            "context_recall": recall,
            "faithfulness": faith,
            "factual_correctness": correct,
        }

    try:
        start = time.perf_counter()
        rows = await asyncio.gather(*(evaluate_one(i) for i in range(len(corpus.questions))))
        elapsed = time.perf_counter() - start
    finally:
        await jev.aclose()
        await openai_client.close()

    print(f"\n{'#':>2}  {'question':<45} {'gold':>5} {'ctxrec':>6} {'faith':>6} {'correct':>7} {'jev_suff':>8}")
    for i, row in enumerate(rows):
        cells = [row[m] for m in METRICS] + [row["jev_sufficient"]]
        print(f"{i:>2}  {row['question'][:45]:<45} " + " ".join(f"{'-' if c is None else f'{c:.2f}':>{w}}" for c, w in zip(cells, (5, 6, 6, 7, 8))))

    print(f"\nConfig: {config}")
    for m in METRICS:
        values = [row[m] for row in rows if row[m] is not None]
        mean = f"{statistics.mean(values):.2f}" if values else "n/a"
        print(f"  {m:<20} {mean}  ({len(values)}/{len(rows)} scored)")
    print(f"{jev.calls} Jev calls, {jev.input_tokens} Jev input tokens, {elapsed:.1f}s total")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"config": vars(config), "rows": rows}, f, indent=2)
    print(f"Per-question answers and scores saved to {RESULTS_PATH}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Jev + GPT-5 mini RAG, scored with ragas")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(run(RetrievalConfig(args.top_k)))


if __name__ == "__main__":
    main()
