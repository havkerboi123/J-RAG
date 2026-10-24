"""
End-to-end eval: retrieve pages, answer from them, then check both against the dataset.

  Retrieval: did the dataset's relevant pages make our top 5 / top 10?
  Answer:    which key points of the dataset's reference answer does our answer cover?

    python eval_answers.py --limit 1
    python eval_answers.py --limit 20 --window 3 --neighbours 0
    python eval_answers.py --limit 20 --engine jev --dataset amnesty --window 1
"""

import argparse
import asyncio
import json
import math
import statistics
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from answer import answer
from corpus import DATASETS, load_corpus
from retriever import ENGINES, RetrievalConfig, calls_per_question, make_ranker, retrieve

JUDGE_MODEL = "gpt-5-mini"
RESULTS_PATH = "eval_results.json"
MAX_CONCURRENCY = 4  # questions in flight at once; each fans out to one ranker call per page
PRICE_PER_M_INPUT = {"clef": 0.24, "jev": 0.042}  # USD per million input tokens; output is free

JUDGE_INSTRUCTIONS = (
    "You grade a RAG system's answer against a reference answer. "
    "Split the reference answer into its distinct key points (facts or claims). "
    "For each one, decide whether the system answer conveys it; wording can differ, meaning must match. "
    "Then list any points in the system answer that the reference doesn't mention, "
    "and say whether the system answer contradicts the reference anywhere."
)


class Judgement(BaseModel):
    reference_points: list[str]
    covered: list[str]  # reference points the answer conveys
    missing: list[str]  # reference points the answer doesn't convey
    extra: list[str]  # points in the answer that aren't in the reference
    contradicts_reference: bool


async def judge(client: AsyncOpenAI, question: str, reference: str, response: str) -> Judgement:
    result = await client.responses.parse(
        model=JUDGE_MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {response}",
        text_format=Judgement,
    )
    return result.output_parsed


def ndcg_at_10(ranked_ids: list[str], gold: set[str]) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank, pid in enumerate(ranked_ids[:10], 1) if pid in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    return dcg / ideal


async def run(config: RetrievalConfig, dataset: str, limit: int | None, engine: str) -> None:
    corpus = load_corpus(dataset, limit, images=engine == "clef")
    per_question = calls_per_question(corpus.passages, config)
    print(f"{len(corpus.questions)} questions x {per_question} calls ({len(corpus.passages)} pages, "
          f"window {config.window}) = {len(corpus.questions) * per_question:,} {engine} calls")
    openai_client = AsyncOpenAI()
    ranker = make_ranker(engine)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def evaluate_one(i: int) -> dict:
        question, reference, gold = corpus.questions[i], corpus.references[i], corpus.gold[i]
        async with sem:
            result = await retrieve(ranker, question, corpus.passages, config)
            response = await answer(openai_client, question, result.passages)
            verdict = await judge(openai_client, question, reference, response)
        ranked_ids = [s.passage.id for s in result.ranked]
        n_points = len(verdict.reference_points)
        return {
            "question": question,
            "reference": reference,
            "response": response,
            "gold_pages": sorted(gold),
            "top_pages": ranked_ids[:10],
            "llm_pages": [p.id for p in result.passages],
            "hit@5": float(bool(set(ranked_ids[:5]) & gold)),
            "recall@5": len(set(ranked_ids[:5]) & gold) / len(gold),
            "recall@10": len(set(ranked_ids[:10]) & gold) / len(gold),
            "ndcg@10": ndcg_at_10(ranked_ids, gold),
            "coverage": len(verdict.covered) / n_points if n_points else 0.0,
            "contradicts": verdict.contradicts_reference,
            "judgement": verdict.model_dump(),
        }

    try:
        start = time.perf_counter()
        rows = await asyncio.gather(*(evaluate_one(i) for i in range(len(corpus.questions))))
        elapsed = time.perf_counter() - start
    finally:
        await ranker.aclose()
        await openai_client.close()

    metrics = ["hit@5", "recall@5", "recall@10", "ndcg@10", "coverage"]
    print(f"\n{'#':>2}  " + " ".join(f"{m:>9}" for m in metrics) + "  contra  question")
    for i, row in enumerate(rows):
        cells = " ".join(f"{row[m]:>9.2f}" for m in metrics)
        print(f"{i:>2}  {cells}  {'yes' if row['contradicts'] else 'no':>6}  {row['question'][:50]}")

    print(f"\nConfig: {config}, dataset={dataset}, engine={engine}, {len(rows)} questions")
    for m in metrics:
        print(f"  {m:<10} {statistics.mean(row[m] for row in rows):.2f}")
    print(f"  contradicts reference: {sum(row['contradicts'] for row in rows)}/{len(rows)}")

    cost = ranker.input_tokens * PRICE_PER_M_INPUT[engine] / 1e6
    print(f"\n{ranker.calls:,} {engine} calls, {ranker.input_tokens:,} input tokens, "
          f"${cost:.2f} (${cost / len(rows):.2f} per question); GPT answering and grading not included")
    print(f"{elapsed:.0f}s total")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"config": vars(config), "dataset": dataset, "engine": engine, "rows": rows}, f, indent=2)
    print(f"Per-question answers, page lists and grading saved to {RESULTS_PATH}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval + answer eval against the dataset's labels")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", choices=DATASETS, default="vidore-hr")
    parser.add_argument("--limit", type=int, default=20, help="max questions to evaluate")
    parser.add_argument("--engine", choices=ENGINES, default="clef")
    parser.add_argument("--window", type=int, default=1, help="consecutive pages per Clef call (1 = one page per call)")
    parser.add_argument("--neighbours", type=int, default=1, help="pages either side of each top page also sent to the LLM")
    args = parser.parse_args()

    asyncio.run(run(RetrievalConfig(args.top_k, args.window, args.neighbours), args.dataset, args.limit, args.engine))


if __name__ == "__main__":
    main()
