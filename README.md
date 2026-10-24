# J-RAG

A hobby project for playing around with [Jev](https://typesafe.ai), TypeSafe AI's "System One" decision model, as the retrieval layer of a RAG system.

The question I wanted to answer: **can you build RAG without embeddings or a vector database?** Instead of embedding passages and doing a similarity search, every passage is shown to Jev alongside the user's question, and Jev decides how useful it is. The best passages go to an LLM, which writes the answer.

This is an experiment, not a production system.

## Architecture

```
                      user question
                            │
                            ▼
   ┌─────────────────────────────────────────────────┐
   │  Jev: one call per passage, all in parallel     │
   │                                                 │
   │  state:     { question, passage }               │
   │  questions: sufficient? partial? related?       │
   └─────────────────────────────────────────────────┘
                            │  three probabilities per passage
                            ▼
          score = 0.4·sufficient + 0.4·partial + 0.2·related
                            │
                            ▼
              sort all passages, keep the top 5
                            │
                            ▼
   ┌─────────────────────────────────────────────────┐
   │  GPT-5 mini: answer from the top 5 passages,    │
   │  cite them like [2], say so if they're not      │
   │  enough                                         │
   └─────────────────────────────────────────────────┘
                            │
                            ▼
                          answer
```

### Retrieval with Jev

Jev doesn't take a free-text prompt. Each call sends structured JSON: a `state` (the data to judge) and a set of typed `questions`. Every passage gets its own call:

```json
{
  "model": "jev-latest",
  "state": { "question": "Where was COP15 held in 2022?", "passage": "..." },
  "questions": {
    "sufficient": { "type": "noul", "instructions": "Does this passage contain enough information to answer the user's question?" },
    "partial":    { "type": "noul", "instructions": "Does this passage contain at least part of the information needed ...?" },
    "related":    { "type": "noul", "instructions": "Does this passage contain information related to the user's question?" }
  }
}
```

Jev answers each yes/no (`noul`) question with a probability between 0 and 1. These are combined into one ranking score:

| Jev answer | Weight | Why it's there |
|---|---|---|
| `sufficient` | 0.4 | The passage answers the question on its own |
| `partial` | 0.4 | The passage holds one piece of the answer, e.g. one year in "compare sales in 2014 and 2016" |
| `related` | 0.2 | Breaks ties between passages that are only on topic |

This gives roughly three tiers: about 1.0 for a passage that answers the question alone, about 0.6 for one piece of a multi-part answer, and about 0.2 for passages that are only on topic. Change the weights in `WEIGHTS` in `retriever.py`.

All passages are scored at the same time (`asyncio.gather`, at most 16 requests in flight). They're sorted by score, and the top `k` (default 5) go to the LLM.

### How we got here

The first design grouped passages into collections of 20, chose the best collection, split it into groups of 5, and chose the best group. That needed fewer Jev calls, but Jev was scoring whole groups, so a strong passage could lose because of weak neighbours. Its recall was 35% at 5 passages. Scoring every passage individually raised recall to 97% at the same 5.

## Results so far

The dataset is [amnesty_qa](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa) (`english_v3`): 20 questions, each with 3 source passages. The corpus is all 60 of those passages pooled together, so for each question Jev has to find its 3 among 60, and the other 57 act as distractors.

These numbers come from top 5, with only the `sufficient` and `related` questions. They were measured before `partial` was added.

| Metric | Score | Measures |
|---|---|---|
| Gold recall | **0.97** | Share of each question's 3 source passages that made the top 5 |
| Context recall (ragas) | **0.99** | Whether the top 5 contained what the reference answer needs |
| Faithfulness (ragas) | **0.96** | Whether the answer's claims are supported by the passages |
| Factual correctness (ragas) | **0.41** | How well the answer matches the reference answer |

Retrieval is close to perfect on this dataset. Factual correctness is low mainly because GPT's answers are longer and more detailed than the short reference answers, along with some noisy grading and one question where the passages contradict each other.

Cost: 60 Jev calls per question, about 26k input tokens, roughly $0.001 per question at $0.042 per million input tokens.

**Caveats:** only 20 questions, one run, GPT-5 mini grading its own answers, and a corpus small enough to fit in a single LLM prompt. This dataset is easy for retrieval; a large corpus with overlapping topics would be the real test.

## Files

| File | What it does |
|---|---|
| `load_dataset.py` | Loads the amnesty_qa eval set as a ragas `EvaluationDataset` |
| `corpus.py` | Pools every question's passages into one corpus and records each question's source ("gold") passages for scoring |
| `jev.py` | Small async client for Jev's `/v1/systemone` endpoint, with retries and a concurrency limit |
| `retriever.py` | The Jev questions, scoring weights and top-k selection |
| `answer.py` | Generates the answer with GPT-5 mini from the selected passages |
| `cli.py` | `ask` runs one question end to end; `eval` scores retrieval only |
| `ragas_eval.py` | Full end-to-end run over all 20 questions, scored with ragas |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```
TYPESAFE_API_KEY=...
OPENAI_API_KEY=...
```

## Usage

Ask one question. This prints the top 10 passages with their Jev scores, then the answer:

```bash
uv run python cli.py ask "Where was COP15 held in 2022?"
```

Score retrieval only. Each question's source passages are checked against the top 3, 5, 10 and 15; this calls Jev but not GPT:

```bash
uv run python cli.py eval
```

Run the full ragas eval. This calls Jev and GPT, and saves per-question results to `ragas_results.json`:

```bash
uv run python ragas_eval.py
```

Both `cli.py` and `ragas_eval.py` accept `--top-k` to change how many passages go to the LLM. On `cli.py` it goes before the command, e.g. `cli.py --top-k 10 eval`.

## Limitations and ideas

- **Doesn't scale as is.** Jev reads every passage for every question, so cost and call count grow with the size of the corpus. A cheap first stage such as BM25 could narrow millions of passages to about 50 for Jev to rerank.
- **Fewer calls.** Batching about 10 passages per call, with a separate question for each passage, would cut the repeated question-and-instructions overhead while keeping per-passage scores.
- **Always sends exactly k passages.** A minimum score, or one final Jev call asking "are these enough together?", would stop weak passages from filling the slots.
- **Comparison questions.** The `partial` question helps. Splitting a question into sub-questions before retrieval would help more.
- **Text only.** Jev doesn't accept images. Cloudflare's Clef uses a compatible API and does, or images could be converted to text descriptions when documents are loaded.
- **Compare against established rerankers** such as Cohere Rerank or Qwen3-Reranker on the same eval.
