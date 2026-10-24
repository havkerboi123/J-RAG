# J-RAG

playing around with [Jev](https://typesafe.ai), TypeSafe AI's "System One" decision model, as the retrieval layer of a RAG system.

the question I wanted to answer: **can you build RAG without embeddings or a vector database?** Instead of embedding passages and doing a similarity search, every passage is shown to Jev alongside the user's question, and Jev decides how useful it is. The best passages go to an LLM, which writes the answer.



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

## Experiment

tested on [amnesty_qa](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa): 20 questions, each with 3 source passages. all 60 passages pooled into one corpus, so Jev has to find each question's 3 among 60. answers scored with [ragas](https://docs.ragas.io).

| metric | score |
|---|---|
| source passages in top 5 | 0.97 |
| context recall | 0.99 |
| faithfulness | 0.96 |

 ~$0.001 of Jev per question

