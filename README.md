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

