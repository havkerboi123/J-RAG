# Experiment results

One entry per experiment: what the architecture was, and what happened. Newest at the bottom.

## amnesty_qa (20 questions, 60 text passages)

1. **Jev, collections → groups (top 1 collection of 20, then top 1 group of 5).** 90% of questions got a correct passage into the 5 sent to the LLM, but only 35% of correct passages got through; scoring whole groups let strong passages lose to weak neighbours.
2. **Same, keeping top 2 collections and top 3 groups (15 passages).** Hit rate 100%, recall 85%.
3. **Jev, one call per passage (60 calls), top 5 to GPT-5 mini.** Recall 97%, ragas context recall 0.99, faithfulness 0.96, factual correctness 0.41 (0.52 on a rerun, so the grader is noisy). About $0.001 of Jev per question.

## ViDoRe V3 HR (1,110 page images, 318 questions)

4. **Clef, one page image per call (1,110 calls), GPT-5 mini answering from OCR text; 1 question (women's workforce drivers).** Answer covered all 6 points of the reference, but only 2 of the 14 relevant pages made the top 10 (NDCG@10 ≈ 0.18). ~1,170 tokens per page, $0.31 and 154s per question.
5. **Same retrieval, GPT-5 mini answering from page images; 1 question (Romania youth employment 2020).** Retrieval perfect (relevant page ranked #1, NDCG@10 1.00), but the answer was wrong: 35.5% instead of 24.6%, read from the wrong section of a dense table in a 1024px image. The grader still gave 0.80 coverage, so it needs fixing.
6. **Clef, 3-page windows overlapping by one page (552 calls), each page scored separately within its window; 1 question (Romania, same as #5).** Relevant page dropped from #1 to #2 (NDCG@10 1.00 → 0.63, still in top 5). The answer was correct this time (24.6%), but GPT's input was the same small image, so that's likely luck rather than the windows. ~3,430 tokens per call, $0.45 and 227s per question (vs $0.31 and 156s for one page per call).
