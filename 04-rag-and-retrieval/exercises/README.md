# RAG & Retrieval - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and numpy only, run in under 10 seconds each, use fixed seeds, and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_ir_metrics.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_ir_metrics.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_ir_metrics.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [semantic search](../../12-coding-challenges/08_semantic_search_rag.py), [chunking](../../12-coding-challenges/09_text_chunking.py), [BM25 + RRF + MMR](../../12-coding-challenges/17_hybrid_search_and_rerank.py).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_ir_metrics.py` | Easy | recall@k, precision@k, MRR, DCG and nDCG with graded relevance, average precision (Part 9) | 30 min |
| 2 | `ex02_infonce.py` | Medium | InfoNCE loss and exact gradients with in-batch and hard negatives, temperature effects (Part 3) | 45 min |
| 3 | `ex03_ivf_index.py` | Medium | k-means, inverted lists, nprobe versus recall and scan cost (Part 4) | 45 min |
| 4 | `ex04_product_quantization.py` | Medium | PQ codebooks, encode and decode, asymmetric distance tables, memory math (Part 4) | 60 min |
| 5 | `ex05_colbert_maxsim.py` | Easy | Late-interaction MaxSim, why mean pooling differs, storage cost (Part 6) | 30 min |
| 6 | `ex06_parent_doc_acl.py` | Medium | Small-to-big retrieval, ACL pre-filtering versus post-filtering, context budgets (Parts 4, 7) | 60 min |
| 7 | `ex07_mini_hnsw.py` | Hard | Layered graph build and beam search, recall versus ef, cost versus brute force (Part 4) | 90 min |

Suggested order: 1, 5, 2, 3, 4, 6, 7.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator only for logarithms.

### P1. BM25 by hand

A corpus has N = 10,000 chunks and avgdl = 100 tokens. The term "throttle" appears in 10 chunks. A chunk of 200 tokens contains it twice. Compute the BM25 contribution of this term with k1 = 1.2, b = 0.75 (Lucene IDF form).

<details><summary>Answer</summary>

IDF = ln(1 + (10,000 - 10 + 0.5) / (10 + 0.5)) = ln(1 + 951.5) = 6.859. TF component = 2 x 2.2 / (2 + 1.2 x (0.25 + 0.75 x 2)) = 4.4 / (2 + 2.1) = 1.073. Contribution = 6.859 x 1.073 = 7.36. The chunk is twice average length, so length normalisation reduces the TF factor from 1.375 (average length) to 1.073.

</details>

### P2. RRF over three lists

Rankings (k = 60): list 1 = a, b, c. List 2 = b, c, d. List 3 = d, a, b. Compute the fused order.

<details><summary>Answer</summary>

b: 1/62 + 1/61 + 1/63 = 0.04840. a: 1/61 + 1/62 = 0.03252. d: 1/63 + 1/61 = 0.03227. c: 1/63 + 1/62 = 0.03200. Order: b, a, d, c. b wins because it appears in all three lists, even though it is first in only one.

</details>

### P3. nDCG with a bad ordering

The system returns four documents with grades 0, 2, 1, 3 in that order. Compute nDCG@4 with linear gain.

<details><summary>Answer</summary>

DCG = 0/1 + 2/1.585 + 1/2 + 3/2.322 = 0 + 1.262 + 0.5 + 1.292 = 3.054. Ideal order 3, 2, 1, 0: 3 + 1.262 + 0.5 + 0 = 4.762. nDCG = 3.054 / 4.762 = 0.641. The best document sitting at rank 4 costs the most.

</details>

### P4. Memory sizing

20M vectors, 1024 dimensions. Compute the size in float32, with PQ at m = 128 and 8 bits per code, and the layer-0 link memory of an HNSW graph with M = 32 (64 links of 4 bytes per node). Which is the dominant cost in the HNSW-over-float32 design?

<details><summary>Answer</summary>

Float32 vectors: 20M x 4,096 B = 81.9 GB. PQ codes: 20M x 128 B = 2.56 GB. HNSW links: 20M x 64 x 4 B = 5.12 GB. For HNSW over raw float32 the vectors dominate (81.9 GB versus 5.1 GB of links). For HNSW over PQ codes the links (5.1 GB) are larger than the codes (2.6 GB), so graph overhead becomes the dominant term. Add rescoring storage if you keep the originals.

</details>

### P5. Choosing nlist

N = 64M vectors and you plan nprobe = 32. What nlist minimises distance computations per query, what is the cost, and what is the speedup over brute force? Name one reason you might choose a different nlist.

<details><summary>Answer</summary>

nlist = sqrt(32 x 64,000,000) = 45,255. Cost = 2 x 45,255 = 90,510 distance computations, about 707x fewer than 64M. Reasons to deviate: very large nlist makes centroid training and imbalance harder, and the centroid-scan term is not free in practice (it can itself use an HNSW index). The recall you need at that nprobe is the real constraint, so check it against exact search.

</details>

### P6. Post-filter sizing

A tenant filter passes 0.5% of vectors. You need 10 results. How large must the unfiltered candidate list be in expectation, and what goes wrong when the tenant has 0.05%?

<details><summary>Answer</summary>

Expected survivors = k' x 0.005, so k' = 2,000 for 10 survivors on average, and more for a safe margin because the count is random. At 0.05% you need ~20,000 candidates, which wipes out the latency benefit of ANN. This is why selective filters need pre-filtering (brute force over the filtered set when it is small) or filter-aware traversal.

</details>

### P7. Chunk coverage

Chunks are 300 tokens with 60 tokens of overlap. Evidence is an 80-token span at a random position. What fraction of such spans fit entirely inside one chunk, and how much overlap would guarantee containment?

<details><summary>Answer</summary>

Stride = 240, so P = (300 - 80) / 240 = 0.917. Guaranteed containment needs overlap >= 80 tokens (stride <= 220), which raises the chunk count by 300/220 = 1.36x versus 300/240 = 1.25x at 60 tokens overlap.

</details>

### P8. Debugging scenario

After a weekend embedder upgrade, recall@10 on the golden set falls from 0.86 to 0.61, but only for queries containing product codes. The index was rebuilt with the new model. List your first four checks in order.

<details><summary>Answer</summary>

1. Confirm query and passage encoders and prefixes match the new model card, and that the same normalisation is applied at index and query time.
2. Check that the BM25 leg still works and that fusion still includes it. A config change that disabled the lexical leg would hurt exactly the code queries.
3. Check chunk text and maximum input length: new tokenisation may truncate chunks with codes at the end.
4. Compare the ANN index against exact search for those queries. If exact recall is fine, the index parameters need retuning. If exact recall is also low, the embedder is the cause and hybrid weighting or fine-tuning is the repair.

Also check that no old and new vectors are mixed in the same index.

</details>

### P9. Permission design

A document's access is revoked at 10:00. At 10:05 a user in the revoked group asks a question. List the places a stale permission could still leak the content, and how to close each.

<details><summary>Answer</summary>

(1) Index metadata still carries the old ACL: propagate ACL changes through the same pipeline as content, with a bounded delay, or check authoritative permissions at query time. (2) Semantic or answer caches return an earlier answer: key caches by permission scope and invalidate on ACL change. (3) Parent text or a context store has its own copy: re-check ACLs at context assembly. (4) Replicas or a BM25 leg lag behind: apply the same filter and invalidation to every leg. (5) Conversation memory containing earlier retrieved text: do not persist retrieved content across turns without ACL checks. Never rely on the prompt to withhold content.

</details>

### P10. Is the improvement real?

You change the chunk size. Recall@10 goes from 0.70 to 0.75 on 150 queries. Is it an improvement?

<details><summary>Answer</summary>

One system's standard error at recall 0.7 over 150 queries is sqrt(0.7 x 0.3 / 150) = 0.037, a 95% interval of about +-0.073. A 5-point change is inside that range for independent samples. Because both systems ran on the same queries, a paired analysis is more powerful: count queries where the new system hit and the old missed versus the reverse, and bootstrap the difference. Add queries if it is still ambiguous.

</details>
