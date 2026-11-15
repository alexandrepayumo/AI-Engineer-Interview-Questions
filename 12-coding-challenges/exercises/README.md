# Coding Challenges - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). These seven exercises are new: none repeats one of the 19 challenge files in the parent folder. They cover the skills those files assume (vectorising, numerical stability, batch sampling, masks, streaming matchers) and the two formats the folder README says it does not rehearse (async fan-out with safe retries, and find-the-planted-bugs). Everything uses Python 3.12 and numpy, runs in under 10 seconds, uses fixed seeds and needs no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_vectorise_drills.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_vectorise_drills.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_vectorise_drills.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Time yourself like a real round: 30 to 45 minutes for the medium exercises, talking aloud about shapes, complexity and failure modes as you go.

Exercise 7 is different. The buggy code is already in the file, and the stub's `fixed_attention` and `fixed_block` raise `NotImplementedError` until you write them. The tests double as a lesson in which cheap check exposes which bug.

Further practice that overlaps this topic but is not repeated here: [attention](../01_attention.py), [sampling](../03_sampling.py), [BPE](../02_bpe_tokenizer.py), [rate limiting and retry](../11_rate_limiter_and_retry.py), [streaming parser](../13_streaming_parser.py), [speculative decoding](../19_speculative_decoding.py).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_vectorise_drills.py` | Easy | Broadcasting, einsum, head split and merge, take_along_axis, masked mean, checked against loop references (Part 2) | 45 min |
| 2 | `ex02_stable_math.py` | Medium | Log-sum-exp, log-softmax, softplus, logsigmoid, cross-entropy and its gradient on extreme inputs, finite-difference check (Parts 2, 9) | 45 min |
| 3 | `ex03_topp_batch.py` | Medium | Vectorised top-p for a batch with per-row p, boundary rule, inverse-CDF sampling, statistical test (Part 4) | 50 min |
| 4 | `ex04_attention_masks.py` | Medium | Sliding-window, sink and strided masks, masked attention with empty rows, receptive field growth (Part 3) | 50 min |
| 5 | `ex05_trie_tokenizer.py` | Medium | Trie longest match with backtracking, streaming encoder that holds back undecidable text (Part 5) | 50 min |
| 6 | `ex06_async_fanout.py` | Hard | Semaphore cap, timeouts, full-jitter backoff, retryable errors only, idempotency keys, budget cancellation, no orphaned tasks (Parts 7, 10) | 75 min |
| 7 | `ex07_planted_bugs.py` | Hard | Finding four planted bugs in an attention block, and the named checks that catch each (Parts 9, 10) | 60 min |

Suggested order: 1, 2, 4, 3, 5, 7, 6.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator only for powers and logarithms.

### P1. Broadcasting shapes

Give the result shape, or say why it fails: (a) `(8, 1, 5) + (3, 1)`. (b) `(4, 1, 6) * (5, 6)`. (c) `(2, 3) + (3, 2)`. (d) `(B, h, T, T) + (T, T)`. (e) `(B, T) [..., None] == arange(V)`.

<details><summary>Answer</summary>

Align shapes from the right and pad the shorter one with 1s on the left. A dimension pair is compatible when the sizes are equal or one of them is 1. (a) `(3, 1)` becomes `(1, 3, 1)`, so the result is `(8, 3, 5)`. (b) `(5, 6)` becomes `(1, 5, 6)`, result `(4, 5, 6)`. (c) The trailing sizes 3 and 2 differ and neither is 1, so it raises `ValueError`. (d) Result `(B, h, T, T)`: the mask is shared across batch and heads, which is how an additive causal mask is applied. (e) `(B, T, 1) == (V,)` gives `(B, T, V)`, which is the one-hot construction.

</details>

### P2. Where does exp overflow?

(a) At roughly what input does `exp` overflow in float32 and in float64? (b) What does the naive softmax of `[1000, 1001]` return? (c) What does the stable version return, and what are the exact values?

<details><summary>Answer</summary>

(a) exp overflows where the result exceeds the largest finite value: ln(3.4e38) is about 88.7 for float32, and ln(1.8e308) is about 709.8 for float64. (b) `exp(1000)` is `inf`, so the naive version computes `inf / inf = nan` for both entries. (c) Subtract the max (1001) to get `[-1, 0]`. Then exp gives `[0.3679, 1]`, the sum is 1.3679, and the probabilities are `[0.2689, 0.7311]`. The shift changes nothing mathematically because the factor `exp(-m)` cancels between numerator and denominator.

</details>

### P3. Choosing gamma for speculative decoding

The draft accepts each token independently with probability a = 0.8. Drafting one token costs c = 0.1 of a target forward pass. With gamma drafted tokens, one step costs 1 + gamma * c target passes and yields (1 - a^(gamma+1)) / (1 - a) tokens on average. Compute the speedup over plain decoding for gamma = 3, 6 and 9. Which gamma is best of those, and why does it not keep improving?

<details><summary>Answer</summary>

gamma = 3: tokens = (1 - 0.8^4) / 0.2 = 2.952, cost = 1.3, speedup 2.27. gamma = 6: tokens = (1 - 0.8^7) / 0.2 = 3.951, cost = 1.6, speedup 2.47. gamma = 9: tokens = (1 - 0.8^10) / 0.2 = 4.463, cost = 1.9, speedup 2.35. The optimum is near gamma = 6 (2.47). Expected tokens saturate at 1 / (1 - a) = 5 while the draft cost grows linearly, so past some gamma every extra drafted token costs more than it adds. The real optimum shifts with a and c, which is why serving systems tune or adapt gamma. This model assumes independent acceptance, which real text violates.

</details>

### P4. Receptive field of a sliding window

A decoder has 32 layers, each with a causal sliding window of 4,096 tokens (the token itself plus 4,095 before it). What is the furthest back that information can travel from the last position to the first, in principle? Why is the practical reach much shorter?

<details><summary>Answer</summary>

Each layer extends reach by 4,095 positions, so the theoretical limit is 32 x 4,095 = 131,040 tokens. In practice, information passed through many layers is mixed, attenuated and competes with local signal, so effective recall of a specific fact far outside the window is much weaker than the arithmetic suggests. If a task needs exact retrieval of an early fact, put some layers or heads on full attention, or add retrieval. Full-attention layers (as in hybrid designs) solve it directly.

</details>

### P5. KV cache memory

A model has 32 layers, 8 KV heads (grouped-query attention), head dimension 128, and stores the cache in fp16. How many bytes per token? How much for one 32,768-token sequence, and for a batch of 8? What changes if the cache is stored in 8-bit?

<details><summary>Answer</summary>

Per token: 2 (K and V) x 32 layers x 8 heads x 128 x 2 bytes = 131,072 bytes (128 KiB). One 32,768-token sequence: 131,072 x 32,768 = 4 GiB. Batch of 8: 32 GiB, before weights. An 8-bit cache halves each figure (2 GiB, 16 GiB) at some accuracy risk. With full multi-head attention (32 KV heads) the numbers would be 4 times larger, which is why GQA matters at long context.

</details>

### P6. Token bucket by hand

A token bucket has capacity 10 and refills at 2 tokens per second. It starts full. Fifteen one-token requests arrive at t = 0. (a) How many pass? (b) A client retries the rejected ones at t = 2.0 s, all at once. How many pass? (c) What is the maximum sustained rate, and what is the largest burst?

<details><summary>Answer</summary>

(a) 10 pass and 5 are rejected. (b) After 2 s the bucket holds min(10, 0 + 2 x 2) = 4 tokens, so 4 pass and 1 is rejected again. (c) Sustained rate is 2 requests per second. The largest burst is the capacity, 10, and it is only available after the bucket has been idle for at least 5 s. Note the synchronised retry: if many clients retry on the same schedule they re-collide, which is why backoff needs jitter.

</details>

### P7. Retry amplification

A user request passes through three services in a chain. Each calls the next with up to 3 attempts (1 try and 2 retries), and the last service is failing completely. How many calls reach the last service per user request? What mitigations apply?

<details><summary>Answer</summary>

Each layer multiplies attempts: 3 x 3 x 3 = 27 calls reach the failing service for one user request. That is the retry storm pattern, and it makes recovery harder. Mitigations: retry at one layer only (usually the edge), use a retry budget (for example retries limited to about 10% of recent calls), use circuit breakers so a failing dependency is skipped fast, honour `Retry-After` and propagate a deadline so nobody retries after the caller has given up, and add jitter so retries do not synchronise.

</details>

### P8. Name the bug from the symptom

For each symptom, name the most likely planted bug. (a) Training loss turns to NaN only on batches with large logits. (b) Output at position 3 changes when you edit token 9, but shapes are fine. (c) Perplexity is far worse than expected with 8 heads, fine with 1 head. (d) After adding the MLP, the output ignores the input entirely. (e) Greedy decoding with the KV cache differs from decoding without it after about 20 tokens.

<details><summary>Answer</summary>

(a) Softmax or cross-entropy without max subtraction (or a log of a probability that underflowed to 0). (b) A causal mask that is applied after softmax, applied to the wrong axis, or missing. Information leaks from the future through the softmax denominator or through the unmasked scores. (c) A head-merge or head-split bug (reshape without transpose), or scaling by sqrt(d_model) instead of sqrt(d_head). Both vanish with one head. (d) The residual connection around the MLP (or attention) is dropped, so the block output replaces the stream. (e) A cache bug: wrong position offset for the new token, a cache not copied between sequences or beams, or keys cached before positional rotation is applied inconsistently. Test by comparing full-sequence logits at every step.

</details>

### P9. A test plan for a streaming SSE parser

You must test a parser that receives arbitrary byte chunks. List five tests that together would catch most bugs, one sentence each.

<details><summary>Answer</summary>

1. Equivalence under chunking: for a fixed sample stream, parse it whole and split at every possible byte position, and assert the same events. 2. One byte at a time, which is the hardest chunking and catches split `\r\n` and split multi-byte characters. 3. A property test over random chunk sizes with a fixed seed. 4. Edge cases: comment lines, `data:` with and without a space, multi-line data joined by newlines, an event with no trailing blank line at the end of the stream (should it flush or not, decide and test). 5. Malformed input: a partial JSON argument at stream end must produce a clear error or a repaired partial value, never a crash or a silently truncated call.

</details>

### P10. Unbiased pass@k by hand

For one problem you ran n = 20 samples and c = 5 passed. Compute the unbiased pass@3 and compare it with the naive 1 - (1 - 5/20)^3. Which is larger, and why?

<details><summary>Answer</summary>

Unbiased: 1 - C(15, 3) / C(20, 3) = 1 - 455 / 1140 = 0.601. Naive: 1 - 0.75^3 = 0.578. The unbiased estimator is the exact probability that at least one of 3 samples drawn without replacement from your 20 contains a pass. The naive one treats the 3 draws as independent with replacement and plugs in the noisy estimate 0.25, which underestimates in expectation for small n. The gap shrinks as n grows. Report the unbiased form.

</details>
