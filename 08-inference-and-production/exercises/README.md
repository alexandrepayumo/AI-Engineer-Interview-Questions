# Inference, Serving & Production LLM Systems - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and numpy only, run in under 10 seconds each, use fixed seeds, and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_roofline.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`. Small private helpers that start with an underscore are already provided where the design is not the point of the exercise.
3. Run it from this folder: `python ex01_roofline.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_roofline.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [KV cache](../../12-coding-challenges/06_kv_cache.py), [rate limiter and retry](../../12-coding-challenges/11_rate_limiter_and_retry.py), [semantic cache](../../12-coding-challenges/16_semantic_cache.py), [speculative decoding](../../12-coding-challenges/19_speculative_decoding.py).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_roofline.py` | Easy | Arithmetic intensity, ridge point, decode and prefill step time, crossover batch, KV traffic (Parts 1, 2) | 30 min |
| 2 | `ex02_paged_kv_cache.py` | Hard | Block tables, ref counts, chained-hash prefix reuse, LRU eviction, copy-on-write (Part 4) | 90 min |
| 3 | `ex03_quantization.py` | Medium | Symmetric and asymmetric INT8, group-wise INT4, outliers, bits per weight, W8A8 matmul (Part 6) | 60 min |
| 4 | `ex04_batching_sim.py` | Medium | Static versus continuous batching, convoy effect, throughput and latency percentiles (Parts 3, 5) | 60 min |
| 5 | `ex05_capacity_planner.py` | Medium | KV-limited and ITL-limited batch, replica capacity, replicas for a QPS target, $ per million tokens (Part 9) | 60 min |
| 6 | `ex06_circuit_breaker.py` | Medium | Breaker state machine with an injected clock, full-jitter backoff, fallback routing (Part 10) | 60 min |
| 7 | `ex07_queueing.py` | Medium | Little's law, M/M/1 and Erlang C, pooling, event simulation (Parts 3, 9) | 60 min |

Suggested order: 1, 3, 4, 7, 5, 6, 2.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator for arithmetic. Hardware assumptions follow the deep dive: ~989 TFLOP/s dense FP16, ~3.35 TB/s HBM, 80 GB per GPU, unless a problem states otherwise.

### P1. KV sizing

A model has 48 layers, 8 KV heads, head dimension 128 and FP16 KV. Compute the KV bytes per token and per 32K-token sequence. On an 80 GB GPU holding 30 GB of weights with 90% memory utilisation, how many 32K sequences fit?

<details><summary>Answer</summary>

Per token: 2 x 48 x 8 x 128 x 2 = 196,608 bytes (192 KiB). Per 32,768-token sequence: 6.44 GB. KV budget: 0.9 x 80 - 30 = 42 GB. 42 / 6.44 = 6.5, so **6 sequences**. A large context window does not mean large concurrency: six users at 32K fill the card, which is why long-context products price and limit differently.

</details>

### P2. Decode ceiling and tensor parallelism

A 34B model is stored at INT8 weights (1 byte per parameter) on GPUs with 2 TB/s bandwidth. What is the batch-1 decode ceiling on one GPU, on two GPUs with tensor parallelism if communication adds 2 ms per step, and why is the second number not simply doubled?

<details><summary>Answer</summary>

One GPU: 34e9 / 2e12 = 17 ms per token, ~59 tokens per second. Two GPUs: each reads 17 GB at 2 TB/s = 8.5 ms, plus 2 ms of all-reduce time = 10.5 ms, ~95 tokens per second, a 1.6x gain, not 2x. The all-reduce time is fixed per step and does not shrink as more GPUs are added, so scaling is sub-linear and gets worse with TP degree.

</details>

### P3. Ridge point and quantisation

A GPU has 1,000 TFLOP/s and 4 TB/s. What is the ridge point, and at what batch size does weight-only decode become compute-bound with FP16 weights and with INT4 weights (0.5 bytes)? Ignore KV.

<details><summary>Answer</summary>

Ridge = 1e15 / 4e12 = 250 FLOPs per byte. Crossover batch = ridge x bytes_per_weight / 2: FP16 gives 250 x 2 / 2 = 250, INT4 gives 250 x 0.5 / 2 = 62.5. With INT4 the matmul becomes compute-bound at a batch about four times smaller, since every FLOP is paid for with a quarter of the bytes. Below that batch INT4 decode is about 4 times faster than FP16, above it the advantage disappears unless the matmul also runs in low precision.

</details>

### P4. Little's law and KV

A service handles 120 requests per second with a mean time in system of 6 s. Each request holds 2,200 tokens of KV at 160 KiB (163,840 bytes) per token. How much KV memory is needed at the mean, and what happens if mean latency rises to 9 s because of a slow dependency?

<details><summary>Answer</summary>

In flight L = 120 x 6 = 720. KV = 720 x 2,200 x 163,840 = 259.5 GB. At 9 s, L = 1,080 and KV = 389 GB, 50% more memory at the same traffic. If the fleet is sized for 6 s, the extra concurrency exhausts KV, requests queue or are preempted, latency rises further, and the system spirals. Latency is a capacity input, not only an output.

</details>

### P5. Choosing the speculation depth

A drafter costs 8% of a target step (c = 0.08) and has per-token acceptance alpha = 0.75. Compare k = 3 and k = 5. Which is better, and what happens to the choice if alpha drops to 0.5?

<details><summary>Answer</summary>

E(k) = (1 - alpha^(k+1)) / (1 - alpha). k = 3: (1 - 0.3164) / 0.25 = 2.734 tokens at cost 1 + 3 x 0.08 = 1.24, so 2.21x. k = 5: (1 - 0.1780) / 0.25 = 3.288 tokens at cost 1.40, so 2.35x. Depth 5 is slightly better. At alpha = 0.5: k = 3 gives 1.875 / 1.24 = 1.51x, k = 5 gives 1.969 / 1.40 = 1.41x, so the shorter draft wins because later positions are rarely accepted. Also check the batch: at high batch, verification is no longer free.

</details>

### P6. Cache-aware routing

You have 8 replicas of a 70B model. 500 tenants each send a unique 4,000-token system prompt (KV 1.31 GB per prompt at 320 KiB per token). Each replica can keep about 40 GB of prefix cache. Compare prefix hit rates for round-robin routing and for routing by tenant hash, assuming equal traffic per tenant. What breaks with hash routing?

<details><summary>Answer</summary>

Each replica holds 40 / 1.31 = ~30 prefixes. Round-robin sends every tenant to every replica, so each replica sees 500 prefixes and can hold 30: a hit rate near 30 / 500 = 6% (LRU with uniform traffic). Hash routing gives each replica 500 / 8 = 62.5 tenants, so the hit rate is about 30 / 62.5 = 48%, and it rises as replicas are added. What breaks: hot tenants overload their replica (add a load cap with spill to a second replica), failures reshuffle the mapping (use consistent hashing), and the cache competes with running sequences for the same memory, so the true budget is smaller.

</details>

### P7. Which memory plan fits?

You want one 80 GB GPU to serve a 70B model with 16 concurrent 8K-token sequences. Weights at INT4 group 128 are 36.1 GB. Does it fit with FP16 KV? With FP8 KV? Use 90% memory utilisation.

<details><summary>Answer</summary>

KV at FP16 = 16 x 8,192 x 327,680 = 42.9 GB. Total 36.1 + 42.9 = 79.0 GB against 72 GB usable: does not fit. With FP8 KV: 21.5 GB, total 57.6 GB, which fits with ~14 GB spare. The cost is two quality risks to test: INT4 weights and FP8 KV, especially on long-context retrieval. Alternative: two GPUs with TP and FP8 weights, no 4-bit risk, double the hardware cost.

</details>

### P8. Timeout design for a streaming chat route

Measured: P99 TTFT 1.2 s, P99 ITL 80 ms, maximum output 800 tokens. Propose connect, TTFT, idle and total timeouts and say what each detects.

<details><summary>Answer</summary>

Connect 3 s (network or balancer down). TTFT 3 s, about 2.5 times P99, to catch a stuck queue or prefill without firing on normal tails. Idle (inter-chunk) 10 s, over 100 times the P99 ITL, to catch a stalled stream quickly without tripping on a legitimate preemption pause. Total: 1.2 s + 800 x 0.08 s = 65 s expected worst case, so cap at about 90 s to bound cost and user patience. A single 90 s timeout would take 90 s to notice a frozen stream.

</details>

### P9. Why the knee matters

A single server pool behaves like M/M/1 with a service rate of 4 requests per second. Compute mean time in system and the P99 time in system at arrival rates of 2 and 3.6 per second. Then say what changes with 4 such servers sharing one queue at the same per-server load.

<details><summary>Answer</summary>

M/M/1: W = 1 / (mu - lambda), P99 = ln(100) / (mu - lambda). At 2 per second (50% load): W = 0.5 s, P99 = 2.3 s. At 3.6 per second (90% load): W = 2.5 s, P99 = 11.5 s. Load rose 80% while P99 rose 5 times. With 4 servers sharing one queue (M/M/4, a = 3.6 with per-server load 90%, total lambda 14.4) the chance of waiting (Erlang C) is about 0.79 and mean wait is about 0.79 / (16 - 14.4) = 0.49 s, versus 2.25 s of queueing for a single server at the same load. Pooling shrinks waits sharply, which is why one shared pool beats separate per-tenant pools at equal utilisation, and why routing that fragments traffic hurts.

</details>

### P10. Self-host break-even

A fleet of 16 GPUs at $2.50 per hour sustains 8,000 total tokens per second at full load. An API offers equivalent quality at $0.90 per million total tokens. Compute the self-hosted cost per million tokens at 100%, 60% and 30% average utilisation, and the break-even utilisation.

<details><summary>Answer</summary>

Fleet cost = $40 per hour. At full load: 8,000 x 3,600 = 28.8M tokens per hour, so $40 / 28.8M x 1e6 = $1.39 per million. At 60% utilisation: $2.31. At 30%: $4.63. All three are above $0.90, so at this price and throughput the API wins everywhere, because even 100% utilisation costs $1.39. Break-even would need either a lower hardware cost per token (better batching, quantisation, prefix caching raising throughput to above 12,300 tokens per second at full load) or a much higher API price. This is the correct reading of the exercise: the utilisation lever only matters when the full-load cost is already below the API price.

</details>
