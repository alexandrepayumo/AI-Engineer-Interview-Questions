# Exercises - LLM and Transformer Fundamentals

Practice for [deep-dive.md](../deep-dive.md). The coding exercises cover the arithmetic and mechanisms around the architecture, and deliberately avoid what [12-coding-challenges](../../12-coding-challenges/README.md) already has (attention, BPE, sampling, RoPE, layer norm, KV cache, mini GPT). Do those too if you have not.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_param_flop_counter.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN), then replace each `raise NotImplementedError("TODO")` with your implementation. The `_run_tests()` code at the bottom is the specification.
3. Run it from the repo root:

   ```powershell
   python 02-llm-fundamentals\exercises\ex01_param_flop_counter.py
   ```

   Unmodified stubs fail with `NotImplementedError`. That is expected. When your version works it prints `All tests passed.`
4. Compare with `solutions/<same name>.py`. The tests are identical, so a pass is a pass. Read the comments for the key steps.

Dependencies are Python 3.12 and numpy only. Each file runs in well under 10 seconds and uses fixed seeds.

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_param_flop_counter.py` | Easy | Exact parameter counts and forward/training FLOPs for a GQA + SwiGLU config; the 6ND rule and where it is off | 30 min |
| 2 | `ex02_kv_cache_calculator.py` | Easy | KV cache bytes for MHA, GQA, MQA, sliding-window hybrids and MLA; capacity planning | 30 min |
| 3 | `ex03_cross_entropy_perplexity.py` | Medium | Log-sum-exp stability, cross-entropy with ignore index, perplexity, bits per byte, the softmax minus one-hot gradient | 45 min |
| 4 | `ex04_gqa_attention.py` | Medium | Grouped-query attention, verified against a per-head MHA loop; MHA and MQA as the two ends of one dial | 45 min |
| 5 | `ex05_chinchilla_solver.py` | Medium | Deriving and solving the compute-optimal $(N, D)$ split; over-training and the irreducible floor | 45 min |
| 6 | `ex06_moe_router.py` | Hard | Top-k routing, expert-batched forward pass, Switch-style balance loss, total versus active parameters | 60 min |

## Pen-and-paper and design exercises

Try each on paper before opening the answer.

### P1. BPE by hand

The merge list learned in the deep dive is: (t, `</w>`), (s, `t</w>`), (e, `st</w>`), (o, w), (l, ow), (w, `est</w>`). Encode "lowest" and "newer". How many tokens does each become?

<details><summary>Answer</summary>

"lowest" starts as l o w e s t `</w>`. The merges apply in order: t+`</w>`, then s+`t</w>`, then e+`st</w>`, giving l o w `est</w>`. Then o+w gives l `ow` `est</w>`, then l+ow gives `low` `est</w>`. Final: 2 tokens, [`low`, `est</w>`]. "newer" is n e w e r `</w>`. None of the merges match any adjacent pair (the only pairs are n-e, e-w, w-e, e-r, r-`</w>`), so it stays as 6 symbols. A tokeniser trained on this tiny corpus knows the word "newest" but not "newer".

</details>

### P2. Softmax saturation at d_k = 256

Assume $q$ and $k$ have independent unit-variance entries. What is the standard deviation of $q \cdot k$? For logits [16, 0, 0, 0], find the top probability and $1-p$. Repeat after dividing by $\sqrt{d_k}$.

<details><summary>Answer</summary>

The standard deviation is $\sqrt{256} = 16$. Logits [16, 0, 0, 0] give $p = e^{16} / (e^{16} + 3) = 0.9999997$ and $1 - p = 3.4 \times 10^{-7}$, so the gradient factor $p(1 - p)$ is $\approx 3.4 \times 10^{-7}$, effectively zero. After dividing by 16 the logits are [1, 0, 0, 0], $p = 0.475$ and $p(1 - p) = 0.249$.

</details>

### P3. RoPE relative score

One 2D pair with $\theta = 0.5$. Query $q = (2, 0)$ at position 4, key $k = (1, 1)$ at position 1. Compute the score two ways: rotate both, then dot; and use the relative form $q^\top R(n - m) \thinspace k$.

<details><summary>Answer</summary>

Offset $n - m = 1 - 4 = -3$, angle $-3 \times 0.5 = -1.5$ rad. $\cos(-1.5) = 0.0707$ and $\sin(-1.5) = -0.9975$. With $k = (1, 1)$, $R(-1.5) \thinspace k = (\cos - \sin, \sin + \cos)$, which gives $(0.0707 + 0.9975, -0.9975 + 0.0707) = (1.0682, -0.9268)$. Dot with $q = (2, 0)$ gives 2.1365. Rotating both explicitly ($q$ by 2.0 rad, $k$ by 0.5 rad) gives the same 2.1365, and moving both positions by any shift (say to 104 and 101) leaves it unchanged.

</details>

### P4. Count a small model

Config: $V = 32{,}000$, $d = 2048$, 16 layers, 16 query heads, 4 KV heads, $d_{\text{head}} = 128$, SwiGLU with $f = 8192$, RMSNorm, no biases, tied embeddings. Compute total parameters. Then the KV cache per token in fp16, and the cache for a 32k sequence.

<details><summary>Answer</summary>

Attention per layer:

$$Q = 2048 \times 2048 = 4{,}194{,}304$$

$$K, V = 2048 \times 512 \text{ each, together } 2{,}097{,}152$$

$$O = 4{,}194{,}304$$

$$\text{total} = 10{,}485{,}760$$

MLP and norms:

$$\text{MLP: } 3 \times 2048 \times 8192 = 50{,}331{,}648$$

$$\text{Norms} = 4{,}096$$

Totals:

$$\text{One layer} = 60{,}821{,}504$$

$$\text{Sixteen layers} = 973{,}144{,}064$$

$$\text{Embedding (tied, counted once)} = 65{,}536{,}000$$

$$\text{Final norm} = 2{,}048$$

$$\text{Total} = 1{,}038{,}682{,}112 \approx 1.04\text{B}$$

Cache per token: $2 \times 16 \times 4 \times 128 \times 2\ \text{bytes} = 32{,}768\ \text{bytes}$ (32 KiB). At 32k tokens: $32{,}768 \times 32{,}768$ bytes = 1 GiB. With full MHA (16 KV heads) it would be 4 GiB.

</details>

### P5. Training budget

You train a 7B-parameter model on 2T tokens. Estimate FLOPs and tokens per parameter. At an effective $4 \times 10^{14}$ FLOP/s per accelerator, how many accelerator-hours?

<details><summary>Answer</summary>

$$6ND = 6 \times 7 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{22}\ \text{FLOPs}$$

$$\text{tokens per parameter} = \frac{2 \times 10^{12}}{7 \times 10^9} \approx 286$$

$$\text{time} = \frac{8.4 \times 10^{22}}{4 \times 10^{14}} = 2.1 \times 10^8\ \text{s} \approx 58{,}000\ \text{accelerator-hours}$$

The 286 tokens per parameter is more than $10\times$ the ~20 rule of thumb, which is a typical over-training choice to lower serving cost. The 58,000 accelerator-hours are about 1,000 accelerators for ~2.4 days. The 6ND rule ignores the attention term and counts embeddings as matmuls, so treat it as accurate to roughly 10 to 30%.

</details>

### P6. Why tokens per parameter drifts

Starting from $L = E + A/N^{\alpha} + B/D^{\beta}$ and $C = 6ND$, show that the compute-optimal $D/N$ scales as $C^{(\alpha - \beta)/(\alpha + \beta)}$. What does that imply with $\alpha = 0.34$ and $\beta = 0.28$?

<details><summary>Answer</summary>

The optimum gives $N$ proportional to $C^{\beta/(\alpha+\beta)}$. Then:

$$D = \frac{C}{6N} \propto C^{1 - \beta/(\alpha+\beta)} = C^{\alpha/(\alpha+\beta)}$$

$$\frac{D}{N} \propto C^{(\alpha - \beta)/(\alpha + \beta)} = C^{0.06/0.62} = C^{0.097}$$

A hundredfold increase in compute raises the ideal ratio by $100^{0.097} \approx 1.56\times$. The ratio is not a constant, and it would be exactly constant only if $\alpha = \beta$.

</details>

### P7. Plan a long-context deployment

You will serve a 70B GQA model (80 layers, 8 KV heads, $d_{\text{head}}$ 128) in bf16 on one node of 8 accelerators with 80 GiB each. Users send up to 128k-token contexts. How many full-length sequences fit, and what options raise it to 40?

<details><summary>Answer</summary>

$$\text{total memory} = 8 \times 80\ \text{GiB} = 640\ \text{GiB} \approx 687 \times 10^9\ \text{bytes}$$

$$\text{cache budget} \approx 687 \times 10^9 - 140 \times 10^9\ (\text{weights}) - 69 \times 10^9\ (10\text{ percent reserve}) \approx 478 \times 10^9\ \text{bytes}$$

$$\text{one full 128k sequence} = 131{,}072 \times 327{,}680 = 42.9 \times 10^9\ \text{bytes}$$

$$\text{sequences} = 478 / 42.9 \approx 11$$

So about 11 sequences fit. Options: an fp8 cache doubles it to ~22. Most real traffic is far shorter than 128k, so paged allocation with admission control by actual length raises effective concurrency a lot. Prefix caching shares common system prompts. A hybrid or sliding-window architecture, or MLA, cuts the per-token cost severalfold. A second node with request routing is the fallback. A good answer states the arithmetic first, then picks levers by which factor of the cache formula they change (heads, bytes, cached length), and mentions the quality risk of cache quantisation and the need to test retrieval at 128k.

</details>

### P8. Perplexity and bits per byte

A model scores 2.0 nats per token on a corpus where the tokeniser averages 3.5 bytes per token. Give perplexity, bits per token and bits per byte. A second model with a different tokeniser averages 4.5 bytes per token and scores 2.4 nats per token. Which compresses better?

<details><summary>Answer</summary>

Perplexity $= e^{2.0} = 7.39$. Bits per token $= 2.0 / \ln 2 = 2.885$. Bits per byte $= 2.885 / 3.5 = 0.824$. Second model: $2.4 / \ln 2 = 3.463$ bits per token, divided by 4.5 bytes per token gives 0.769 bits per byte. Its per-token loss and perplexity ($e^{2.4} = 11.0$) are worse, yet it compresses the text better per byte, because each of its tokens carries more bytes. This is why cross-tokeniser comparisons use bits per byte.

</details>

### P9. MoE shape

An MoE layer has $d = 2048$, expert width $f = 1408$ (SwiGLU), 64 routed experts with top-6 routing, and 2 shared experts. Ignoring the router, what are the total and active MLP parameters per layer, and their ratio?

<details><summary>Answer</summary>

One expert is $3 \times 2048 \times 1408 = 8{,}650{,}752$ parameters. Total: $(64 + 2) \times 8{,}650{,}752 = 570{,}949{,}632$. Active: $(6 + 2) \times 8{,}650{,}752 = 69{,}206{,}016$. Ratio total to active is 8.25. Memory must hold the 571M, while FLOPs per token correspond to 69M. Fine-grained experts (small $f$, many of them, higher $k$) allow a large total capacity at modest active cost, and the shared experts are always on.

</details>
