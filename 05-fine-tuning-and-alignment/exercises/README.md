# Fine-tuning and Alignment - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and numpy only, run in under 10 seconds each, use fixed seeds, and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_sft_masking_packing.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_sft_masking_packing.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_sft_masking_packing.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [LoRA forward pass and merge](../../12-coding-challenges/14_lora_adapter.py) is the basic adapter layer. These exercises start where it stops.

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_sft_masking_packing.py` | Easy | Prompt-token masking, multi-turn masks, masked cross-entropy, sequence packing with correct targets, positions and block-diagonal attention (Part 2) | 45 min |
| 2 | `ex02_memory_estimator.py` | Easy | Bytes per parameter for full fine-tuning, LoRA and QLoRA, adapter parameter counts, activations and checkpointing (Parts 3, 4, 5) | 40 min |
| 3 | `ex03_bradley_terry_reward_model.py` | Medium | Pairwise logistic loss, exact gradients, reward model training, separable-data blow-up, tabular Bradley-Terry (Part 7) | 60 min |
| 4 | `ex04_dpo_loss.py` | Medium | DPO loss and gradients, implicit reward, reward statistics, a tabular DPO policy, the KL-optimal policy check (Part 8) | 60 min |
| 5 | `ex05_grpo_advantages.py` | Medium | Group-relative advantages, Dr. GRPO variant, clipping regimes, sample versus token loss aggregation (Part 9) | 45 min |
| 6 | `ex06_nf4_quantization.py` | Medium | Blockwise absmax quantisation, NF4 codebook from normal quantiles, error versus block size, outliers, storage arithmetic (Part 5) | 60 min |
| 7 | `ex07_kl_penalty_and_optimal_policy.py` | Hard | KL estimators k1, k2, k3, per-token KL-shaped rewards, returns, closed-form KL-regularised optimum (Parts 7, 8) | 75 min |

Suggested order: 2, 1, 6, 3, 7, 4, 5.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator only for logarithms and exponentials.

### P1. Memory for a 13B model

A 13B-parameter model (hidden 5120, FFN 13824, 40 layers, full multi-head attention so K and V are also 5120 wide). Estimate training memory excluding activations for (a) full fine-tuning with Adam in mixed precision, and (b) QLoRA with rank 16 on all seven linear matrices per layer, NF4 with double quantisation (4.127 bits per weight).

<details><summary>Answer</summary>

(a) 16 bytes per parameter x 13e9 = 208 GB. (b) Base: 13e9 x 4.127 / 8 = 6.71 GB. Adapter parameters per layer: four attention matrices of 5120 x 5120 give 4 x 16 x 10,240 = 655,360, and three MLP matrices (5120 x 13824) give 3 x 16 x 18,944 = 909,312, so 1,564,672 per layer and 62.6M over 40 layers. At 16 bytes each that is 1.0 GB. Total 7.7 GB before activations. The ratio is about 27x, and almost all of it comes from not storing grads, master weights and Adam states for the frozen base, plus the 4-bit base.

</details>

### P2. LoRA scaling

Compare (r = 8, alpha = 16) with (r = 64, alpha = 16) under standard LoRA scaling alpha / r. Then under rsLoRA scaling alpha / sqrt(r). What goes wrong with the second configuration, and how does rsLoRA change it?

<details><summary>Answer</summary>

Standard: scales are 16/8 = 2 and 16/64 = 0.25. With alpha fixed, raising the rank shrinks the multiplier eight-fold, so the rank-64 adapter learns more slowly and its extra capacity is damped. rsLoRA: 16/sqrt(8) = 5.66 and 16/sqrt(64) = 2. The multiplier falls only by sqrt(8) = 2.83x. rsLoRA is motivated by keeping learning stable as rank grows, so higher ranks can be used without retuning alpha. In practice many teams keep alpha proportional to r (alpha = 2r) under standard scaling, which holds the multiplier at 2 and gets the same effect by hand.

</details>

### P3. Bradley-Terry probabilities

A reward model scores response A at 1.8 and response B at 0.3. (a) What is the modelled probability that annotators prefer A? (b) If annotators actually preferred A in 90% of comparable pairs, what reward gap would match that? (c) Why can the reward model's absolute scores not be compared across prompts?

<details><summary>Answer</summary>

(a) Gap 1.5, sigmoid(1.5) = 0.818. (b) logit(0.9) = ln(9) = 2.197. (c) The Bradley-Terry loss depends only on differences between two responses to the same prompt, so adding any prompt-specific constant leaves the loss unchanged. The absolute level is not identified. This is why advantage estimates, baselines and DPO's implicit reward are all used as within-prompt comparisons.

</details>

### P4. One DPO pair by hand

beta = 0.1. Sequence log-probs: chosen policy -12, chosen reference -14. Rejected policy -20, rejected reference -19. Compute the implicit rewards, z, the loss, and the gradient of the loss with respect to the chosen and rejected policy log-probs (batch of one).

<details><summary>Answer</summary>

Chosen implicit reward = 0.1 x (-12 + 14) = +0.2. Rejected = 0.1 x (-20 + 19) = -0.1. z = 0.2 - (-0.1) = 0.3. Loss = -ln sigmoid(0.3) = 0.554. Weight sigmoid(-0.3) = 0.426. Gradient wrt chosen log-prob = -0.1 x 0.426 = -0.0426 (raise it), wrt rejected = +0.0426 (lower it). The pair is already ranked correctly (z > 0) but only mildly, so the gradient is still large. At z = 0 the weight would be exactly 0.5.

</details>

### P5. GRPO on a small group

A group of 8 samples has 3 correct (reward 1) and 5 wrong (reward 0). Compute the advantages with std normalisation and without. A second prompt has all 8 correct. What advantage do its samples get and what does that mean for the batch?

<details><summary>Answer</summary>

Mean = 0.375, population std = sqrt(0.375 x 0.625) = 0.484. Normalised: correct = 0.625 / 0.484 = +1.291, wrong = -0.375 / 0.484 = -0.775. Without std (Dr. GRPO): +0.625 and -0.375. For the all-correct prompt, every reward equals the mean, so every advantage is 0 and the prompt contributes no gradient (it only costs generation compute). Dynamic-sampling filters such as DAPO's drop these groups and sample replacements, so the batch is full of informative prompts.

</details>

### P6. Choose the approach

A support assistant must (1) answer from a product policy that changes weekly, (2) always reply in a fixed JSON schema with a house tone, (3) run at under 300 ms time-to-first-token on a tight budget. Which combination of prompting, RAG and fine-tuning do you choose, and in what order do you build it?

<details><summary>Answer</summary>

Weekly-changing policy is knowledge, so it goes to RAG over the policy documents, not into weights. Fixed schema and tone are form. Start with a prompt plus constrained decoding (schema-enforced output) and measure. If the system prompt with examples is long enough to break the latency budget, or tone and schema adherence stay below target, fine-tune a smaller model with LoRA on a few thousand reviewed examples of retrieval-conditioned answers (retrieved context in the prompt, loss on the answer only), which shortens the prompt and lets a smaller model meet the latency target. Keep an eval set with both policy-grounded questions and general chat, and re-run it after each adapter version. Order: prompt, RAG, then fine-tune for form and cost.

</details>

### P7. Is this win rate real?

A fine-tuned model beats the baseline in 120 of 200 judged pairwise comparisons (60%). Give a 95% interval. How many comparisons would you need for a +-3 point interval at 60%? Name two ways the number could still be misleading.

<details><summary>Answer</summary>

Standard error = sqrt(0.6 x 0.4 / 200) = 0.0346, so the 95% interval is 60% +- 6.8 points, about 53% to 67%. It excludes 50%, but barely. For +-3 points you need n = 1.96^2 x 0.24 / 0.03^2 = ~1,025 comparisons. Misleading even then: LLM judges prefer longer answers and the first position (so randomise order and control length), and ties or prompts that overlap the training data inflate the rate (so decontaminate and report tie handling). A judge also may not track the real objective, so validate it against human labels on a sample.

</details>

### P8. Diagnose a regression

After SFT on 20,000 support tickets (3 epochs, full fine-tuning, LR 2e-5), the task metric rises from 62% to 81% but a general benchmark drops from 68% to 61% and the model now answers every question in ticket format. List the likely causes and an ordered set of fixes.

<details><summary>Answer</summary>

Causes: narrow data with a single format, three epochs at a high learning rate for full fine-tuning, and no replay of general data. Fixes in order of cost: (1) mix 10-25% general instruction data back in, (2) cut to 1-2 epochs and lower the LR (for example 1e-5), (3) switch to LoRA, which changes a low-rank subspace and typically forgets less (and learns less), (4) interpolate the tuned weights back toward the base (weight averaging or a merge with a coefficient picked by sweeping on both evals). Re-run both evals after each change. Do not decide on the task metric alone.

</details>

### P9. Packing utilisation

Eight training examples have lengths 120, 300, 80, 640, 200, 150, 900 and 410 tokens. Compare token utilisation when padding all eight to the longest in one batch, versus packing greedily into 4096-token rows. What must be fixed at packed boundaries?

<details><summary>Answer</summary>

Total = 2,800 tokens. Padding to 900 gives 8 x 900 = 7,200 slots, so utilisation = 2,800 / 7,200 = 38.9%. Greedy packing fits all 2,800 tokens into one row of 4,096, utilisation 68.4% (and with a smaller row, say 3,000, it would be 93%). At boundaries you must: mask the label of the first token of each example from the previous example's last position (otherwise the model is trained to predict an unrelated sequence's opening), restart position ids, and use a block-diagonal causal attention mask or equivalent so examples cannot attend to each other. Exercise 1 implements all three.

</details>
