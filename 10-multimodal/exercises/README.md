# Multimodal Models - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and numpy (the WER exercise uses only the standard library), run in under 10 seconds each, use fixed seeds and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_patchify_tokens.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_patchify_tokens.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_patchify_tokens.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [attention from scratch](../../12-coding-challenges/01_attention.py) is the building block of every encoder in this folder, and [evaluation metrics](../../12-coding-challenges/12_eval_metrics.py) covers precision, recall and ranking metrics that also apply to multimodal retrieval.

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_patchify_tokens.py` | Easy | ViT patchify and unpatchify, patch counts, quadratic attention, fixed-tile and per-patch token budgets, dynamic tiling (Parts 1, 3) | 45 min |
| 2 | `ex02_clip_loss.py` | Medium | Symmetric InfoNCE, temperature, SigLIP sigmoid loss, zero-shot classification, retrieval recall (Part 2) | 50 min |
| 3 | `ex03_log_mel.py` | Medium | Mel scale, triangular filterbank, framing, Hann-windowed STFT power, log-mel of a synthetic sine (Part 6) | 60 min |
| 4 | `ex04_wer.py` | Medium | Edit-distance table, backtrace alignment, S/D/I counts, WER above 1, CER, normalisation traps (Part 6) | 45 min |
| 5 | `ex05_diffusion.py` | Hard | Forward process in one jump, linear and cosine schedules, SNR, optimal noise predictor, DDIM, classifier-free guidance (Part 7) | 75 min |
| 6 | `ex06_boxes.py` | Medium | IoU, coordinate conventions, greedy matching, average precision, grounding accuracy (Part 8) | 50 min |
| 7 | `ex07_video_sampling.py` | Easy | Frame sampling, token-budget arithmetic, keyframe selection, coarse-then-fine cost (Part 8) | 40 min |

Suggested order: 1, 2, 4, 3, 7, 6, 5.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator only for exponentials and logarithms. Prices and token figures are illustrative.

### P1. Resolution and encoder cost

A ViT with 14 px patches reads a 224 x 224 image and a 448 x 448 image. Compute the patch tokens for each, the number of query-key pairs in one self-attention layer for each, and the LLM-side token count at 448 if the projector merges each 2 x 2 group of patches into one token.

<details><summary>Answer</summary>

224 / 14 = 16, so 16 x 16 = 256 patches. 448 / 14 = 32, so 32 x 32 = 1,024 patches. Attention pairs: 256^2 = 65,536 versus 1,024^2 = 1,048,576, which is 16 times more for a 2 times larger side. The 2 x 2 merge gives 1,024 / 4 = 256 tokens for the LLM, so the encoder pays for the full 448 resolution while the LLM context pays only for 256 tokens. This is why encoder compute and LLM context cost are separate budgets.

</details>

### P2. Cost of a document pipeline

A 100-page PDF is sent as images at 1,568 tokens per page, with input priced at 3 dollars per million tokens. (a) What does one document cost, and 10,000 documents? (b) A cascade runs OCR on every page at 1.5 dollars per 1,000 pages and escalates 15% of the pages to the VLM. What is the cost per document now?

<details><summary>Answer</summary>

(a) 100 x 1,568 = 156,800 tokens, times 3 / 1,000,000 = 0.470 dollars per document, so about 4,700 dollars for 10,000 documents (input only). (b) OCR: 100 pages x 1.5 / 1,000 = 0.15 dollars. Escalated VLM: 0.15 x 0.470 = 0.0706 dollars. Total about 0.22 dollars per document, roughly 2.1 times cheaper. The saving shrinks as the escalation rate grows, and breaks even near 68% escalation (0.15 + 0.470 x r = 0.470 gives r = 0.68), so the classifier that decides which pages are hard matters more than the OCR price.

</details>

### P3. How big is the CLIP logit matrix?

A contrastive batch has N = 32,768 pairs. How many entries does the full N x N logit matrix have, and how much memory is that in float32? Name two ways systems avoid materialising it on one device.

<details><summary>Answer</summary>

32,768^2 = 1,073,741,824 entries, which is 4.29 GB in float32 (about 1.07 billion x 4 bytes). Options: shard the batch across devices so each device holds only a (N / D) x N block and gathers text embeddings, and compute the loss in chunks (the SigLIP approach, which rotates text embeddings between devices and accumulates per-chunk sigmoid losses because it needs no batch-wide softmax denominator). Gradient caching of embeddings is a third option.

</details>

### P4. Temperature by hand

In a batch of N = 4, an image has cosine similarity 0.9 with its caption and 0.5 with each of the 3 other captions. Compute the image-to-text loss for this image at temperature 1 and at temperature 0.1.

<details><summary>Answer</summary>

T = 1: p = e^0.9 / (e^0.9 + 3 e^0.5) = 2.4596 / (2.4596 + 4.9462) = 0.3321, loss = -ln(0.3321) = 1.102. T = 0.1: logits are 9 and 5, p = e^9 / (e^9 + 3 e^5) = 8103.1 / (8103.1 + 445.2) = 0.9479, loss = 0.0535. The same similarities give a 20 times smaller loss at low temperature, and the gradient from this pair becomes small, so training is dominated by harder pairs. CLIP learns the temperature (it ends near 0.01) so the model can sharpen as embeddings improve.

</details>

### P5. One diffusion step by hand

At some timestep abar = 0.25. A clean value is x0 = 2.0 and the noise draw is eps = 1.0. Compute x_t and the SNR. If the network predicts eps perfectly, what is the recovered x0?

<details><summary>Answer</summary>

x_t = sqrt(0.25) x 2.0 + sqrt(0.75) x 1.0 = 1.0 + 0.866 = 1.866. SNR = 0.25 / 0.75 = 0.333. Recovery: x0_hat = (x_t - sqrt(1 - abar) x eps) / sqrt(abar) = (1.866 - 0.866) / 0.5 = 2.0. A perfect eps prediction recovers x0 exactly in one step. A real network only knows E[eps | x_t], so a one-step recovery gives a blurry average of plausible images and not a sample, which is why sampling takes many steps.

</details>

### P6. CFG arithmetic and cost

At one step the unconditional noise prediction for a latent element is 0.20 and the conditional one is 0.50. What is the guided value at scale 7.5? A sampler runs 30 steps with CFG. How many network evaluations is that, compared with no guidance?

<details><summary>Answer</summary>

Guided = 0.20 + 7.5 x (0.50 - 0.20) = 2.45, far outside the range of either prediction, which is why large scales oversaturate. With guidance every step needs the conditional and the unconditional pass: 60 evaluations versus 30. Batching the two branches keeps latency similar but doubles compute and memory. Guidance-distilled models fold the guidance into one pass.

</details>

### P7. A voice agent latency budget

A pipeline spends: endpointing 200 ms, ASR finalisation 150 ms, LLM time to first token 350 ms, then 200 ms of streaming until the first complete sentence, TTS time to first audio 150 ms, network 100 ms. Compute voice-to-voice latency against an 800 ms target and propose cuts that reach it.

<details><summary>Answer</summary>

Total = 200 + 150 + 350 + 200 + 150 + 100 = 1,150 ms, 350 ms over. Cuts: a semantic turn detector shortens endpointing to about 100 ms on complete utterances (-100). Streaming ASR whose final result is already computed when the VAD fires cuts finalisation to about 50 ms (-100). Prompt caching or a smaller model for the first turn brings time to first token to about 200 ms (-150). Flushing TTS on the first clause instead of the first sentence cuts the wait for text to about 100 ms (-100). New total = 100 + 50 + 200 + 100 + 150 + 100 = 700 ms. The risks: a semantic detector can still cut users off, and clause-level TTS can produce choppy prosody, so measure interruption rate and perceived naturalness, not only the median latency.

</details>

### P8. Will the video fit?

A 45-minute video is sampled at 1 frame per second with 258 tokens per frame, plus an audio track costing 32 tokens per second. How many tokens is that? What sampling rate fits a 200,000-token budget if audio is kept?

<details><summary>Answer</summary>

45 minutes = 2,700 s. Frames: 2,700 x 258 = 696,600. Audio: 2,700 x 32 = 86,400. Total 783,000 tokens. With audio kept, (200,000 - 86,400) / 258 = 440 frames, so 440 / 2,700 = 0.163 fps, one frame every 6.1 s. Alternatives: replace the audio with a transcript (cheaper, loses tone), lower the per-frame resolution, or run the two-pass pattern (coarse pass plus targeted high-fps windows) so the budget is spent where the questions are.

</details>

### P9. Read a POPE-style result

A yes/no probe asks "Is there a X in the image?" on 1,000 questions, 500 about present objects and 500 about absent ones. The model answers yes on 475 of the present ones and on 200 of the absent ones. Compute accuracy, precision, recall, F1 and the yes-rate, and say what the numbers show.

<details><summary>Answer</summary>

TP = 475, FN = 25, FP = 200, TN = 300. Accuracy = 775 / 1,000 = 77.5%. Precision = 475 / 675 = 70.4%. Recall = 475 / 500 = 95%. F1 = 2 x 0.704 x 0.95 / (0.704 + 0.95) = 0.809. Yes-rate = 675 / 1,000 = 67.5% against a true rate of 50%. The model is right about what is present but says yes to 40% of absent objects, a yes-bias typical of object hallucination. Accuracy alone hides it, so report the yes-rate and split by how the absent objects were chosen (random, popular, co-occurring).

</details>

### P10. Storage for page-image retrieval

A ColPali-style index stores 1,024 patch vectors of 128 dimensions per page in fp16. For 5,000 pages, how much storage is that? How would it compare with one 1,024-dimension dense vector per page, and what do you get for the extra bytes?

<details><summary>Answer</summary>

Per page 1,024 x 128 x 2 bytes = 262,144 bytes (256 KiB). For 5,000 pages that is about 1.31 GB. One 1,024-dimension fp16 vector is 2 KB per page, 10 MB for 5,000 pages, so the multi-vector index is 128 times larger. The extra bytes buy late interaction: each query token finds its best matching patch, so a query about one table cell or axis label can match a small region of the page instead of being averaged into a single vector. Mitigations: binarise or quantise the vectors (an order of magnitude smaller), prune patches (background patches carry little signal), and use the multi-vector scores only to rerank candidates from a cheap first stage.

</details>
