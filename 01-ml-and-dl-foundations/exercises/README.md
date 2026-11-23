# Exercises - ML & Deep Learning Foundations

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3 and numpy only, run in under 10 seconds each, and use fixed seeds.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub file, for example `ex01_logistic_regression.py`. The module docstring has the TASK (exact signatures), HINTS, and WHAT YOU SHOULD LEARN.
3. Replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests()`.
4. Run it from this folder: `python ex01_logistic_regression.py`. An untouched stub fails with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
5. If you are stuck, read the hints, then compare with `solutions/ex01_logistic_regression.py`. The solution file has the same tests and comments on the key steps.

Work in order. Exercises 2 and 3 are the ones to be able to redo from a blank file.

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_logistic_regression.py` | Easy | Bernoulli NLL, stable BCE from logits, the $p - y$ gradient, finite-difference gradient check, effect of L2 | 30 min |
| 2 | `ex02_scalar_autograd.py` | Medium | Computational graph, topological order, gradient accumulation at fan-out, building backprop from local derivatives | 60 min |
| 3 | `ex03_mlp_backprop.py` | Medium | Matrix backprop for a tanh MLP with softmax cross-entropy, cached activations, numerical gradient for every parameter | 60 min |
| 4 | `ex04_optimisers.py` | Medium | Momentum, Adam with bias correction, AdamW versus Adam + L2, warmup + cosine schedule, stability limit on an ill-conditioned quadratic | 60 min |
| 5 | `ex05_metrics.py` | Medium | P/R/F1, rank-based ROC-AUC with ties, average precision, ECE, temperature scaling, AUC versus AP under imbalance | 60 min |
| 6 | `ex06_cv_bias_variance.py` | Medium | K-fold and group k-fold, polynomial ridge fit, measuring $\text{bias}^2$ and variance by simulation | 60 min |
| 7 | `ex07_norms_init_dropout.py` | Hard | He/Xavier signal propagation across depth, LayerNorm/RMSNorm/BatchNorm axes and train/eval behaviour, inverted dropout, InfoNCE | 75 min |

More practice that builds on these: [../../12-coding-challenges/README.md](../../12-coding-challenges/README.md) (attention, layer norm and softmax, evaluation metrics, LoRA and others).

## Pen-and-paper and design exercises

Attempt each on paper before opening the answer.

### P1. Diagnose from numbers

Three runs of the same network report (train loss, validation loss): A = (0.90, 0.95), B = (0.05, 0.60), C = (0.30, 0.31). Label noise is believed to be about 0.25. For each run, say whether it is dominated by bias, variance or neither, and give the first change you would try.

<details><summary>Answer</summary>

A: both are high and close, far above the noise floor of 0.25. This is bias (underfitting). Try a larger model, longer training or a better learning rate, and check for bugs before touching regularisation. B: train loss (0.05) is below the noise floor and validation is 12 times higher. This is variance (overfitting). Try more data or augmentation, weight decay or dropout, early stopping, a smaller model. C: close to the noise floor and no gap. Neither dominates, so extra effort should go to data quality or a different problem framing, since the model is already near the floor.

</details>

### P2. Softmax cross-entropy gradient

Logits are $z = [1, 2, 3]$ and the true class is index 0. Compute the probabilities, the loss, and the gradient with respect to the logits. Then state which entry has the largest magnitude and why.

<details><summary>Answer</summary>

$$\exp(z - 3) = [e^{-2}, e^{-1}, 1] = [0.1353, 0.3679, 1], \qquad \text{sum} = 1.5032$$

$$p = [0.0900, 0.2447, 0.6652]$$

$$L = -\ln 0.0900 = 2.408$$

$$\frac{\partial L}{\partial z} = p - \text{onehot} = [0.0900 - 1, 0.2447, 0.6652] = [-0.910, 0.245, 0.665]$$

The entries sum to zero. The largest magnitude is the true class's entry (-0.910), because the model gives it low probability. The wrongly dominant class 2 is next (0.665), which is how the update lowers the logit that is too high.

</details>

### P3. Hand backprop through a ReLU

Let $L = \tfrac{1}{2} \big(\mathrm{relu}(wx + b) - y\big)^2$ with $x = 2$, $w = 1.5$, $b = -1$, $y = 1$. Compute $L$, $\partial L / \partial w$, $\partial L / \partial b$ and $\partial L / \partial x$. Then set $b = -4$ and repeat. What does the second case show?

<details><summary>Answer</summary>

First case: pre-activation $u = 1.5 \times 2 - 1 = 2$, $r = \mathrm{relu}(u) = 2$, error $= r - y = 1$, $L = 0.5$. Backward: $\partial L / \partial r = \text{error} = 1$. The ReLU passes the gradient because $u \gt 0$, so $\partial L / \partial u = 1$. Then $\partial L / \partial w = 1 \cdot x = 2$, $\partial L / \partial b = 1$, $\partial L / \partial x = 1 \cdot w = 1.5$. Second case: $u = 3 - 4 = -1$, $r = 0$, error $= -1$, $L = 0.5$. The ReLU gate is closed ($u \lt 0$), so $\partial L / \partial u = 0$ and all three gradients are 0, even though the loss is not zero. This is the dead-unit problem: if a unit's pre-activation is negative for every example, it receives no gradient and cannot recover.

</details>

### P4. Two Adam steps by hand

A single weight $w = 1.0$, learning rate $\eta = 0.01$, default betas $(\beta_1, \beta_2) = (0.9, 0.999)$, $\epsilon$ negligible. The gradients are 4.0 then 0.0. Compute $w$ after each step. Why does the weight move on the second step although the gradient is zero?

<details><summary>Answer</summary>

Step 1:

$$m = 0.4, \quad v = 0.016, \quad \hat m = 4, \quad \hat v = 16 \quad \text{(bias-corrected)}$$

$$\text{step} = 0.01 \times 4 / 4 = 0.01, \quad w = 0.99$$

Step 2, with $g = 0$:

$$m = 0.9 \times 0.4 = 0.36, \quad v = 0.999 \times 0.016 = 0.015984$$

$$\hat m = \frac{0.36}{1 - 0.81} = 1.895, \quad \hat v = \frac{0.015984}{1 - 0.998001} = 7.996, \quad \sqrt{\hat v} = 2.828$$

$$\text{step} = 0.01 \times 1.895 / 2.828 = 0.0067, \quad w = 0.9833$$

The weight moves because the first-moment EMA still remembers the earlier gradient (momentum), and the second-moment EMA remembers its magnitude, so the update is 0.67 of a full step in the same direction.

</details>

### P5. Choose the initial scale

A ReLU MLP has hidden layers of width 1024. What standard deviation do He initialisation and Xavier initialisation give for a 1024 by 1024 weight matrix? If you accidentally used Xavier, by what factor would the activation standard deviation shrink after 20 layers?

<details><summary>Answer</summary>

He: $\sqrt{2 / 1024} = 0.0442$. Xavier: $\sqrt{2 / (1024 + 1024)} = \sqrt{1 / 1024} = 0.03125$. Xavier is lower by a factor of $\sqrt{2}$ in standard deviation, and each ReLU layer already halves the second moment, so per layer the activation std shrinks by about $1 / \sqrt{2}$. After 20 layers: $2^{-10} = 1 / 1024$, about $10^{-3}$.

</details>

### P6. Threshold and base rate

(a) A screening test has sensitivity 90% and specificity 95% and is applied where prevalence is 2%. What is its precision (positive predictive value)? (b) A missed case costs 5 units and a false alarm costs 1 unit, and the model is calibrated. What probability threshold minimises expected cost?

<details><summary>Answer</summary>

(a) Per 1,000 people: 20 positive, 980 negative. $\text{TP} = 0.9 \times 20 = 18$. $\text{FP} = 0.05 \times 980 = 49$. Precision = 18 / 67 = 26.9%. Most alerts are false even though the test looks good. (b) Predict positive when $p \gt c_{\text{FP}} / (c_{\text{FP}} + c_{\text{FN}}) = 1 / 6 = 0.167$.

</details>

### P7. Leakage audit

A team predicts hospital readmission from patient visits. They have 200,000 visits from 40,000 patients, standardise all features, randomly split visits 80/20 for train and test, and report AUC 0.94. Production AUC turns out to be 0.71. List at least four likely causes and the fix for each.

<details><summary>Answer</summary>

1. Group leakage: visits from the same patient appear in train and test, so the model identifies patients. Fix: split by patient (group k-fold).
2. Temporal leakage: random splits train on future visits to predict past ones. Fix: split by time, train on earlier periods and test on later ones.
3. Preprocessing leakage: scaling, imputation or encoders fitted on all data before the split. Fix: fit inside the training fold only.
4. Target leakage: features recorded after the outcome (discharge codes, follow-up scheduling) are used as inputs. Fix: audit each feature's timestamp relative to the prediction time.
5. Distribution shift between the development period and production, or training-serving skew in how features are computed. Fix: monitor input distributions, compute features through a shared pipeline, and evaluate on a recent holdout.
6. Repeated model selection on the test set. Fix: keep a sealed final test set.

</details>

### P8. Embedding pipeline design

You build semantic search over support tickets. The vendor's encoder was trained with cosine similarity. Your vector index is configured for inner product, vectors are not normalised, and the team plans to swap the encoder for a newer one next quarter. What goes wrong, and what do you change?

<details><summary>Answer</summary>

With unnormalised vectors and an inner-product index, ranking mixes direction and magnitude. Long vectors (often from long or unusual text) can outrank closer matches, and the result silently differs from what the encoder was trained for. Fix: L2-normalise vectors at ingestion and at query time, so inner product, cosine and L2 give the same ranking, or configure the index for cosine. For the encoder swap: embeddings from different models live in different spaces, so you must re-embed the whole corpus (build the new index alongside the old one and switch over), and you must not mix old and new vectors in one index. Evaluate both encoders on a labelled set of your own tickets with recall@k and MRR before switching. Do not reuse a similarity threshold across models, since typical cosine values differ between encoders.

</details>
