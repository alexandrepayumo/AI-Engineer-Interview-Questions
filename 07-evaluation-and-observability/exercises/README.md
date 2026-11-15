# Evals & Observability - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and numpy only (plus the standard library), run in under 10 seconds each, use fixed seeds, and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_bootstrap_ci.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_bootstrap_ci.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_bootstrap_ci.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [pass@k, QA F1, exact match and a basic pairwise judge harness](../../12-coding-challenges/12_eval_metrics.py) and, for retrieval metrics, [the RAG exercises](../../04-rag-and-retrieval/exercises/README.md).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_bootstrap_ci.py` | Easy | Percentile bootstrap CI for accuracy, paired versus unpaired bootstrap of a model difference, bootstrap p-value (Part 5) | 40 min |
| 2 | `ex02_cohens_kappa.py` | Easy | Percent agreement, confusion matrix, Cohen's kappa, weighted kappa, the prevalence effect (Parts 2, 4) | 40 min |
| 3 | `ex03_position_bias.py` | Medium | Swapped-order judging, consistency rate, first-slot share, exact binomial test, debiased win rate (Part 4) | 45 min |
| 4 | `ex04_sample_size.py` | Medium | Sample size for two accuracies, paired design, power curve, minimum detectable effect (Part 5) | 50 min |
| 5 | `ex05_elo_bradley_terry.py` | Medium | Online Elo, order dependence, Bradley-Terry maximum likelihood by the MM update, Elo scale (Part 6) | 60 min |
| 6 | `ex06_trace_spans.py` | Medium | Span tree, self time with parallel children, subtree cost, critical path, percentiles (Part 8) | 75 min |
| 7 | `ex07_drift_psi_ks.py` | Medium | Population stability index, two-sample KS statistic and p-value, thresholds (Part 8) | 50 min |

Suggested order: 1, 2, 4, 3, 5, 7, 6.

## Pen-and-paper and design exercises

Attempt each before opening the answer. A calculator is fine for square roots and logarithms.

### P1. Error bars on a small suite

Your suite has 60 examples and the new prompt scores 51 correct, the old prompt 46. Compute each pass rate, the standard error of the difference if the two runs were independent, and say whether the gap is distinguishable from noise.

<details><summary>Answer</summary>

Rates: 51/60 = 0.85 and 46/60 = 0.767. Difference = 0.083. Standard error of each: sqrt(0.85 x 0.15 / 60) = 0.046 and sqrt(0.767 x 0.233 / 60) = 0.055. Independent difference SE = sqrt(0.046^2 + 0.055^2) = 0.072. The 95% interval is 0.083 +- 0.141, which comfortably includes zero. Only a paired analysis could rescue the claim: if, say, 9 examples flipped to correct and 4 flipped to wrong, the exact binomial test on 13 discordant pairs gives p = 0.27 (two-sided), still not significant. Five more points on 60 examples is not evidence.

</details>

### P2. Kappa from a table

A judge and a human label 200 answers as good or bad. Both say good on 120, both say bad on 40, the judge says good and the human bad on 30, the judge says bad and the human good on 10. Compute percent agreement and Cohen's kappa. Is the judge usable?

<details><summary>Answer</summary>

po = (120 + 40) / 200 = 0.80. Judge good rate = 150/200 = 0.75, human good rate = 130/200 = 0.65. pe = 0.75 x 0.65 + 0.25 x 0.35 = 0.4875 + 0.0875 = 0.575. kappa = (0.80 - 0.575) / (1 - 0.575) = 0.529, "moderate" agreement. The errors are lopsided: the judge passes 30 answers that humans reject versus 10 the other way, so it is lenient. Before using it, look at those 30 cases, tighten the rubric, and re-measure. For gating, a lenient judge is the dangerous direction.

</details>

### P3. Correcting a judge's pass rate

Judge sensitivity (passes answers humans pass) is 0.90 and specificity (fails answers humans fail) is 0.80, estimated on a labelled calibration set. In production the judge passes 70% of answers. What is the estimated true pass rate, and why is the raw 70% misleading?

<details><summary>Answer</summary>

observed = p x sens + (1 - p) x (1 - spec), so p = (observed - (1 - spec)) / (sens + spec - 1) = (0.70 - 0.20) / 0.70 = 0.714. The correction is small here but grows as specificity drops. The calibration estimates have their own error, so the corrected rate needs a wider interval than the raw one. Corrections only hold if the production distribution resembles the calibration set.

</details>

### P4. Choosing the test for a flaky pipeline

An eval has 300 examples. The model runs at temperature 0.7 and each example is run once. Version B scores 2 points above version A. Name two separate sources of variance in that number and one design change that reduces each.

<details><summary>Answer</summary>

(1) Example sampling variance: the 300 examples are a sample of the real distribution. Reduce by pairing both versions on the same examples and using a paired test or paired bootstrap, and by adding examples. (2) Generation variance: a single sample at temperature 0.7 is a random draw. Reduce by running several samples per example and scoring the mean (the clusters are then examples, so the standard error must be computed over examples, not over all runs). Judge noise is a third source if a model grades the outputs: fix the judge's temperature and use consistent order swapping.

</details>

### P5. Design a golden set for a support bot

You are launching a customer-support assistant that answers from a help-centre RAG corpus and can issue refunds through a tool. Outline the first golden set: size, sources, strata, label types and how you would keep it from rotting.

<details><summary>Answer</summary>

Start with about 100 examples, since the team can read and label them in a day or two. Sources: 40 from real tickets (stratified by intent and by outcome including escalations), 20 adversarial or policy cases (refund over the limit, prompt injection in a ticket, out-of-scope questions), 20 synthetic edge cases generated from a scenario taxonomy and human-reviewed, 20 retrieval-hard cases where the answer needs two documents. Labels: a reference answer or key facts for answerable cases, the gold source chunk ids for retrieval, expected tool calls with arguments for the refund cases, and an explicit "should refuse or escalate" flag. Graders: assertions for tool calls and refusal, claim-level faithfulness judge for answers, humans for a 20-case calibration slice. Hold out 30% as a test split that prompt authors do not look at. Version it in the repo, record the dataset hash in every result, and add each new production failure within a week. Retire cases that every version passes into a regression suite.

</details>

### P6. Reading a latency percentile table

For one route over a day you see p50 = 1.4 s, p95 = 4.8 s, p99 = 14 s, with 200,000 requests. Roughly how many requests were slower than 14 s? An agent makes 5 independent calls to this route in sequence. What fraction of agent runs hit at least one call slower than 14 s, and what does that say about which percentile to alert on?

<details><summary>Answer</summary>

About 1% of 200,000, so ~2,000 requests above 14 s. For 5 sequential calls: 1 - 0.99^5 = 4.9%, so about 1 agent run in 20 contains a 14 s call, close to the single-call p95 rate. Tail latency compounds across fan-out and sequential steps, so alert on p95 and p99 per route and on the end-to-end task latency distribution, not on the mean or p50.

</details>

### P7. Interpreting PSI

A weekly check of a judge-score histogram against the launch week gives expected bins [0.10, 0.20, 0.40, 0.20, 0.10] and actual bins [0.05, 0.10, 0.35, 0.30, 0.20]. Compute PSI and decide what you do next.

<details><summary>Answer</summary>

Terms (a - e) x ln(a / e): bin 1: -0.05 x ln(0.5) = 0.0347. Bin 2: -0.10 x ln(0.5) = 0.0693. Bin 3: -0.05 x ln(0.875) = 0.0067. Bin 4: 0.10 x ln(1.5) = 0.0405. Bin 5: 0.10 x ln(2) = 0.0693. PSI = 0.2205, "moderate" (0.1 to 0.25). Scores moved upward. The score shift is a symptom with three possible causes: input mix changed (check intent distribution), the model or prompt changed (check version tags in traces), or the judge changed. Pull a sample of high-score and low-score traces, read them, and re-run the golden set against the current configuration to separate real quality change from a shifted input mix.

</details>

### P8. Rollout plan for a model swap

Your product uses model X. Model Y is cheaper and scores equal on your golden set. Write the rollout plan, naming the gate at each step.

<details><summary>Answer</summary>

(1) Offline: full golden set, paired comparison with a confidence interval on quality, all guardrail assertions at zero tolerance, and cost and latency measured. Gate: no guardrail regression, quality difference lower bound above the agreed non-inferiority margin. (2) Shadow: mirror a slice of live traffic to Y without user exposure and compare with judge scores and error rates, which also catches distribution gaps the golden set missed. (3) Canary: 1 to 5% of users, watching error rate, refusal rate, latency p95, cost per task, and guardrail metrics, with an automatic rollback. (4) A/B: user-level randomisation on a task-success metric with sample size computed beforehand and a sample-ratio check. (5) Full rollout with the old model kept pinned and warm for rollback. Pin exact model versions throughout, and record the version in every trace.

</details>

### P9. Why a benchmark gain may not transfer

A vendor reports +4 points on a public coding benchmark for a new model. List four reasons this may not help your internal coding assistant, and the check you run for each.

<details><summary>Answer</summary>

(1) Contamination: benchmark items may be in training data. Check with fresh, post-cutoff internal tasks. (2) Distribution gap: the benchmark is small self-contained problems and your work is multi-file changes in a private codebase. Check on tasks sampled from your own repositories with your test suites. (3) Harness sensitivity: the gain may depend on the vendor's scaffold, prompt or sampling. Check by running the model in your own harness. (4) Noise: +4 points on a few hundred items can be within the confidence interval. Check with a paired bootstrap on your own suite. Also check cost and latency, which the headline number ignores.

</details>

### P10. Spotting a broken A/B test

An A/B test with 50/50 allocation reports 10,400 users in control and 9,600 in treatment. The treatment shows a 2% lift on task completion with p = 0.03. Do you ship?

<details><summary>Answer</summary>

No. The sample ratio is off: with 20,000 users the expected split is 10,000 each, and 400 over gives chi-square = 400^2 / 10,000 + 400^2 / 10,000 = 32 (p well below 0.001). A sample ratio mismatch means assignment or logging is broken, for example treatment users dropping out of logging when the new variant errors or is slow. The lift cannot be trusted until the cause is found. Fix the pipeline, rerun, and always run an SRM check before reading metrics.

</details>
