# Safety, Security & Responsible AI - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and the standard library only, run in well under 10 seconds each, use fixed seeds and need no network. They are defensive: they build detectors, encoders, metrics and policy layers, and contain no attack payloads.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_pii_redactor.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_pii_redactor.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_pii_redactor.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [taint tracking and a capability guard for agents](../../06-agents-and-tool-use/exercises/ex06_taint_guard.py), [evaluation metrics](../../12-coding-challenges/12_eval_metrics.py), [a basic agent loop](../../12-coding-challenges/10_agent_loop.py).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_pii_redactor.py` | Medium | Luhn checksum, regex candidates, span merging and priority, reversible placeholders (Part 5) | 60 min |
| 2 | `ex02_spotlighting_canary.py` | Medium | Delimiting, datamarking and base64 encoding of untrusted text, random boundaries, canary leak detection (Part 3) | 45 min |
| 3 | `ex03_guardrail_threshold.py` | Medium | Confusion matrix, threshold for a target FPR, cost-weighted threshold, Bayes cut, base-rate precision (Part 6) | 50 min |
| 4 | `ex04_fairness_metrics.py` | Easy | Demographic parity, disparate impact, equalised odds, counterfactual flip rate (Part 9) | 40 min |
| 5 | `ex05_redteam_asr.py` | Medium | Per-category ASR with Wilson intervals, two-proportion z, zero-success bounds, best-of-N (Part 4) | 50 min |
| 6 | `ex06_moderation_pipeline.py` | Medium | Severity tiers, strictness multiplier, fail-closed on missing classifiers, streaming gate (Parts 6, 10) | 55 min |
| 7 | `ex07_memorisation_check.py` | Medium | n-gram overlap, longest verbatim span, membership-inference AUC (Part 5) | 45 min |

Suggested order: 4, 5, 3, 1, 2, 6, 7.

## Pen-and-paper and design exercises

Attempt each before opening the answer. A calculator is fine for powers and logarithms.

### P1. Precision at a realistic base rate

A prompt-injection classifier has TPR 90% and FPR 0.5%. In a deployment, 1 request in 2,000 is an attack and there are 400,000 requests a day. (a) How many attacks are blocked and missed per day? (b) How many benign requests are blocked? (c) What is the precision? (d) What FPR would give precision of 50%?

<details><summary>Answer</summary>

Attacks = 400,000 / 2,000 = 200, benign = 399,800. (a) Blocked = 0.9 x 200 = 180, missed = 20. (b) Benign blocked = 0.005 x 399,800 = 1,999. (c) Precision = 180 / (180 + 1,999) = 8.3%. (d) For precision 0.5, FP = TP = 180, so FPR = 180 / 399,800 = 0.045%. That is 11 times lower than the current FPR, which is why a second stage or a stricter threshold is needed before anyone acts on a block automatically.

</details>

### P2. Retry arithmetic

A jailbreak succeeds against your assistant 0.2% of the time per independent attempt. (a) What is the chance that at least one of 500 attempts succeeds? (b) How many attempts give a 90% chance? (c) What does this say about accepting "0.2%"?

<details><summary>Answer</summary>

(a) 1 - 0.998^500 = 1 - e^(500 x ln 0.998) = 1 - e^(-1.001) = 63.3%. (b) Solve 1 - 0.998^n = 0.9, so n = ln(0.1) / ln(0.998) = 1,150 attempts. (c) A per-attempt rate is not a risk statement for a public system with cheap retries. What matters is the consequence of one success. Bound the blast radius, add rate limits per account, and monitor repeated near-miss patterns.

</details>

### P3. Sample size for a red-team claim

You want to claim, at 95% confidence, that the injection ASR for a tool-using agent on one attack family is below 2%. (a) How many independent attempts with zero successes do you need? (b) You run 100 attempts and see 1 success. What can you claim? (Use the Wilson interval, z = 1.96.)

<details><summary>Answer</summary>

(a) n = ln(0.05) / ln(0.98) = 148.3, so 149 attempts. (b) p = 0.01, z^2 = 3.8416. denominator = 1 + 3.8416/100 = 1.0384. centre = (0.01 + 0.0192) / 1.0384 = 0.0281. half = 1.96 x sqrt(0.01 x 0.99 / 100 + 3.8416 / 40,000) / 1.0384 = 1.96 x sqrt(0.000099 + 0.000096) / 1.0384 = 1.96 x 0.01396 / 1.0384 = 0.0264. Interval = ~0.2% to ~5.4%. You cannot claim below 2%. You would need more attempts, or to accept a higher bound.

</details>

### P4. Choose the lethal-trifecta cut

A meeting-assistant agent reads the user's calendar and email (private data), opens links found in invitations (untrusted content), and can post summaries to a shared chat channel and send email (exfiltration channels). Name three different architectures that each remove one leg, and the cost of each.

<details><summary>Answer</summary>

(1) Remove untrusted content: never open links from invitations in the privileged session. Cost: no link-based context. (2) Remove private data from the browsing step: a separate quarantined summariser opens links with no mailbox access and returns a typed summary variable. Cost: more components, and the summary cannot use private context. (3) Remove or gate the exfiltration channel: summaries go only to the user's own inbox, and posting to the shared channel or sending external email needs approval showing the real recipient and content. Cost: friction on sharing. Option 3 is cheapest to build, option 2 preserves the most capability if the controller is done properly.

</details>

### P5. Pick a fairness metric

For each, choose the main metric and justify in one sentence: (a) an LLM ranks job applicants for human interview. (b) a bank shows loan officers a default-risk score. (c) a content-moderation classifier flags posts for removal across dialect groups.

<details><summary>Answer</summary>

(a) Equal opportunity (equal TPR): qualified candidates should be shortlisted at equal rates, and also check the four-fifths impact ratio since selection-rate rules often apply. (b) Calibration within groups: a score of 0.2 should mean 20% default risk for every group, since officers interpret the number. (c) Equal FPR across groups: the harm is wrongly removing benign posts of one dialect group, so the false-positive rates must match. In each case document why that metric fits and report the others for transparency.

</details>

### P6. Abstention threshold

A legal-research assistant answers a question correctly with value +1 and wrongly with cost -6. (a) Above what confidence should it answer? (b) The model is calibrated and 60% of questions have confidence 0.95, 25% have 0.80, 15% have 0.50. Compute the expected net value per question when it answers only above the threshold, and when it answers everything.

<details><summary>Answer</summary>

(a) p > 6 / 7 = 0.857. (b) Answering only the 0.95 group: expected value per answered question = 0.95 - 6 x 0.05 = 0.65. Per question overall = 0.6 x 0.65 = 0.39. Answer everything: group values are 0.95 - 0.3 = 0.65, 0.80 - 1.2 = -0.40, 0.50 - 3.0 = -2.50. Overall = 0.6 x 0.65 + 0.25 x (-0.40) + 0.15 x (-2.50) = 0.39 - 0.10 - 0.375 = -0.085. Abstaining on 40% of questions turns a loss into a gain.

</details>

### P7. Design a log policy

You must keep enough logs to investigate an incident but your privacy policy promises minimal retention of user content. Write the policy in five bullet points.

<details><summary>Answer</summary>

(1) Always log metadata: timestamps, user and tenant IDs (pseudonymous), model and prompt versions, tool names and argument hashes, guardrail verdicts, approval records. Retain for the audit period. (2) Store prompt and output text redacted for PII, retained for a short window (for example 30 days). (3) Keep raw text in a separate store, encrypted, access restricted to a small on-call group, with every read logged and a ticket required. (4) Raw retention is shorter than redacted retention, except when legal hold or an open incident applies. (5) Logs are append-only, and deletion requests propagate to all stores including traces and evaluation datasets. Exact windows are a legal and business decision.

</details>

### P8. Counterfactual audit design

You must test a vendor's CV-screening LLM for gender bias with black-box access only. Outline the test and the statistic.

<details><summary>Answer</summary>

Take ~200 real or realistic CVs with identifying details removed. For each, create a variant that differs only in a gendered name and pronouns, using several names per group. Run each variant several times (for example 5) to average over sampling noise. Per pair, record whether the original or the swapped version scores higher or passes the cut. Count discordant pairs b (original wins) and c (swapped wins) and compute z = (b - c) / sqrt(b + c). Report the flip rate and the shift in mean score with intervals. Add intersections (name signals for ethnicity) and check that the swap changes nothing else in the text. If z is large, document and escalate to the vendor, and keep a human reviewing decisions.

</details>
