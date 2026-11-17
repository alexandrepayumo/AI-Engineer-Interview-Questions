# Agents, Tool Use & MCP - Exercises

Practice material for [../deep-dive.md](../deep-dive.md). The coding exercises use Python 3.12 and the standard library (numpy in one case), run in under 10 seconds each, use fixed seeds and need no network.

## How to use

1. Read the matching part of the deep dive.
2. Open the stub, for example `ex01_schema_validator.py`. Read the docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and replace each `raise NotImplementedError("TODO")` with your implementation. Do not edit `_run_tests`.
3. Run it from this folder: `python ex01_schema_validator.py`. Untouched stubs fail with `NotImplementedError`, which is expected. When your code is right it prints `All tests passed.`
4. Compare with `solutions/ex01_schema_validator.py`. The tests are identical, so the solution is one valid answer, not the only one. Read its comments on key steps.

Further practice that overlaps this topic but is not repeated here: [a basic agent loop](../../12-coding-challenges/10_agent_loop.py), [rate limiting and retry](../../12-coding-challenges/11_rate_limiter_and_retry.py), [evaluation metrics](../../12-coding-challenges/12_eval_metrics.py), [constrained JSON decoding](../../12-coding-challenges/18_constrained_json_decoding.py).

## Coding exercises

| # | File | Difficulty | What it teaches | Est. time |
|---|---|---|---|---|
| 1 | `ex01_schema_validator.py` | Easy | JSON-Schema subset validation, the bool-is-int trap, error messages a model can act on (Parts 2, 4) | 40 min |
| 2 | `ex02_react_loop.py` | Medium | Agent loop with scripted model, iteration and token budgets, duplicate-call guard, errors as observations (Parts 1, 3, 4, 8) | 60 min |
| 3 | `ex03_dag_executor.py` | Medium | Topological levels, critical path, limited-worker makespan, failure isolation (Part 6) | 60 min |
| 4 | `ex04_plan_execute.py` | Medium | Plan-and-execute, replanning with completed work, separate retry and replan budgets (Part 3) | 45 min |
| 5 | `ex05_agent_memory.py` | Medium | Recency, relevance and importance scoring, decay with half-life, reinforcement, pruning (Part 5) | 60 min |
| 6 | `ex06_taint_guard.py` | Hard | Capability guard with taint tracking, lethal trifecta, egress allowlist, approval tiers (Part 9) | 75 min |
| 7 | `ex07_pass_k.py` | Medium | Unbiased pass@k and $\text{pass}^k$ estimators, aggregation over tasks, Wilson intervals (Part 10) | 45 min |

Suggested order: 1, 2, 4, 3, 5, 6, 7.

## Pen-and-paper and design exercises

Attempt each before opening the answer. Use a calculator only for powers and logarithms.

### P1. Loop cost with and without caching

A loop has a fixed prefix of $S = 6{,}000$ tokens (system prompt and tool definitions). Each step adds $t = 1{,}500$ tokens. It makes 12 calls. Compute the total input tokens without caching, then the effective cost in full-price token equivalents if cached reads cost 10% of the normal rate.

<details><summary>Answer</summary>

Without caching:

$$12 \times 6{,}000 + 1{,}500 \times \frac{12 \times 11}{2} = 72{,}000 + 99{,}000 = 171{,}000 \text{ tokens}$$

With caching, only new tokens are uncached: $6{,}000 + 11 \times 1{,}500 = 22{,}500$. Cached reads $= 171{,}000 - 22{,}500 = 148{,}500$, which cost $14{,}850$ equivalents. Total $= 22{,}500 + 14{,}850 = 37{,}350$, about 4.6x cheaper. The saving depends on a byte-stable prefix. Changing anything near the top of the prompt each turn removes it.

</details>

### P2. Hitting 90% on a 12-step task

Per-step success is 0.97. (a) What is the 12-step success rate? (b) A recovery mechanism fixes 70% of step failures. What is it now? (c) Alternatively, consolidating tools halves the steps to 6. (d) What if you do both?

<details><summary>Answer</summary>

(a) $0.97^{12} = 0.694$. (b) The effective step failure is $0.03 \times 0.3 = 0.009$, so $p = 0.991$ and $0.991^{12} = 0.897$. (c) $0.97^6 = 0.833$. (d) $0.991^6 = 0.947$. Recovery helps more than consolidation here, and the two combine. Real failures correlate, so treat these as upper estimates and measure end-to-end.

</details>

### P3. Workflow or agent?

Decide for each and justify in one sentence: (a) classify support emails into five queues and draft a templated reply. (b) Investigate why a nightly data pipeline failed using logs, schema diffs and recent commits. (c) Extract line items from invoices and write them to a ledger.

<details><summary>Answer</summary>

(a) Workflow: routing plus a fixed drafting step, with a known set of branches. (b) Agent: the number and order of lookups depends on what each log reveals, and the outcome (a diagnosis) can be checked. Keep it read-only. (c) Workflow: parse, validate, write. No branching worth model-directed control, and a ledger write deserves deterministic validation.

</details>

### P4. Scheduling with a DAG

Tasks and durations: a = 2, b = 3, c = 1, d = 4, e = 2. Dependencies: c needs a, d needs a, e needs b and c. Compute the levels, the critical path, and the makespan with 2 workers using alphabetical greedy scheduling (start the alphabetically first ready task whenever a worker is free). Can a different rule do better?

<details><summary>Answer</summary>

Levels: [a, b], [c, d], [e]. Critical path: $a \to d = 6$, $a \to c \to e = 5$, $b \to e = 5$, so 6. Total work is 12, so with 2 workers the lower bound is $\max(6, 12/2) = 6$. Alphabetical greedy: at $t = 0$ start a and b. At $t = 2$, a finishes and both c and d are ready but only one worker is free, so c starts (finishes at 3). At $t = 3$, b and c are done, d starts (finishes at 7), e starts at 3 (finishes at 5). Makespan = 7. Starting d first at $t = 2$ (a rule that prioritises the longest remaining path) gives d finishing at 6, c running on the other worker from 3 to 4, e from 4 to 6, so the makespan is 6, equal to the lower bound. Order matters when workers are scarce, so schedulers prioritise by critical-path length.

</details>

### P5. Memory ranking

Half-life 5, weights (1, 1, 1), now = 20, query vector fixed. Memory A: last access 15, importance 0.2, relevance 0.9. Memory B: last access 0, importance 0.9, relevance 0.6. Memory C: last access 19, importance 0.1, relevance 0.3. Rank them. After B is retrieved once, how does its score change?

<details><summary>Answer</summary>

Recency: A $= 0.5^{5/5} = 0.5$, B $= 0.5^{20/5} = 0.0625$, C $= 0.5^{1/5} = 0.871$. Scores: A $= 0.5 + 0.9 + 0.2 = 1.6$. B $= 0.0625 + 0.6 + 0.9 = 1.5625$. C $= 0.871 + 0.3 + 0.1 = 1.271$. Order: A, B, C. After B is retrieved, its last access becomes 20, so its recency is 1.0 and its score is $1 + 0.6 + 0.9 = 2.5$, which puts it first. Reinforcement is a feedback loop: whatever gets retrieved becomes more retrievable. That is usually desirable, but it can lock in a stale fact, so contradictions need explicit handling.

</details>

### P6. Trifecta audit

A coding assistant reads the private repository, runs shell commands with outbound network access, and fetches documentation pages from the web. Which trifecta legs are present? Give two structural changes that break it, and say what each costs.

<details><summary>Answer</summary>

All three: private data (the repo), untrusted content (web pages, and also third-party dependency READMEs and issue text), and an exfiltration channel (network-enabled shell, and the web fetch tool itself). Options: (1) Run the shell in a sandbox with no network or an egress proxy allowing only the package registry. This costs the ability to call arbitrary APIs. (2) Split roles: a research agent fetches web pages and returns a validated, typed summary with no repo access, and the coding agent never touches raw web text. This costs an extra hop and some lost detail. Adding approval for network commands also helps, with the approval-fatigue caveat.

</details>

### P7. Reliability metrics from trial counts

Five tasks were run 4 times each with these success counts: 4, 4, 3, 2, 0. Compute pass@1, pass@2, $\text{pass}^2$ and $\text{pass}^4$, averaged over tasks.

<details><summary>Answer</summary>

$\text{pass@}1 = (1 + 1 + 0.75 + 0.5 + 0)/5 = 0.65$. $\text{pass}^2$ = mean of $\binom{c}{2}/\binom{4}{2}$ with $\binom{4}{2} = 6$: $(1 + 1 + 3/6 + 1/6 + 0)/5 = 2.667/5 = 0.533$. $\text{pass}^4$ is 1 only for the tasks with $c = 4$, so $2/5 = 0.4$. $\text{pass@}2 = 1 - \binom{4 - c}{2}/6$ per task: 1, 1, 1 (only one failure exists), $1 - 1/6 = 0.833$, 0, so the mean is $3.833/5 = 0.767$. The 0.65 to 0.40 drop shows how fast consistency decays.

</details>

### P8. Debugging a stuck agent

In production, 4% of runs hit the 30-iteration cap. Traces show many of them calling `search_docs` with slightly different queries and getting empty results. List four checks in order.

<details><summary>Answer</summary>

1. Look at the empty results. Do they come from a real "no match" or from an error swallowed into an empty list? If errors are hidden, return them as explicit observations. 2. Check the index or data source for the target documents (coverage), since the agent may be searching for something that is not there. 3. Review the tool description and result format. Does it say what to do when nothing is found ("try fewer keywords, or call list_collections")? Add a recovery hint and a cap on consecutive empty searches that triggers escalation. 4. Add duplicate and near-duplicate detection and a stop rule such as "after 3 empty searches, report not found". Then re-run the failing traces as a regression set and track the cap-hit rate.

</details>

### P9. Idempotency keys

You build `create_refund(order_id, amount)` and the runtime retries on timeout. Design the idempotency key and say what happens in two cases: (a) the retry has the same arguments after a lost response, (b) the model, in a later turn, calls the tool again with a different amount for the same order.

<details><summary>Answer</summary>

Derive the key from the logical action, for example run id + step id + tool name, or a hash of (order_id, amount, reason) scoped to the run. (a) The retry reuses the key, the server returns the stored first result, and no second refund is created. (b) A new model decision is a new action. If the key includes the amount or step id it creates a second refund, which may be wrong. Add a business rule: refunds on an order cannot exceed the order total and a second refund needs approval. Idempotency stops duplicates from retries. It does not stop mistaken new actions, so pair it with limits and approval.

</details>

### P10. Do you need subagents?

A research task needs four independent investigations. Each reads about 80,000 tokens of material. Each could produce a 1,200-token summary. The model has a 200,000-token context window. Compare a single agent with an orchestrator and four workers on context use and total tokens.

<details><summary>Answer</summary>

A single agent would accumulate $4 \times 80{,}000 = 320{,}000$ tokens of reading, which exceeds the 200,000 window, so it needs aggressive clearing or compaction mid-task and risks losing detail. With four workers, each uses its own 80,000-token context. The orchestrator's context grows by $4 \times 1{,}200 = 4{,}800$ tokens. Total tokens processed is still at least 320,000 plus overhead, so cost is not lower, and multi-agent runs often cost several times a single chat. The gain is isolation and parallel wall-clock time, and it pays off only if the answer quality justifies the spend.

</details>
