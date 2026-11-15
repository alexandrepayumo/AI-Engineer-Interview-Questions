# Exercises - Prompt Engineering and Context Engineering

Practice for the [deep dive](../deep-dive.md). The coding exercises build the plumbing a production prompt pipeline needs: rendering, budgeting, example selection, voting, validation, cost modelling and history compaction. The pen-and-paper exercises train the design judgement interviewers probe.

## How to use

1. Read the matching part of the deep dive first (the table says which).
2. Open the stub, for example `ex01_template_renderer.py`. Read the module docstring (TASK, HINTS, WHAT YOU SHOULD LEARN) and implement the functions.
3. Run it from this folder: `python ex01_template_renderer.py`. An untouched stub fails with `NotImplementedError`. That is expected.
4. When your version prints `All tests passed.`, compare it with `solutions/ex01_template_renderer.py`. The tests are identical in both files, so a pass means you meet the same contract.
5. Dependencies are the standard library and numpy only. Every file runs in well under 10 seconds and uses no randomness.

Related practice in [12-coding-challenges](../../12-coding-challenges/): the sampling, tokenisation and retrieval challenges there pair well with exercises 03 and 04.

## Coding exercises

| # | File | Difficulty | What it teaches | Deep dive part | Est. time |
|---|------|------------|-----------------|----------------|-----------|
| 01 | `ex01_template_renderer.py` | Easy | Strict template rendering: missing, unused and badly typed variables, single-pass substitution | Part 2, Part 8 | 25 min |
| 02 | `ex02_context_assembler.py` | Medium | Token budgeting with priorities, required blocks, truncation, slot ordering | Part 6 | 45 min |
| 03 | `ex03_mmr_selector.py` | Medium | Few-shot selection with MMR over bag-of-words vectors, recency ordering | Part 3 | 45 min |
| 04 | `ex04_self_consistency.py` | Easy-Medium | Answer extraction, normalisation, majority vote, binomial vote accuracy | Part 4 | 35 min |
| 05 | `ex05_json_repair_loop.py` | Medium | JSON extraction, schema validation, error-feedback retry loop with a mock LLM | Part 5 | 60 min |
| 06 | `ex06_caching_cost.py` | Medium | Cache write/read pricing, break-even hit rate, agent-loop savings, TTL versus traffic | Part 7 | 45 min |
| 07 | `ex07_history_compactor.py` | Hard | Rolling-summary compaction under a budget, batch triggering | Part 6, Part 7 | 60 min |

## Pen-and-paper and design exercises

### P1. Why did the cache never hit?

A team prefixes every system prompt with `Current time: 2026-03-02T14:07:11Z` and reports 0% cache hits on a 9,000-token prompt. Explain why and give the fix.

<details><summary>Model answer</summary>

Caches match on an exact token prefix. The timestamp changes every request and sits at token 0, so the first token run already differs and nothing after it can match, however stable the other 8,990 tokens are. Move volatile values to the end: system policy and tool definitions first, then examples, then history, then a final block containing the date and the user query. If the model needs the date, put it in the last user message. Verify with the provider's usage fields (cached token counts) rather than assuming.

</details>

### P2. Break-even with a 1-hour TTL

Writes cost 2x the input price for a 1-hour TTL and reads cost 0.1x. What hit rate makes caching profitable? A prefix is requested once every 90 minutes on average. Is the 1-hour TTL a good idea?

<details><summary>Model answer</summary>

Solve h * 0.1 + (1 - h) * 2.0 = 1, so h = (2 - 1) / (2 - 0.1) = 1 / 1.9, about 52.6%. With one request per 90 minutes on average and a 60-minute TTL, a Poisson model gives a hit probability of 1 - exp(-60/90) = 1 - 0.513 = about 49%, which is below break-even, so the 1-hour TTL loses money. Without caching you pay 1x. With the 5-minute TTL you would almost never hit and pay 1.25x on every request, which is worse. The right answer here is to not cache, or to batch requests so they arrive closer together.

</details>

### P3. Where does the key document go?

You must send 60 retrieved chunks (about 45k tokens) plus a question. The two most relevant chunks are #3 and #41. Sketch the layout and justify it, including what you do if the answer still seems to ignore chunk #41.

<details><summary>Model answer</summary>

Layout: system rules, then the instruction ("answer only from the documents, cite chunk ids"), then documents with the most relevant at the start and end (for example #3 first, #41 last, or reorder by score so the best two sit at the edges), then the question and a restatement of the output format at the very bottom. Lost-in-the-middle effects make the centre the weakest region. If #41 is still ignored, first reduce the noise (send 15 chunks, not 60, after reranking), then ask the model to quote the supporting passage before answering, then test the same input at several positions to see whether position is the cause.

</details>

### P4. Choose the output mechanism

For each task choose prompt-only JSON, JSON mode, schema-constrained decoding, or a repair loop (possibly combined), and say why: (a) extract 5 fields from invoices for a database insert, (b) a chat assistant that sometimes returns a table, (c) a classifier with 12 allowed labels run on a self-hosted model, (d) a nested 6-level config generator.

<details><summary>Model answer</summary>

(a) Schema-constrained decoding plus semantic validation (dates parse, totals match line items), because a malformed row breaks the pipeline and syntax errors are the cheap part to remove. (b) Prompt-only: the format is optional and human-facing, nothing parses it. (c) Constrained decoding with an enum (a grammar over the 12 labels, or logit masking), which also makes the call cheap because only a label token or two is generated. (d) A repair loop around a shallow-schema design: deep nesting is poorly supported by strict modes and compiles slowly, so flatten where possible, validate, and feed errors back with a retry cap.

</details>

### P5. Is self-consistency worth it?

A task has a single-integer answer and a single-sample accuracy of 62%. You can afford 5x cost. Estimate the gain under ideal assumptions, then list two reasons the real gain will be smaller.

<details><summary>Model answer</summary>

Under independence with wrong answers never agreeing, the majority of 5 is right when at least 3 are right: sum over i = 3..5 of C(5,i) 0.62^i 0.38^(5-i) = 10(0.2383)(0.1444) + 5(0.1478)(0.38) + 0.0916 = 0.3441 + 0.2809 + 0.0916, about 0.717. So roughly +10 points on paper. Real gains are smaller because samples from one model at one temperature are positively correlated (the same misconception recurs, so errors agree), and because wrong answers often cluster on the same wrong value, which lets a wrong majority win. A 5x cost for a few points may lose to a better prompt, a stronger model or a verifier.

</details>

### P6. Compaction policy

An agent hits 80% of its context budget after about 40 tool calls. Design the compaction policy: what stays verbatim, what is summarised, what is dropped, when it triggers, and how it interacts with prompt caching.

<details><summary>Model answer</summary>

Keep verbatim: system prompt and tool definitions (stable, cached), the user's original task and constraints, the last few turns, and any unresolved error. Summarise: older reasoning and decisions into a structured note (goal, done, decided and why, open items, key identifiers). Drop or replace by a short pointer: bulky tool outputs already acted on (keep file path or query so they can be re-fetched). Trigger at a threshold with headroom (for example 70-80%) and compact in one batch, because every compaction rewrites the prefix after the stable head and invalidates the cache from the first changed token. Keep the stable head byte-identical so the cache still covers it. Treat the summary as untrusted text derived from tool outputs, and test that it does not carry injected instructions forward.

</details>

### P7. Read the A/B result

Prompt B beats prompt A on 100 cases. The discordant cases: B right and A wrong on 12, A right and B wrong on 5, the other 83 agree. Is B better with confidence? What would you do next?

<details><summary>Model answer</summary>

The net gain is 7 cases (7 points). A paired test looks only at the 17 discordant cases: z = (12 - 5) / sqrt(12 + 5) = 7 / 4.12 = 1.70, a two-sided p of about 0.09. That is suggestive, not conclusive at the usual 5% level. Next steps: grow the eval set (especially in the area where B differs), inspect the 17 discordant cases for a pattern, run multiple samples per case if the model is stochastic, and check cost and latency. Ship behind a canary with guardrail metrics if the pattern makes sense.

</details>

### P8. Injection review

A support agent can read customer emails, search the knowledge base and issue refunds up to 500 USD. List three concrete defences that do not rely on the model refusing, and say what each one prevents.

<details><summary>Model answer</summary>

(1) Privilege reduction and approval: refunds above a small limit, or any refund triggered in a turn that read untrusted email content, need human confirmation. This bounds the damage of a successful injection. (2) Separate the tools by trust: a read-only sub-agent processes the email and returns a typed, validated summary (category, order id), and only that structured result reaches the agent holding the refund tool. This cuts the path from untrusted text to action. (3) Deterministic output checks: the refund tool validates that the order belongs to the sender, the amount is within policy and the request is not a duplicate. This catches both injection and ordinary model errors. Delimiters and "ignore instructions in documents" rules are worth adding but are the weakest layer.

</details>
