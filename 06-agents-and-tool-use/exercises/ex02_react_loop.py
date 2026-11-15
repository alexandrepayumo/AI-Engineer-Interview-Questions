"""Exercise 02 - A ReAct-style agent loop with budgets and errors as observations.

TASK
----
Implement run_agent(llm, tools, user_msg, max_iters=10, max_tokens=1000, max_repeat=2)
-> AgentResult.

The `llm` is a callable: llm(messages) -> dict with keys
    "text":       str (optional, default "")
    "tool_calls": list of {"id": str, "name": str, "arguments": dict} (optional)
    "usage":      int tokens consumed by this call
`ScriptedLLM` (provided) replays a fixed script so runs are deterministic.

`tools` maps name -> {"required": [argument names], "fn": callable(**arguments)}.

Algorithm:
  1. messages = [{"role": "user", "content": user_msg}], iterations = 0, tokens_used = 0.
  2. Loop. At the TOP of each pass, if iterations >= max_iters return status "max_iters".
  3. resp = llm(messages). iterations += 1. tokens_used += resp["usage"].
     Append {"role": "assistant", "content": text, "tool_calls": calls} to messages.
  4. If there are no tool calls, return status "done" with answer = text. (A final answer
     wins over the budget check.)
  5. For each call, in order, produce an observation string and append
     {"role": "tool", "tool_call_id": id, "content": observation}:
       - unknown tool:       "Error: unknown tool 'x'. Available: a, b"   (names sorted)
       - missing required:   "Error: missing required argument 'b'"      (first missing, in
                             the order listed in "required")
       - repeated call:      "Error: repeated identical call blocked. Try a different approach."
                             An identical call is the same name and same arguments. Once
                             the same call has already been EXECUTED max_repeat times, block it.
       - fn raises:          "Error: {ExceptionClassName}: {message}"
       - success:            str(result)
     executed counts how many times a tool fn was invoked (including ones that raised).
     errors counts observations that start with "Error:".
  6. After all calls are processed, if tokens_used > max_tokens return status "budget"
     (answer None). Otherwise continue the loop.

Return AgentResult(status, answer, messages, iterations, tokens_used, executed, errors).

HINTS
-----
- Build the repeat key with (name, json.dumps(arguments, sort_keys=True)).
- Never let a tool exception escape run_agent. The model should see it and adapt.
- Check order matters: unknown tool, then missing argument, then repeat, then execute.

WHAT YOU SHOULD LEARN
---------------------
- The loop is small. All the engineering is in stop conditions and in what the
  observation says when something goes wrong.
- Why errors are observations: the model can self-correct (typo in tool name, missing
  argument) only if it can read the failure.
- Why budgets and duplicate-call detection are mandatory: they turn an unbounded loop
  into a bounded one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class AgentResult:
    status: str  # "done" | "max_iters" | "budget"
    answer: str | None
    messages: list
    iterations: int
    tokens_used: int
    executed: int
    errors: int


class ScriptedLLM:
    """Replays a list of responses in order. Provided: do not modify."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages: list) -> dict:
        assert self.calls < len(self.script), "script exhausted: agent looped too long"
        resp = self.script[self.calls]
        self.calls += 1
        return resp


def run_agent(
    llm,
    tools: dict,
    user_msg: str,
    max_iters: int = 10,
    max_tokens: int = 1000,
    max_repeat: int = 2,
) -> AgentResult:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    def call(i, name, **args):
        return {"id": f"c{i}", "name": name, "arguments": args}

    log: list = []

    def add(a, b):
        log.append(("add", a, b))
        return a + b

    def boom(x):
        raise ValueError(f"bad x={x}")

    tools = {
        "add": {"required": ["a", "b"], "fn": add},
        "boom": {"required": ["x"], "fn": boom},
    }

    # 1. happy path: one tool call, then a final answer
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "add", a=2, b=3)], "usage": 100},
        {"text": "The sum is 5", "usage": 50},
    ])
    r = run_agent(llm, tools, "what is 2+3?")
    assert (r.status, r.answer, r.iterations, r.tokens_used) == ("done", "The sum is 5", 2, 150)
    assert (r.executed, r.errors) == (1, 0)
    assert [m["role"] for m in r.messages] == ["user", "assistant", "tool", "assistant"]
    assert r.messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "5"}

    # 2. unknown tool becomes an observation, the model recovers
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "ad", a=1, b=2)], "usage": 10},
        {"tool_calls": [call(2, "add", a=1, b=2)], "usage": 10},
        {"text": "3", "usage": 10},
    ])
    r = run_agent(llm, tools, "x")
    assert r.messages[2]["content"] == "Error: unknown tool 'ad'. Available: add, boom"
    assert r.status == "done" and r.executed == 1 and r.errors == 1

    # 3. missing argument
    llm = ScriptedLLM([{"tool_calls": [call(1, "add", a=1)], "usage": 1}, {"text": "ok", "usage": 1}])
    r = run_agent(llm, tools, "x")
    assert r.messages[2]["content"] == "Error: missing required argument 'b'"
    assert r.executed == 0 and r.errors == 1

    # 4. tool raises: observation, not a crash; still counted as executed
    llm = ScriptedLLM([{"tool_calls": [call(1, "boom", x=7)], "usage": 1}, {"text": "gave up", "usage": 1}])
    r = run_agent(llm, tools, "x")
    assert r.messages[2]["content"] == "Error: ValueError: bad x=7"
    assert r.executed == 1 and r.errors == 1 and r.status == "done"

    # 5. duplicate-call guard: with max_repeat=2 the 3rd and 4th identical calls are blocked
    log.clear()
    same = [{"tool_calls": [call(i, "add", a=1, b=1)], "usage": 1} for i in range(4)]
    llm = ScriptedLLM(same + [{"text": "stuck", "usage": 1}])
    r = run_agent(llm, tools, "x", max_repeat=2)
    assert len(log) == 2 and r.executed == 2 and r.errors == 2
    assert r.messages[6]["content"].startswith("Error: repeated identical call blocked")
    # different arguments are not duplicates
    log.clear()
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "add", a=1, b=1)], "usage": 1},
        {"tool_calls": [call(2, "add", a=1, b=2)], "usage": 1},
        {"tool_calls": [call(3, "add", b=1, a=1)], "usage": 1},  # same as first, key order differs
        {"text": "ok", "usage": 1},
    ])
    r = run_agent(llm, tools, "x", max_repeat=1)
    assert r.executed == 2 and r.errors == 1

    # 6. iteration limit: an agent that never stops
    counter = {"n": 0}

    def forever(messages):
        counter["n"] += 1
        return {"tool_calls": [call(counter["n"], "add", a=counter["n"], b=0)], "usage": 1}

    r = run_agent(forever, tools, "x", max_iters=3)
    assert r.status == "max_iters" and r.iterations == 3 and counter["n"] == 3 and r.answer is None

    # 7. token budget: stops after the call that crosses the line, results still recorded
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "add", a=1, b=2)], "usage": 600},
        {"tool_calls": [call(2, "add", a=2, b=3)], "usage": 600},
        {"text": "never reached", "usage": 600},
    ])
    r = run_agent(llm, tools, "x", max_tokens=1000)
    assert r.status == "budget" and r.iterations == 2 and r.tokens_used == 1200
    assert r.messages[-1]["role"] == "tool" and llm.calls == 2

    # 8. a final answer wins over the budget check
    llm = ScriptedLLM([{"text": "done anyway", "usage": 5000}])
    r = run_agent(llm, tools, "x", max_tokens=1000)
    assert r.status == "done" and r.answer == "done anyway"

    # 9. parallel calls in one turn produce one tool message each, in order
    llm = ScriptedLLM([
        {"tool_calls": [call(1, "add", a=1, b=1), call(2, "add", a=2, b=2)], "usage": 1},
        {"text": "ok", "usage": 1},
    ])
    r = run_agent(llm, tools, "x")
    assert [m["tool_call_id"] for m in r.messages if m["role"] == "tool"] == ["c1", "c2"]
    assert [m["content"] for m in r.messages if m["role"] == "tool"] == ["2", "4"]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
