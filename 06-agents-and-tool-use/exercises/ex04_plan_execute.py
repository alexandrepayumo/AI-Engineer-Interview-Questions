"""Exercise 04 - Plan-and-execute with replanning on failure.

TASK
----
Implement:

    format_replan_prompt(goal, completed, failure) -> str
        The text a planner model would receive after a failure. Exactly these lines
        joined with "\\n":
            "Goal: {goal}"
            "Completed:"
            one line per completed step: "- {step} -> {result}"   (or "- (none)" if empty)
            "Failed: {failure}"
            "Return only the remaining steps."

    run_plan(goal, planner, executor, max_replans=2, max_step_attempts=1,
             max_total_steps=20) -> PlanResult
        planner(goal, completed, failure) -> list[str]
            completed is a list of (step, result) tuples so far, failure is None for the
            first call or a string "{step}: {error message}" after a failure. It returns the
            REMAINING steps (never repeat completed work).
        executor(step) -> str  returns a result or raises StepError(message).

    Algorithm:
      - steps = planner(goal, [], None).
      - While steps is non-empty: if len(completed) >= max_total_steps stop with status
        "step_limit". Try the first step up to max_step_attempts times. On success append
        (step, result) to completed and drop the step. If all attempts fail: if replans >=
        max_replans stop with status "failed", else replans += 1 and
        steps = planner(goal, list(completed), f"{step}: {message}") using the LAST error message.
      - Empty steps means status "done".
      - log gets one string per event: "ok: {step}", "fail: {step}: {message}" (every
        failed attempt), "replan #{n}" (n counts from 1).
    PlanResult(status, completed, replans, log).

HINTS
-----
- Retrying the same step handles transient errors cheaply. Replanning handles
  "this approach cannot work". The two budgets are separate.
- Pass a copy of completed to the planner so it cannot mutate your state.
- The replan budget is what stops a planner that keeps proposing the same doomed step.

WHAT YOU SHOULD LEARN
---------------------
- Plan-and-execute separates the expensive planning call from cheap step execution.
- A plan is a hypothesis about the environment; replanning is how you recover when it is wrong.
- Why completed work is passed back (avoid redoing or contradicting finished steps) and why
  every loop needs a hard budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class StepError(Exception):
    """Raised by an executor when a step fails."""


@dataclass
class PlanResult:
    status: str  # "done" | "failed" | "step_limit"
    completed: list
    replans: int
    log: list


def format_replan_prompt(goal: str, completed: list[tuple[str, str]], failure: str) -> str:
    raise NotImplementedError("TODO")


def run_plan(
    goal: str,
    planner: Callable,
    executor: Callable,
    max_replans: int = 2,
    max_step_attempts: int = 1,
    max_total_steps: int = 20,
) -> PlanResult:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # prompt formatting
    p = format_replan_prompt("find X", [("search", "3 hits")], "fetch: 404")
    assert p == "Goal: find X\nCompleted:\n- search -> 3 hits\nFailed: fetch: 404\nReturn only the remaining steps."
    p0 = format_replan_prompt("g", [], "s: e")
    assert p0 == "Goal: g\nCompleted:\n- (none)\nFailed: s: e\nReturn only the remaining steps."

    # 1. happy path, no replanning
    r = run_plan("g", lambda g, c, f: ["a", "b", "c"], lambda s: s.upper())
    assert r.status == "done" and r.replans == 0
    assert r.completed == [("a", "A"), ("b", "B"), ("c", "C")]
    assert r.log == ["ok: a", "ok: b", "ok: c"]

    # 2. replanning keeps finished work and swaps the broken step
    planner_calls: list = []

    def planner(goal, completed, failure):
        planner_calls.append((list(completed), failure))
        if failure is None:
            return ["search", "fetch_v1", "summarise"]
        return ["fetch_v2", "summarise"]

    def executor(step):
        if step == "fetch_v1":
            raise StepError("404 not found")
        return f"result of {step}"

    r = run_plan("g", planner, executor)
    assert r.status == "done" and r.replans == 1
    assert [s for s, _ in r.completed] == ["search", "fetch_v2", "summarise"]
    assert len(planner_calls) == 2
    assert planner_calls[1] == ([("search", "result of search")], "fetch_v1: 404 not found")
    assert r.log == [
        "ok: search", "fail: fetch_v1: 404 not found", "replan #1",
        "ok: fetch_v2", "ok: summarise",
    ]

    # 3. a planner that keeps proposing the doomed step hits the replan budget
    n = {"plans": 0}

    def stubborn(goal, completed, failure):
        n["plans"] += 1
        return ["x"]

    def always_fail(step):
        raise StepError("no")

    r = run_plan("g", stubborn, always_fail, max_replans=2)
    assert r.status == "failed" and r.replans == 2 and n["plans"] == 3 and r.completed == []

    # 4. transient failure is absorbed by a step retry, no replan needed
    attempts = {"n": 0}

    def flaky(step):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise StepError("timeout")
        return "ok"

    r = run_plan("g", lambda g, c, f: ["only"], flaky, max_step_attempts=2)
    assert r.status == "done" and r.replans == 0 and attempts["n"] == 2
    assert r.log == ["fail: only: timeout", "ok: only"]
    # without retries the same flake forces a replan
    attempts["n"] = 0
    r = run_plan("g", lambda g, c, f: ["only"], flaky, max_step_attempts=1)
    assert r.status == "done" and r.replans == 1

    # 5. step cap
    r = run_plan("g", lambda g, c, f: [f"s{i}" for i in range(100)], lambda s: "x", max_total_steps=5)
    assert r.status == "step_limit" and len(r.completed) == 5

    # 6. empty plan is trivially done; planner output is not aliased
    r = run_plan("g", lambda g, c, f: [], lambda s: "x")
    assert r.status == "done" and r.completed == [] and r.log == []

    seen: list = []

    def mutating_planner(goal, completed, failure):
        seen.append(completed)
        completed.append(("hack", "hack"))  # must not leak into the run's own state
        return ["a"] if failure is None else ["b"]

    def exec_a_fails(step):
        if step == "a":
            raise StepError("x")
        return "ok"

    r = run_plan("g", mutating_planner, exec_a_fails)
    assert ("hack", "hack") not in r.completed


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
