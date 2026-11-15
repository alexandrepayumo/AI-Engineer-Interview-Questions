"""Exercise 03 - Dependency-aware tool execution (DAG, levels, failure propagation).

TASK
----
A model that emits several tool calls in one turn often implies dependencies: step C
needs the output of A and B. Implement:

    topo_levels(deps) -> list[list[str]]
        deps maps task name -> list of task names it depends on. Return the tasks grouped
        into levels: level 0 has no dependencies, level k holds tasks whose dependencies
        are all in levels < k and that have at least one dependency in level k-1. Names
        inside a level are sorted alphabetically. Raise ValueError("unknown dependency: x")
        if a dependency is not a key of deps, and ValueError("cycle detected") for cycles.

    critical_path(deps, durations) -> float
        Length of the longest dependency chain, summing durations along the path. This
        is the minimum possible wall-clock time with unlimited workers.

    schedule_makespan(deps, durations, workers) -> float
        Wall-clock time with `workers` parallel slots. Greedy list scheduling: whenever a
        worker is free, start the alphabetically first task whose dependencies are all
        finished. Advance time to the next task completion. Return the finish time of
        the last task (0.0 for an empty graph).

    execute(tasks) -> dict
        tasks maps name -> (deps, fn), where fn(inputs: dict[str, object]) -> object and
        inputs maps each dependency name to its result. Run level by level (alphabetical
        within a level). Return {"results": {name: value}, "status": {name: s}, "levels": levels}
        where s is "ok", "failed" (fn raised; store nothing in results) or "skipped" (any
        dependency failed or was skipped; fn is not called).

HINTS
-----
- Kahn's algorithm gives levels directly: repeatedly take all nodes with indegree 0.
- critical_path is a DP over the topological order: finish[n] = durations[n] + max(finish[d]).
- A failed task must not stop independent branches. Only its descendants are skipped.

WHAT YOU SHOULD LEARN
---------------------
- Parallel tool calls are only safe when they are independent; dependencies define waves.
- Critical path versus total work: the speedup bound for fan-out (Amdahl in miniature).
- Failure isolation: one broken branch should not discard work in the others.
"""
from __future__ import annotations

from typing import Callable


def topo_levels(deps: dict[str, list[str]]) -> list[list[str]]:
    for name, ds in deps.items():
        for d in ds:
            if d not in deps:
                raise ValueError(f"unknown dependency: {d}")
    remaining = {n: set(ds) for n, ds in deps.items()}
    placed: set[str] = set()
    levels: list[list[str]] = []
    while remaining:
        # everything whose dependencies are all placed is ready this wave
        ready = sorted(n for n, ds in remaining.items() if ds <= placed)
        if not ready:  # nodes left but none ready: there is a cycle
            raise ValueError("cycle detected")
        levels.append(ready)
        placed.update(ready)
        for n in ready:
            del remaining[n]
    return levels


def critical_path(deps: dict[str, list[str]], durations: dict[str, float]) -> float:
    finish: dict[str, float] = {}
    for level in topo_levels(deps):  # level order guarantees deps are done first
        for n in level:
            start = max((finish[d] for d in deps[n]), default=0.0)
            finish[n] = start + durations[n]
    return max(finish.values(), default=0.0)


def schedule_makespan(deps: dict[str, list[str]], durations: dict[str, float], workers: int) -> float:
    topo_levels(deps)  # validates the graph
    done: set[str] = set()
    started: set[str] = set()
    running: list[tuple[float, str]] = []  # (finish_time, name)
    t = 0.0
    while len(done) < len(deps):
        ready = sorted(n for n in deps if n not in started and set(deps[n]) <= done)
        while ready and len(running) < workers:  # fill free workers, alphabetical order
            n = ready.pop(0)
            started.add(n)
            running.append((t + durations[n], n))
        running.sort()
        t, n = running.pop(0)  # jump to the next completion
        done.add(n)
    return float(t)


def execute(tasks: dict[str, tuple[list[str], Callable]]) -> dict:
    deps = {n: list(ds) for n, (ds, _) in tasks.items()}
    levels = topo_levels(deps)
    results: dict[str, object] = {}
    status: dict[str, str] = {}
    for level in levels:
        for n in level:
            ds, fn = tasks[n]
            if any(status[d] != "ok" for d in ds):  # failed or skipped upstream
                status[n] = "skipped"
                continue
            try:
                results[n] = fn({d: results[d] for d in ds})
                status[n] = "ok"
            except Exception:
                status[n] = "failed"
    return {"results": results, "status": status, "levels": levels}


def _run_tests() -> None:
    # levels
    deps = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    assert topo_levels(deps) == [["a"], ["b", "c"], ["d"]]
    assert topo_levels({}) == []
    assert topo_levels({"x": [], "y": [], "z": []}) == [["x", "y", "z"]]
    # a long chain next to a short one: levels follow the longest chain to each node
    deps2 = {"a": [], "b": ["a"], "c": ["b"], "d": [], "e": ["d", "c"]}
    assert topo_levels(deps2) == [["a", "d"], ["b"], ["c"], ["e"]]

    for bad, msg in [
        ({"a": ["b"], "b": ["a"]}, "cycle detected"),
        ({"a": ["a"]}, "cycle detected"),
        ({"a": ["zzz"]}, "unknown dependency: zzz"),
    ]:
        try:
            topo_levels(bad)
        except ValueError as e:
            assert str(e) == msg, str(e)
        else:
            raise AssertionError("expected ValueError")

    # critical path and makespan on a diamond: a(1) -> b(3), c(2) -> d(1)
    dur = {"a": 1.0, "b": 3.0, "c": 2.0, "d": 1.0}
    assert critical_path(deps, dur) == 5.0
    assert schedule_makespan(deps, dur, workers=1) == 7.0  # total work
    assert schedule_makespan(deps, dur, workers=2) == 5.0  # reaches the critical path
    assert schedule_makespan(deps, dur, workers=8) == 5.0  # more workers do not help
    assert schedule_makespan({}, {}, workers=2) == 0.0

    # independent fan-out: 4 tasks of 2s each
    fan = {f"t{i}": [] for i in range(4)}
    fd = {n: 2.0 for n in fan}
    assert critical_path(fan, fd) == 2.0
    assert schedule_makespan(fan, fd, 1) == 8.0
    assert schedule_makespan(fan, fd, 2) == 4.0
    assert schedule_makespan(fan, fd, 4) == 2.0
    # makespan is never below the critical path or total_work / workers
    for w in (1, 2, 3):
        m = schedule_makespan(deps2, {"a": 2, "b": 1, "c": 4, "d": 3, "e": 1}, w)
        total = 2 + 1 + 4 + 3 + 1
        assert m >= critical_path(deps2, {"a": 2, "b": 1, "c": 4, "d": 3, "e": 1}) - 1e-9
        assert m >= total / w - 1e-9

    # execute: results flow along edges
    tasks = {
        "price": ([], lambda inp: 10),
        "qty": ([], lambda inp: 3),
        "total": (["price", "qty"], lambda inp: inp["price"] * inp["qty"]),
        "label": (["total"], lambda inp: f"total={inp['total']}"),
    }
    out = execute(tasks)
    assert out["results"]["label"] == "total=30"
    assert set(out["status"].values()) == {"ok"}
    assert out["levels"] == [["price", "qty"], ["total"], ["label"]]

    # failure isolation: a failed task skips only its descendants
    calls: list[str] = []

    def mk(name, fail=False):
        def fn(inp):
            calls.append(name)
            if fail:
                raise RuntimeError("boom")
            return name
        return fn

    tasks = {
        "a": ([], mk("a")),
        "b": (["a"], mk("b", fail=True)),
        "c": (["a"], mk("c")),
        "d": (["b", "c"], mk("d")),
        "e": (["d"], mk("e")),
        "f": ([], mk("f")),
    }
    out = execute(tasks)
    assert out["status"] == {"a": "ok", "f": "ok", "b": "failed", "c": "ok", "d": "skipped", "e": "skipped"}
    assert calls == ["a", "f", "b", "c"]  # d and e never ran
    assert out["results"] == {"a": "a", "f": "f", "c": "c"}


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
