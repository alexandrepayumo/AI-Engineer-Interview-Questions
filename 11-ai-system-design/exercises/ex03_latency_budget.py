"""Exercise 03 - Latency budget: critical path, Monte Carlo p95 and budget allocation.

TASK
----
A request is a DAG of stages. `deps` maps each stage to the list of stages that must
finish before it starts. Stages with no dependency relation run in parallel.

    topo_order(deps) -> list[str]
        A valid topological order (dependencies first). Stages that only appear as a
        dependency count as stages with no dependencies of their own. Raise ValueError on
        a cycle. Ties are broken alphabetically so the output is deterministic.

    critical_path(latency, deps) -> tuple[float, list[str]]
        latency maps stage -> milliseconds. Return (end-to-end time, stages on the longest
        path in execution order). Raise ValueError if deps mention a stage missing from
        latency, or on a cycle.

    sample_end_to_end(stages, deps, n, seed) -> np.ndarray
        stages maps name -> (median_ms, sigma). Each stage latency is lognormal:
        median * exp(sigma * Z) with Z standard normal, drawn independently per stage
        and per trial from np.random.default_rng(seed) in topo_order order, one
        rng.standard_normal(n) call per stage. A stage finishes at
        max(finish of its deps, default 0) + its latency. Return the finish time of the
        slowest stage for each trial (array of length n).

    percentile(samples, q) -> float
        q in [0, 100]. Use np.percentile with the default linear interpolation.

    allocate_budget(total_ms, weights, reserve_frac=0.1) -> dict[str, float]
        Hold back reserve_frac of total_ms as slack for network and jitter, then split the
        rest across stages in proportion to weights. Raise ValueError if any weight is
        not positive or reserve_frac is outside [0, 1).

HINTS
-----
- Process stages in topo_order so every dependency has a finish time already.
- A sum of p95s over-estimates the p95 of a sum for independent stages. The max of
  parallel stages does the opposite: the p95 of a max is above the max of the p95s.
- sigma = 0 gives deterministic latencies, so sampling must match critical_path exactly.

WHAT YOU SHOULD LEARN
-----------------------
- Why you budget by critical path, not by adding up every stage.
- Why per-stage p95 targets do not compose to an end-to-end p95, and how to check by
  simulation.
- How parallel fan-out makes the tail worse (waiting for the slowest of k calls).
"""
from __future__ import annotations

import numpy as np


def topo_order(deps: dict[str, list[str]]) -> list[str]:
    raise NotImplementedError("TODO")


def critical_path(latency: dict[str, float], deps: dict[str, list[str]]) -> tuple[float, list[str]]:
    raise NotImplementedError("TODO")


def sample_end_to_end(
    stages: dict[str, tuple[float, float]], deps: dict[str, list[str]], n: int, seed: int
) -> np.ndarray:
    raise NotImplementedError("TODO")


def percentile(samples: np.ndarray, q: float) -> float:
    raise NotImplementedError("TODO")


def allocate_budget(
    total_ms: float, weights: dict[str, float], reserve_frac: float = 0.1
) -> dict[str, float]:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # voice-style pipeline: asr -> (retrieve || guard) -> llm -> tts
    deps = {"retrieve": ["asr"], "guard": ["asr"], "llm": ["retrieve", "guard"], "tts": ["llm"]}
    lat = {"asr": 100.0, "retrieve": 80.0, "guard": 30.0, "llm": 300.0, "tts": 90.0}
    total, path = critical_path(lat, deps)
    assert total == 570.0 and path == ["asr", "retrieve", "llm", "tts"]
    # making guard the slow branch moves the critical path
    lat2 = dict(lat, guard=200.0)
    total2, path2 = critical_path(lat2, deps)
    assert total2 == 690.0 and path2 == ["asr", "guard", "llm", "tts"]
    # independent stages: the longest one is the answer
    assert critical_path({"a": 5.0, "b": 9.0}, {}) == (9.0, ["b"])
    # topo order is valid and deterministic
    order = topo_order(deps)
    assert order == ["asr", "guard", "retrieve", "llm", "tts"]
    for n, ds in deps.items():
        assert all(order.index(d) < order.index(n) for d in ds)
    # errors
    for bad_deps in ({"a": ["b"], "b": ["a"]},):
        try:
            topo_order(bad_deps)
        except ValueError:
            pass
        else:
            raise AssertionError("expected cycle error")
    try:
        critical_path({"a": 1.0}, {"a": ["zzz"]})
    except ValueError:
        pass
    else:
        raise AssertionError("expected unknown stage error")

    # deterministic stages (sigma 0) must reproduce the critical path exactly
    det = {k: (v, 0.0) for k, v in lat.items()}
    s = sample_end_to_end(det, deps, 50, seed=1)
    assert s.shape == (50,) and np.allclose(s, 570.0)

    # sums: independent lognormal stages in series
    chain = {"a": (100.0, 0.5), "b": (100.0, 0.5), "c": (100.0, 0.5), "d": (100.0, 0.5)}
    cdeps = {"b": ["a"], "c": ["b"], "d": ["c"]}
    samp = sample_end_to_end(chain, cdeps, 40_000, seed=11)
    p95_total = percentile(samp, 95)
    single = 100.0 * np.exp(0.5 * 1.6448536269514722)  # analytic p95 of one stage
    assert abs(single - 227.5) < 0.5
    # p95 of the sum is well below the sum of the per-stage p95s
    assert p95_total < 0.8 * 4 * single
    # but above the sum of the medians
    assert p95_total > 400.0
    # same seed reproduces, different seed differs
    assert np.array_equal(samp, sample_end_to_end(chain, cdeps, 40_000, seed=11))
    assert not np.array_equal(samp, sample_end_to_end(chain, cdeps, 40_000, seed=12))

    # fan-out: waiting for the slowest of k parallel calls makes the tail worse
    def fan(k: int) -> float:
        st = {f"w{i}": (100.0, 0.5) for i in range(k)}
        st["join"] = (1.0, 0.0)
        d = {"join": [f"w{i}" for i in range(k)]}
        return percentile(sample_end_to_end(st, d, 40_000, seed=5), 95)

    p1, p4, p16 = fan(1), fan(4), fan(16)
    assert p1 < p4 < p16
    assert p4 > single  # p95 of a max of 4 exceeds the p95 of one
    assert p16 > 1.3 * p1

    # percentile basics
    arr = np.arange(101, dtype=float)
    assert percentile(arr, 50) == 50.0 and percentile(arr, 95) == 95.0 and percentile(arr, 100) == 100.0

    # budget allocation
    b = allocate_budget(1000.0, {"retrieve": 2, "llm": 6, "guard": 1, "post": 1}, reserve_frac=0.1)
    assert abs(sum(b.values()) - 900.0) < 1e-9
    assert abs(b["llm"] - 540.0) < 1e-9 and abs(b["retrieve"] - 180.0) < 1e-9
    assert abs(sum(allocate_budget(500.0, {"a": 1, "b": 1}, 0.0).values()) - 500.0) < 1e-9
    for args in ((1000.0, {"a": 0}, 0.1), (1000.0, {"a": 1}, 1.0), (1000.0, {"a": 1}, -0.1)):
        try:
            allocate_budget(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
