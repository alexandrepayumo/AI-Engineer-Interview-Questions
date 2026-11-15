"""Exercise 07 - pass@k, pass^k and confidence intervals from trial results.

TASK
----
Given n independent trials of an agent on a task, c of which succeeded, implement:

    pass_at_k(n, c, k) -> float
        Unbiased estimate that at least one of k trials drawn without replacement from the
        n observed trials succeeds: 1 - C(n-c, k) / C(n, k). Return 1.0 when n - c < k.
    pass_hat_k(n, c, k) -> float
        Unbiased estimate that all k drawn trials succeed (the tau-bench "pass^k"):
        C(c, k) / C(n, k). Returns 0.0 when c < k.
    Both raise ValueError unless 1 <= k <= n and 0 <= c <= n.

    iid_pass_at_k(p, k) -> float     1 - (1 - p) ** k   (known per-trial success rate p)
    iid_pass_hat_k(p, k) -> float    p ** k

    aggregate(results, k) -> dict
        results maps task id -> list of booleans (one per trial; lengths may differ but
        each must be >= k, otherwise ValueError). Return the mean over tasks of
        {"pass@1": pass_at_k(n, c, 1), "pass@k": ..., "pass^k": ...}.

    wilson_interval(successes, n, z=1.96) -> tuple[float, float]
        Wilson score interval for a proportion. n == 0 returns (0.0, 1.0).
        centre = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)

HINTS
-----
- math.comb gives exact binomial coefficients, so there is no overflow or float
  cancellation for the sizes used here.
- pass^k <= pass@1 <= pass@k always. A big gap between pass@1 and pass^k means the agent
  is capable but inconsistent.
- The naive plug-in p**k from a single p-hat is biased for small n. The combinatorial form
  uses the actual trials.

WHAT YOU SHOULD LEARN
---------------------
- pass@k measures capability (any success), pass^k measures reliability (every success).
  Customer-facing agents live and die by the second.
- How fast consistency decays: p = 0.9 gives pass^8 of only 0.43.
- Why you need many trials per task and a confidence interval before comparing two agents.
"""
from __future__ import annotations

import math


def _check(n: int, c: int, k: int) -> None:
    if not (1 <= k <= n and 0 <= c <= n):
        raise ValueError(f"invalid arguments n={n} c={c} k={k}")


def pass_at_k(n: int, c: int, k: int) -> float:
    _check(n, c, k)
    if n - c < k:  # fewer than k failures exist, so any k draws contain a success
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    _check(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)  # all k draws come from the c successes


def iid_pass_at_k(p: float, k: int) -> float:
    return 1.0 - (1.0 - p) ** k


def iid_pass_hat_k(p: float, k: int) -> float:
    return p ** k


def aggregate(results: dict[str, list[bool]], k: int) -> dict[str, float]:
    one, at_k, hat_k = [], [], []
    for task, trials in results.items():
        n, c = len(trials), sum(1 for t in trials if t)
        if n < k:
            raise ValueError(f"task {task!r} has {n} trials, need at least {k}")
        one.append(pass_at_k(n, c, 1))
        at_k.append(pass_at_k(n, c, k))
        hat_k.append(pass_hat_k(n, c, k))
    m = len(results)
    return {"pass@1": sum(one) / m, "pass@k": sum(at_k) / m, "pass^k": sum(hat_k) / m}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _run_tests() -> None:
    # hand-computed values: n=10, c=3
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(pass_at_k(10, 3, 5) - (1 - 21 / 252)) < 1e-12  # C(7,5)/C(10,5)
    assert abs(pass_hat_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(pass_hat_k(10, 3, 2) - 3 / 45) < 1e-12        # C(3,2)/C(10,2)
    assert pass_hat_k(10, 3, 4) == 0.0                       # only 3 successes exist
    assert pass_at_k(10, 3, 8) == 1.0                        # only 7 failures exist

    # extremes
    assert pass_at_k(5, 0, 3) == 0.0 and pass_hat_k(5, 0, 3) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0 and pass_hat_k(5, 5, 3) == 1.0
    assert pass_hat_k(5, 4, 5) == 0.0 and pass_at_k(5, 1, 5) == 1.0

    # ordering property for every (n, c, k)
    for n in range(1, 9):
        for c in range(n + 1):
            for k in range(1, n + 1):
                lo, mid, hi = pass_hat_k(n, c, k), pass_at_k(n, c, 1), pass_at_k(n, c, k)
                assert lo <= mid + 1e-12 and mid <= hi + 1e-12, (n, c, k)
                assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    # monotone in k: pass@k rises, pass^k falls
    ats = [pass_at_k(20, 12, k) for k in range(1, 21)]
    hats = [pass_hat_k(20, 12, k) for k in range(1, 21)]
    assert all(a <= b + 1e-12 for a, b in zip(ats, ats[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(hats, hats[1:]))

    # invalid inputs
    for bad in [(5, 2, 0), (5, 2, 6), (5, 6, 1), (5, -1, 1)]:
        try:
            pass_at_k(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # the i.i.d. model: how fast consistency decays
    assert abs(iid_pass_hat_k(0.9, 8) - 0.9 ** 8) < 1e-12
    assert round(iid_pass_hat_k(0.9, 8), 2) == 0.43
    assert abs(iid_pass_at_k(0.8, 8) - (1 - 0.2 ** 8)) < 1e-12
    assert round(iid_pass_hat_k(0.95, 20), 3) == 0.358

    # for large n the combinatorial estimator approaches the i.i.d. formula
    n, c = 2000, 1800
    assert abs(pass_hat_k(n, c, 4) - 0.9 ** 4) < 2e-3

    # aggregation over tasks: A always succeeds, B is a coin flip, C never succeeds
    res = {
        "A": [True, True, True, True],
        "B": [True, False, True, False],
        "C": [False, False, False, False],
    }
    agg = aggregate(res, 2)
    assert abs(agg["pass@1"] - 0.5) < 1e-12
    assert abs(agg["pass@k"] - (1 + 5 / 6 + 0) / 3) < 1e-12
    assert abs(agg["pass^k"] - (1 + 1 / 6 + 0) / 3) < 1e-12
    # unequal trial counts are fine as long as each has >= k
    agg2 = aggregate({"A": [True, True, True], "B": [True, False]}, 2)
    assert abs(agg2["pass^k"] - (1.0 + 0.0) / 2) < 1e-12
    try:
        aggregate({"A": [True]}, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for too few trials")

    # Wilson interval: 8 of 10 at 95%
    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 or abs(lo0) < 1e-12
    assert 0.1 < hi0 < 0.2  # zero failures observed does not mean zero risk
    # interval narrows as n grows at the same rate
    w_small = wilson_interval(8, 10)
    w_big = wilson_interval(80, 100)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])
    # two agents at 0.70 vs 0.75 on 40 tasks cannot be told apart
    a, b = wilson_interval(28, 40), wilson_interval(30, 40)
    assert a[1] > b[0]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
