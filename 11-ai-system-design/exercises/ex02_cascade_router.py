"""Exercise 02 - Model cascade simulator: quality versus cost for a confidence threshold.

TASK
----
A cascade sends every request to a cheap model first. If the cheap model's confidence is
at least `threshold`, its answer is accepted. Otherwise the request escalates to a large
model. Implement:

    simulate_cascade(conf, ok_small, ok_large, threshold, cost_small, cost_large,
                     pay_small_on_escalation=True) -> dict
        conf, ok_small, ok_large are equal-length numpy arrays (float, bool, bool).
        For each request: accepted if conf >= threshold.
          accepted  -> correct = ok_small,  cost = cost_small
          escalated -> correct = ok_large,  cost = cost_large
                       + cost_small if pay_small_on_escalation else 0
        Return {"quality": mean correct, "cost": mean cost per request,
                "escalation_rate": fraction escalated}.
        Raise ValueError for empty or unequal-length inputs.

    sweep(conf, ok_small, ok_large, thresholds, cost_small, cost_large) -> list[dict]
        One simulate_cascade result per threshold, each with a "threshold" key added.

    cheapest_threshold(rows, min_quality) -> float | None
        Among rows whose quality >= min_quality, return the threshold with the lowest
        cost (ties: the lower threshold). None if no row qualifies.

    expected_cost(escalation_rate, cost_small, cost_large) -> float
        Closed form when the small model is always called first.

    make_synthetic(n, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]
        Given below: a calibrated-ish confidence signal.

HINTS
-----
- A boolean mask `accepted = conf >= threshold` makes everything vectorised.
- Threshold 0 means "never escalate", threshold above max confidence means "always
  escalate", so the sweep spans the all-small and all-large endpoints.
- The cascade can only beat the large model on quality if the large model is not
  perfect on the items where the small one is confident and right.

WHAT YOU SHOULD LEARN
---------------------
- Quality and cost trade off along a curve you can compute offline from a labelled set.
- Paying for the small call on every escalation is the hidden cost of a cascade, unlike a
  router that decides before calling any model.
- Confidence quality (how well it separates right from wrong) decides how good the curve
  can get. A useless confidence signal gives a straight line between the endpoints.
"""
from __future__ import annotations

import numpy as np


def make_synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small model right 70% of the time, large model right 92%. Confidence is higher on
    average when the small model is right."""
    rng = np.random.default_rng(seed)
    ok_small = rng.random(n) < 0.70
    ok_large = rng.random(n) < 0.92
    conf = np.where(ok_small, rng.beta(6, 2, n), rng.beta(2, 4, n))
    return conf, ok_small, ok_large


def simulate_cascade(
    conf: np.ndarray,
    ok_small: np.ndarray,
    ok_large: np.ndarray,
    threshold: float,
    cost_small: float,
    cost_large: float,
    pay_small_on_escalation: bool = True,
) -> dict:
    raise NotImplementedError("TODO")


def sweep(
    conf: np.ndarray,
    ok_small: np.ndarray,
    ok_large: np.ndarray,
    thresholds: list[float],
    cost_small: float,
    cost_large: float,
) -> list[dict]:
    raise NotImplementedError("TODO")


def cheapest_threshold(rows: list[dict], min_quality: float) -> float | None:
    raise NotImplementedError("TODO")


def expected_cost(escalation_rate: float, cost_small: float, cost_large: float) -> float:
    # everyone pays the small call, the escalated fraction also pays the large call
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # hand-built example, 4 requests
    conf = np.array([0.9, 0.8, 0.4, 0.2])
    ok_s = np.array([True, False, True, False])
    ok_l = np.array([True, True, True, True])
    r = simulate_cascade(conf, ok_s, ok_l, 0.5, cost_small=1.0, cost_large=10.0)
    # accepted: first two (correct: T, F). escalated: last two (correct: T, T)
    assert abs(r["quality"] - 0.75) < 1e-12
    assert abs(r["escalation_rate"] - 0.5) < 1e-12
    assert abs(r["cost"] - (1 + 1 + 11 + 11) / 4) < 1e-12
    # a router that decides first does not pay for the small call on escalation
    r2 = simulate_cascade(conf, ok_s, ok_l, 0.5, 1.0, 10.0, pay_small_on_escalation=False)
    assert abs(r2["cost"] - (1 + 1 + 10 + 10) / 4) < 1e-12
    # the threshold is inclusive
    r3 = simulate_cascade(conf, ok_s, ok_l, 0.8, 1.0, 10.0)
    assert abs(r3["escalation_rate"] - 0.5) < 1e-12

    # endpoints on the synthetic set
    conf, ok_s, ok_l = make_synthetic(20_000, seed=7)
    all_small = simulate_cascade(conf, ok_s, ok_l, 0.0, 1.0, 20.0)
    all_large = simulate_cascade(conf, ok_s, ok_l, 1.01, 1.0, 20.0)
    assert all_small["escalation_rate"] == 0.0 and all_small["cost"] == 1.0
    assert abs(all_small["quality"] - ok_s.mean()) < 1e-12
    assert all_large["escalation_rate"] == 1.0 and all_large["cost"] == 21.0
    assert abs(all_large["quality"] - ok_l.mean()) < 1e-12
    assert 0.68 < all_small["quality"] < 0.72 and 0.90 < all_large["quality"] < 0.94

    # sweep: escalation and cost rise with the threshold
    ths = [round(0.05 * i, 2) for i in range(0, 21)]
    rows = sweep(conf, ok_s, ok_l, ths, 1.0, 20.0)
    assert [r["threshold"] for r in rows] == ths
    esc = [r["escalation_rate"] for r in rows]
    cost = [r["cost"] for r in rows]
    assert all(a <= b for a, b in zip(esc, esc[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(cost, cost[1:]))
    # the closed form matches the simulation for every row
    for r in rows:
        assert abs(r["cost"] - expected_cost(r["escalation_rate"], 1.0, 20.0)) < 1e-9

    # a useful confidence signal beats the straight line between the endpoints
    mid = next(r for r in rows if r["threshold"] == 0.5)
    frac = (mid["cost"] - all_small["cost"]) / (all_large["cost"] - all_small["cost"])
    line_quality = all_small["quality"] + frac * (all_large["quality"] - all_small["quality"])
    assert mid["quality"] > line_quality + 0.02

    # a useless confidence signal (random) lies on that line
    rng = np.random.default_rng(3)
    junk = rng.random(len(conf))
    jrow = simulate_cascade(junk, ok_s, ok_l, 0.5, 1.0, 20.0)
    jfrac = (jrow["cost"] - 1.0) / 20.0
    jline = ok_s.mean() + jfrac * (ok_l.mean() - ok_s.mean())
    assert abs(jrow["quality"] - jline) < 0.02

    # picking the cheapest threshold that meets a quality target
    target = 0.85
    t = cheapest_threshold(rows, target)
    assert t is not None
    chosen = next(r for r in rows if r["threshold"] == t)
    assert chosen["quality"] >= target
    assert all(r["cost"] >= chosen["cost"] - 1e-12 for r in rows if r["quality"] >= target)
    assert cheapest_threshold(rows, 0.999) is None
    # ties resolve to the lower threshold
    tie = [{"threshold": 0.3, "quality": 0.9, "cost": 2.0},
           {"threshold": 0.2, "quality": 0.9, "cost": 2.0}]
    assert cheapest_threshold(tie, 0.9) == 0.2

    # cascades cost less than always-large at the same target
    assert chosen["cost"] < all_large["cost"]

    # input validation
    for args in ((np.array([]), np.array([], dtype=bool), np.array([], dtype=bool)),
                 (np.array([0.5]), np.array([True, False]), np.array([True]))):
        try:
            simulate_cascade(*args, 0.5, 1.0, 2.0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
