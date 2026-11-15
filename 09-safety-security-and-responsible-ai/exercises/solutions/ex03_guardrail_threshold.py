"""Exercise 03 - Guardrail threshold selection: target false-positive rate and expected cost.

TASK
----
A guardrail classifier returns a score in [0, 1]; the request is blocked when
score >= threshold. Labels: 1 = harmful (should be blocked), 0 = benign.

    confusion(scores, labels, t) -> tuple[int, int, int, int]
        (tp, fp, tn, fn) at threshold t.

    rates(tp, fp, tn, fn) -> dict[str, float]
        keys "precision", "recall", "fpr". A zero denominator gives 0.0 for precision
        and recall, and 0.0 for fpr.

    candidate_thresholds(scores) -> list[float]
        Sorted unique scores plus math.nextafter(max(scores), math.inf) (a threshold that
        blocks nothing).

    threshold_for_fpr(scores, labels, target_fpr) -> float
        The SMALLEST candidate threshold whose false-positive rate on the benign examples
        is <= target_fpr. (Smallest threshold = most blocking = highest recall under the
        constraint.) Raises ValueError if there are no benign examples.

    best_threshold_by_cost(scores, labels, c_fp, c_fn) -> tuple[float, float]
        (threshold, total_cost) minimising fp * c_fp + fn * c_fn over the candidates.
        Ties go to the HIGHEST threshold (fewest blocks).

    bayes_threshold(c_fp, c_fn) -> float
        For a calibrated probability p of harm, block when p >= c_fp / (c_fp + c_fn).

    precision_at_prevalence(tpr, fpr, prevalence) -> float
        P(harmful | blocked) = tpr * prev / (tpr * prev + fpr * (1 - prev)). Return 0.0
        when the denominator is 0.

HINTS
-----
- Sort once if you like, but an O(n * candidates) loop is fine at these sizes.
- Compare floats with the same arithmetic you used to build the candidates.

WHAT YOU SHOULD LEARN
---------------------
- You pick a threshold from an operating requirement ("at most 1% of benign users blocked"
  or "a miss costs 20x a false alarm"), not from 0.5.
- Cost asymmetry moves the optimal threshold, and the Bayes rule gives the calibrated
  answer directly.
- Base rates dominate: a classifier with 95% recall and 1% FPR has under 9% precision when
  only 1 in 1,000 requests is harmful. This is why guardrail alerts need a second stage.
"""
from __future__ import annotations

import math


def confusion(scores: list[float], labels: list[int], t: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        blocked = s >= t
        if blocked and y == 1:
            tp += 1
        elif blocked and y == 0:
            fp += 1
        elif not blocked and y == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def rates(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "fpr": fpr}


def candidate_thresholds(scores: list[float]) -> list[float]:
    return sorted(set(scores)) + [math.nextafter(max(scores), math.inf)]


def threshold_for_fpr(scores: list[float], labels: list[int], target_fpr: float) -> float:
    if not any(y == 0 for y in labels):
        raise ValueError("need at least one benign example")
    # FPR only falls as t rises, so the first (smallest) feasible candidate is the answer.
    for t in candidate_thresholds(scores):
        _, fp, tn, _ = confusion(scores, labels, t)
        if fp / (fp + tn) <= target_fpr:
            return t
    raise AssertionError("unreachable: the last candidate blocks nothing")


def best_threshold_by_cost(scores: list[float], labels: list[int],
                           c_fp: float, c_fn: float) -> tuple[float, float]:
    best_t, best_cost = 0.0, math.inf
    for t in candidate_thresholds(scores):  # ascending, so <= keeps the highest tie
        _, fp, _, fn = confusion(scores, labels, t)
        cost = fp * c_fp + fn * c_fn
        if cost <= best_cost:
            best_t, best_cost = t, cost
    return best_t, best_cost


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    return c_fp / (c_fp + c_fn)


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    num = tpr * prevalence
    den = num + fpr * (1.0 - prevalence)
    return num / den if den else 0.0


def _run_tests() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 1, 0, 0]
    assert confusion(scores, labels, 0.6) == (3, 1, 3, 1)
    assert confusion(scores, labels, 0.3) == (4, 2, 2, 0)
    assert confusion(scores, labels, 2.0) == (0, 0, 4, 4)
    r = rates(3, 1, 3, 1)
    assert r == {"precision": 0.75, "recall": 0.75, "fpr": 0.25}
    assert rates(0, 0, 5, 0) == {"precision": 0.0, "recall": 0.0, "fpr": 0.0}

    cands = candidate_thresholds(scores)
    assert cands[:3] == [0.1, 0.2, 0.3] and cands[-2] == 0.9 and cands[-1] > 0.9
    assert len(cands) == 9
    assert confusion(scores, labels, cands[-1])[:2] == (0, 0)

    # cost asymmetry moves the optimum (hand-computed in the deep dive)
    assert best_threshold_by_cost(scores, labels, 1, 5) == (0.3, 2)
    assert best_threshold_by_cost(scores, labels, 10, 1) == (0.8, 2)
    t_eq, c_eq = best_threshold_by_cost(scores, labels, 1, 1)
    assert c_eq == 2 and t_eq == 0.8  # 0.3, 0.6 and 0.8 tie, highest wins
    assert abs(bayes_threshold(1, 5) - 1 / 6) < 1e-12
    assert abs(bayes_threshold(10, 1) - 10 / 11) < 1e-12
    assert bayes_threshold(1, 1) == 0.5

    # FPR targeting on a synthetic set: 100 benign scores 0.00..0.99, 100 harmful 0.50..0.995
    neg = [i / 100 for i in range(100)]
    pos = [0.5 + i * 0.005 for i in range(100)]
    sc = neg + pos
    lb = [0] * 100 + [1] * 100
    for target in (0.0, 0.01, 0.05, 0.2, 1.0):
        t = threshold_for_fpr(sc, lb, target)
        _, fp, tn, _ = confusion(sc, lb, t)
        assert fp / (fp + tn) <= target + 1e-12, (target, t)
        # minimality: the next smaller candidate must violate the constraint
        cs = candidate_thresholds(sc)
        i = cs.index(t)
        if i > 0:
            _, fp2, tn2, _ = confusion(sc, lb, cs[i - 1])
            assert fp2 / (fp2 + tn2) > target + 1e-12, (target, t)
    t05 = threshold_for_fpr(sc, lb, 0.05)
    assert 0.94 < t05 <= 0.95
    assert confusion(sc, lb, t05)[1] == 5
    assert threshold_for_fpr(sc, lb, 0.0) > 0.99  # blocks nothing benign
    assert threshold_for_fpr(sc, lb, 1.0) == min(sc)
    # recall under the constraint falls as the constraint tightens
    recalls = [rates(*confusion(sc, lb, threshold_for_fpr(sc, lb, f)))["recall"]
               for f in (0.0, 0.01, 0.05, 0.2)]
    assert recalls == sorted(recalls) and recalls[0] < recalls[-1]
    try:
        threshold_for_fpr([0.5], [1], 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    # base rates: 95% recall, 1% FPR, 1 harmful per 1,000
    p = precision_at_prevalence(0.95, 0.01, 0.001)
    assert abs(p - 0.00095 / (0.00095 + 0.00999)) < 1e-12 and round(p, 4) == 0.0868
    assert precision_at_prevalence(0.95, 0.01, 0.5) > 0.98
    assert precision_at_prevalence(0.0, 0.0, 0.1) == 0.0
    # tightening FPR 10x buys back precision
    assert precision_at_prevalence(0.95, 0.001, 0.001) > 0.48


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
