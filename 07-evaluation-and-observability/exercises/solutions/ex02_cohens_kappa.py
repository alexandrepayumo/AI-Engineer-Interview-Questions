"""Exercise 02 - Judge-human agreement: percent agreement, Cohen's kappa, weighted kappa.

TASK
----
Implement these functions. `a` and `b` are equal-length lists of labels from two raters
(for example a human and an LLM judge). Labels are hashable (strings or ints).

    percent_agreement(a, b) -> float
        Fraction of items where the labels match.
    confusion_matrix(a, b, labels) -> np.ndarray
        Square int matrix M with M[i, j] = number of items where a == labels[i] and
        b == labels[j].
    cohens_kappa(a, b) -> float
        (po - pe) / (1 - pe), where po is observed agreement and pe is the chance
        agreement implied by each rater's marginal label frequencies:
        pe = sum over labels of P_a(label) * P_b(label).
        If pe == 1 (both raters always give one single label) return 1.0.
    weighted_kappa(a, b, labels, weights="linear") -> float
        For ordinal labels given in order. kappa_w = 1 - sum(W * O) / sum(W * E),
        with O the observed proportions (confusion / n), E the expected proportions
        (outer product of the row and column marginals), and the disagreement weight
        W[i, j] = |i - j| / (k - 1) for "linear" or (|i - j| / (k - 1)) ** 2 for
        "quadratic".
    Raise ValueError if the lists have different lengths or are empty.

HINTS
-----
- Build the confusion matrix first; both kappas follow from it.
- Marginals: M.sum(axis=1) / n for rater a, M.sum(axis=0) / n for rater b.

WHAT YOU SHOULD LEARN
---------------------
- 90% agreement can coexist with a kappa near zero or negative when one label dominates
  (the prevalence effect), so report kappa and the confusion matrix, not only agreement.
- Weighted kappa gives partial credit to near-misses on ordinal scales, which is what
  you want when comparing 1-5 judge scores with human scores.
"""
from __future__ import annotations

import numpy as np


def percent_agreement(a: list, b: list) -> float:
    _check(a, b)
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _check(a: list, b: list) -> None:
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("lists must be non-empty and of equal length")


def confusion_matrix(a: list, b: list, labels: list) -> np.ndarray:
    _check(a, b)
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(a, b):
        m[index[x], index[y]] += 1
    return m


def cohens_kappa(a: list, b: list) -> float:
    _check(a, b)
    labels = sorted(set(a) | set(b), key=str)
    m = confusion_matrix(a, b, labels)
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) / n) @ (m.sum(axis=0) / n))  # chance agreement from marginals
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def weighted_kappa(a: list, b: list, labels: list, weights: str = "linear") -> float:
    _check(a, b)
    m = confusion_matrix(a, b, labels)
    n = m.sum()
    k = len(labels)
    obs = m / n
    exp = np.outer(m.sum(axis=1) / n, m.sum(axis=0) / n)
    dist = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    w = dist if weights == "linear" else dist ** 2  # disagreement weights
    denom = (w * exp).sum()
    if denom == 0:
        return 1.0
    return float(1 - (w * obs).sum() / denom)


def _run_tests() -> None:
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 (rater a first)
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert abs(percent_agreement(a, b) - 0.7) < 1e-12
    cm = confusion_matrix(a, b, ["y", "n"])
    assert cm.tolist() == [[20, 5], [10, 15]]
    # pe = 0.5 * 0.6 + 0.5 * 0.4 = 0.5, so kappa = (0.7 - 0.5) / 0.5 = 0.4
    assert abs(cohens_kappa(a, b) - 0.4) < 1e-12

    # perfect agreement with two labels -> 1; total disagreement on balanced labels -> -1
    assert cohens_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == 1.0
    assert abs(cohens_kappa(["y", "n", "y", "n"], ["n", "y", "n", "y"]) + 1.0) < 1e-12
    # both raters use one label only
    assert cohens_kappa(["y"] * 5, ["y"] * 5) == 1.0

    # prevalence effect: 90% agreement, kappa slightly negative
    a = ["y"] * 95 + ["n"] * 5
    b = ["y"] * 90 + ["n"] * 5 + ["y"] * 5
    assert abs(percent_agreement(a, b) - 0.90) < 1e-12
    k = cohens_kappa(a, b)
    assert abs(k - (0.90 - 0.905) / (1 - 0.905)) < 1e-12 and k < 0

    # symmetric in the two raters
    x = ["a", "b", "c", "a", "b", "c", "a"]
    y = ["a", "b", "b", "a", "c", "c", "b"]
    assert abs(cohens_kappa(x, y) - cohens_kappa(y, x)) < 1e-12

    # weighted kappa on a 1-3 scale: near-misses cost less than far misses
    human = [1, 1, 2, 2, 3, 3, 1, 2, 3, 2]
    near = [1, 2, 2, 3, 3, 2, 1, 2, 3, 2]   # all 3 errors are off by one
    far = [1, 3, 2, 1, 3, 1, 1, 2, 3, 2]    # 3 errors, two of them off by two
    assert percent_agreement(human, near) == percent_agreement(human, far) == 0.7
    kn = weighted_kappa(human, near, [1, 2, 3])
    kf = weighted_kappa(human, far, [1, 2, 3])
    assert kn > kf
    assert weighted_kappa(human, human, [1, 2, 3]) == 1.0
    # quadratic weights punish the far misses harder than linear weights
    assert weighted_kappa(human, far, [1, 2, 3], "quadratic") < kf
    # with two labels, linear weighted kappa equals unweighted kappa
    a2 = ["y"] * 25 + ["n"] * 25
    b2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert abs(weighted_kappa(a2, b2, ["y", "n"]) - cohens_kappa(a2, b2)) < 1e-12

    for bad in (lambda: percent_agreement([1], [1, 2]), lambda: cohens_kappa([], [])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
