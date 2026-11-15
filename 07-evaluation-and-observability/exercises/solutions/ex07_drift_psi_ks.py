"""Exercise 07 - Drift detection on score distributions: PSI and the KS test.

TASK
----
Implement:

    psi(expected, actual, n_bins=10, eps=1e-4) -> float
        Population Stability Index of `actual` against the reference sample `expected`.
        Bin edges are the interior quantiles of `expected` at i / n_bins (i = 1..n_bins-1),
        with the outer bins open-ended (-inf, +inf). Compute the proportion of each sample
        in each bin (np.searchsorted(edges, x, side="right") gives the bin index), clip
        proportions below at `eps`, then
        PSI = sum((a_i - e_i) * ln(a_i / e_i)).
    psi_from_proportions(e, a, eps=1e-4) -> float
        The same formula for already-binned proportions (lists of equal length).
    psi_status(value) -> str
        "stable" if value < 0.1, "moderate" if value < 0.25, else "major".
    ks_statistic(x, y) -> float
        Two-sample Kolmogorov-Smirnov statistic: the maximum absolute difference between
        the two empirical CDFs, evaluated at every observed value (use
        np.searchsorted(sorted_sample, points, side="right") / n).
    ks_pvalue(d, n, m) -> float
        Asymptotic two-sided p-value. With ne = n * m / (n + m) and
        lam = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * d:
        p = 2 * sum_{j=1..100} (-1)**(j-1) * exp(-2 * j**2 * lam**2), clipped to [0, 1].
        Return 1.0 if d == 0.

HINTS
-----
- Use np.quantile(expected, np.arange(1, n_bins) / n_bins) for the edges.
- Ties in the reference can create duplicate edges; that is fine for searchsorted.

WHAT YOU SHOULD LEARN
---------------------
- PSI compares binned distributions and gives rule-of-thumb thresholds; KS is
  distribution-free and gives a p-value, but with large samples it flags trivial shifts.
- Both detect a shift in judge scores, token counts or latencies without any labels.
- Binning from the reference sample keeps the comparison anchored to "normal".
"""
from __future__ import annotations

import math

import numpy as np


def psi_from_proportions(e: list[float], a: list[float], eps: float = 1e-4) -> float:
    e_arr = np.clip(np.asarray(e, dtype=float), eps, None)
    a_arr = np.clip(np.asarray(a, dtype=float), eps, None)
    return float(np.sum((a_arr - e_arr) * np.log(a_arr / e_arr)))


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.arange(1, n_bins) / n_bins)  # reference quantiles

    def proportions(x: np.ndarray) -> np.ndarray:
        bins = np.searchsorted(edges, x, side="right")
        return np.bincount(bins, minlength=n_bins) / len(x)

    return psi_from_proportions(proportions(expected), proportions(actual), eps)


def psi_status(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "major"


def ks_statistic(x, y) -> float:
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.sort(np.asarray(y, dtype=float))
    points = np.concatenate([xs, ys])
    cdf_x = np.searchsorted(xs, points, side="right") / len(xs)
    cdf_y = np.searchsorted(ys, points, side="right") / len(ys)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def ks_pvalue(d: float, n: int, m: int) -> float:
    if d == 0:
        return 1.0
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(min(1.0, max(0.0, p)))


def _run_tests() -> None:
    # hand-computed PSI: terms 0.2*ln(0.6) magnitude and 0.2*ln(2)
    e = [0.5, 0.3, 0.2]
    a = [0.3, 0.3, 0.4]
    expected = (0.3 - 0.5) * math.log(0.3 / 0.5) + 0.0 + (0.4 - 0.2) * math.log(0.4 / 0.2)
    assert abs(psi_from_proportions(e, a) - expected) < 1e-12
    assert abs(psi_from_proportions(e, a) - 0.2408) < 1e-4
    assert psi_from_proportions(e, e) == 0.0
    # symmetric in its two arguments
    assert abs(psi_from_proportions(e, a) - psi_from_proportions(a, e)) < 1e-12
    # an empty bin does not produce infinity thanks to eps
    assert math.isfinite(psi_from_proportions([0.5, 0.5], [1.0, 0.0]))

    assert psi_status(0.05) == "stable"
    assert psi_status(0.1) == "moderate"
    assert psi_status(0.2408) == "moderate"
    assert psi_status(0.25) == "major"

    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 5000)
    same = rng.normal(0.0, 1.0, 5000)
    shifted = rng.normal(0.5, 1.0, 5000)
    wide = rng.normal(0.0, 2.0, 5000)
    p_same = psi(ref, same)
    p_shift = psi(ref, shifted)
    p_wide = psi(ref, wide)
    assert p_same < 0.02
    assert psi_status(p_same) == "stable"
    assert 0.1 < p_shift < 0.35, p_shift
    assert p_wide > p_shift
    assert psi(ref, ref) == 0.0
    # values outside the reference range land in the open-ended outer bins
    assert math.isfinite(psi(ref, ref + 100.0))
    assert psi(ref, ref + 100.0) > 1.0

    # KS by hand: F_x - F_y is largest (2/3) at 2 and at 3
    assert abs(ks_statistic([1, 2, 3], [2.5, 3.5, 4.5]) - 2 / 3) < 1e-12
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    # symmetric, and handles different sample sizes and ties
    assert abs(ks_statistic([1, 1, 2, 5], [1, 3]) - ks_statistic([1, 3], [1, 1, 2, 5])) < 1e-12
    assert abs(ks_statistic([1, 1, 2, 5], [1, 3]) - 0.25) < 1e-12

    # p-values: identical samples give 1; a clear shift gives a tiny p; no shift is large
    assert ks_pvalue(0.0, 100, 100) == 1.0
    d_same = ks_statistic(ref, same)
    d_shift = ks_statistic(ref, shifted)
    assert ks_pvalue(d_same, 5000, 5000) > 0.01
    assert ks_pvalue(d_shift, 5000, 5000) < 1e-10
    # known value: lam about 1.358 gives p about 0.05
    n = 10_000
    ne = n / 2
    d = 1.358 / (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne))
    assert abs(ks_pvalue(d, n, n) - 0.05) < 0.002
    # the same small shift is "significant" with enough data, a sign KS flags trivial drift
    assert ks_pvalue(0.02, 200, 200) > 0.5
    assert ks_pvalue(0.02, 100_000, 100_000) < 1e-6


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
