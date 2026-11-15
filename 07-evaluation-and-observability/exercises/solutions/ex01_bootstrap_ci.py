"""Exercise 01 - Bootstrap confidence intervals, unpaired and paired.

TASK
----
Implement these functions. `scores` / `a` / `b` are per-example scores (0/1 or floats)
as numpy arrays or lists. Use `rng = np.random.default_rng(seed)` and draw ONE index matrix
`rng.integers(0, n, size=(n_boot, n))` per call.

    bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=0) -> (mean, lo, hi)
        Percentile bootstrap CI of the mean: resample examples with replacement,
        take the alpha/2 and 1-alpha/2 percentiles of the resampled means.
    paired_bootstrap_diff(a, b, n_boot=2000, alpha=0.05, seed=0) -> (diff, lo, hi, p)
        a and b are scores of two systems on the SAME examples. Resample example
        indices once per replicate and use the same indices for both systems.
        diff = mean(b) - mean(a). p is a two-sided bootstrap p-value:
        min(1, 2 * min(P(d* <= 0), P(d* >= 0))) over the resampled differences d*.
        Raise ValueError if the lengths differ.
    unpaired_bootstrap_diff(a, b, n_boot=2000, alpha=0.05, seed=0) -> (diff, lo, hi, p)
        Same outputs, but resample a and b INDEPENDENTLY (ignores the pairing).

HINTS
-----
- x[idx] with idx of shape (n_boot, n) gives all resamples at once; mean over axis=1.
- For the paired version, resample the per-example difference d = b - a.
- np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)]).

WHAT YOU SHOULD LEARN
---------------------
- A bootstrap CI needs no formula and works for any statistic.
- Pairing removes example difficulty from the variance: the same 200 examples can give a
  significant paired result and a non-significant unpaired one.
- The p-value comes from the resampled distribution of the difference, not a t table.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(scores, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)  # one mean per resample
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def _p_value(diffs: np.ndarray) -> float:
    # Two-sided: how often does the resampled difference fall on each side of zero?
    return float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    d = b - a  # per-example difference carries all the signal
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi), _p_value(diffs)


def unpaired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)  # independent resamples
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(b.mean() - a.mean()), float(lo), float(hi), _p_value(diffs)


def _run_tests() -> None:
    # constant scores: no uncertainty
    m, lo, hi = bootstrap_ci([1.0] * 50)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)

    # 80 correct out of 100: SE = 0.04, so the 95% interval is about +-0.078
    x = np.array([1] * 80 + [0] * 20)
    m, lo, hi = bootstrap_ci(x)
    assert m == 0.8 and lo < 0.8 < hi
    assert 0.13 < hi - lo < 0.18
    # same seed, same answer; larger n shrinks the interval
    assert bootstrap_ci(x) == (m, lo, hi)
    big = np.array([1] * 800 + [0] * 200)
    _, lo2, hi2 = bootstrap_ci(big)
    assert (hi2 - lo2) < 0.5 * (hi - lo)

    # paired vs unpaired on the same data. 200 examples:
    # 20 examples only B gets right, 8 only A gets right, 100 both right, 72 both wrong.
    a = np.array([0] * 20 + [1] * 8 + [1] * 100 + [0] * 72)
    b = np.array([1] * 20 + [0] * 8 + [1] * 100 + [0] * 72)
    d, plo, phi, pp = paired_bootstrap_diff(a, b)
    assert abs(d - 12 / 200) < 1e-12
    assert plo > 0 and pp < 0.05  # paired: significant, interval about (0.01, 0.11)
    d2, ulo, uhi, up = unpaired_bootstrap_diff(a, b)
    assert abs(d2 - d) < 1e-12
    assert ulo < 0 < uhi and up > 0.05  # unpaired: not significant
    assert (uhi - ulo) > (phi - plo)  # pairing tightens the interval

    # identical systems: zero difference, p = 1
    d, lo, hi, p = paired_bootstrap_diff(a, a)
    assert (d, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)

    # length mismatch
    try:
        paired_bootstrap_diff([1, 0], [1, 0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
