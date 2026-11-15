"""Exercise 04 - Sample size and power for comparing two accuracies.

TASK
----
Use `statistics.NormalDist` for the normal quantile and CDF (z = NormalDist().inv_cdf(q)).
Implement, with two-sided tests:

    n_per_arm_unpaired(p1, p2, alpha=0.05, power=0.8) -> int
        Examples per arm to detect p1 vs p2 with independent samples:
        n = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1*(1-p1) + p2*(1-p2)))**2
            / (p1 - p2)**2
        where pbar = (p1 + p2) / 2, z_a = z(1 - alpha/2), z_b = z(power). Round up.
    n_paired(p_discordant, delta, alpha=0.05, power=0.8) -> int
        Examples needed when both systems run on the same examples. p_discordant is the
        fraction of examples where exactly one system is right, delta the true accuracy
        difference (delta < p_discordant):
        n = (z_a * sqrt(pd) + z_b * sqrt(pd - delta**2))**2 / delta**2. Round up.
    power_unpaired(p1, p2, n, alpha=0.05) -> float
        Power of the unpaired test with n examples per arm:
        Phi((|p1-p2| * sqrt(n) - z_a * sqrt(2*pbar*(1-pbar))) / sqrt(p1*(1-p1) + p2*(1-p2)))
    min_detectable_effect(p, n, alpha=0.05, power=0.8) -> float
        Smallest delta > 0 such that power_unpaired(p, p + delta, n, alpha) >= power
        (cap p + delta below 1). Use bisection on delta in (0, 1 - p); 60 iterations.

HINTS
-----
- Both sample-size formulas grow like 1 / delta**2: halving the effect quadruples n.
- Power is increasing in delta, so bisection works for the minimum detectable effect.

WHAT YOU SHOULD LEARN
---------------------
- Detecting a 3-point gain near 80% accuracy needs about 2,600 examples per arm unpaired.
- Pairing the same examples across systems can cut that by roughly 3x when the systems
  disagree on only ~10% of examples.
- A 100-example eval can only resolve gaps of ~15 points; "small win" claims need a power
  calculation first.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()


def n_per_arm_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    za = _N.inv_cdf(1 - alpha / 2)
    zb = _N.inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num ** 2 / (p1 - p2) ** 2)


def n_paired(p_discordant: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    za = _N.inv_cdf(1 - alpha / 2)
    zb = _N.inv_cdf(power)
    # variance of the per-example difference is pd - delta^2
    num = za * math.sqrt(p_discordant) + zb * math.sqrt(p_discordant - delta ** 2)
    return math.ceil(num ** 2 / delta ** 2)


def power_unpaired(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    za = _N.inv_cdf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    z = (abs(p1 - p2) * math.sqrt(n) - za * math.sqrt(2 * pbar * (1 - pbar))) / math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return _N.cdf(z)


def min_detectable_effect(p: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    lo, hi = 0.0, (1 - p) * (1 - 1e-9)
    for _ in range(60):  # bisection: power increases with delta
        mid = (lo + hi) / 2
        if power_unpaired(p, p + mid, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def _run_tests() -> None:
    # 80% vs 83%: about 2,629 per arm
    n = n_per_arm_unpaired(0.80, 0.83)
    assert 2600 <= n <= 2660, n
    # symmetric in the two rates
    assert n_per_arm_unpaired(0.83, 0.80) == n
    # 80% vs 87%: about 7 points needs far fewer, roughly 1/6
    n7 = n_per_arm_unpaired(0.80, 0.87)
    assert 420 <= n7 <= 470, n7
    # halving the effect roughly quadruples n
    ratio = n_per_arm_unpaired(0.80, 0.82) / n_per_arm_unpaired(0.80, 0.84)
    assert 3.6 < ratio < 4.4, ratio
    # tighter alpha or higher power costs more
    assert n_per_arm_unpaired(0.8, 0.83, alpha=0.01) > n
    assert n_per_arm_unpaired(0.8, 0.83, power=0.9) > n

    # paired: 10% discordant, true gain 3 points -> about 870 examples in total
    npair = n_paired(0.10, 0.03)
    assert 860 <= npair <= 880, npair
    assert npair < n / 2.5  # a large saving over the unpaired design
    # fewer disagreements -> fewer examples needed
    assert n_paired(0.05, 0.03) < npair

    # power at the computed n is close to the target and rises with n
    pw = power_unpaired(0.80, 0.83, n)
    assert 0.795 < pw < 0.81, pw
    assert power_unpaired(0.80, 0.83, 500) < pw < power_unpaired(0.80, 0.83, 6000)
    # no effect: power equals roughly alpha/2 on the wrong side or less; stays below 0.1
    assert power_unpaired(0.80, 0.8001, 100) < 0.1

    # minimum detectable effect on a 100-example eval near 80% is large
    mde = min_detectable_effect(0.80, 100)
    assert 0.13 < mde < 0.18, mde
    assert abs(power_unpaired(0.80, 0.80 + mde, 100) - 0.8) < 1e-3
    # 10x the data shrinks it by about sqrt(10)
    mde1000 = min_detectable_effect(0.80, 1000)
    assert 2.8 < mde / mde1000 < 3.5


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
