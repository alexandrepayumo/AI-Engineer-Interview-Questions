"""Exercise 05 - Red-team harness statistics: attack success rate with Wilson intervals.

TASK
----
You ran attack attempts against your own system. Each result is (category, success).

    wilson_interval(successes, n, z=1.96) -> tuple[float, float]
        Wilson score interval. n == 0 returns (0.0, 1.0).
        centre = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
        Clamp the result to [0, 1].

    summarise(results) -> dict[str, dict[str, float]]
        results is a list of (category, success_bool). Return a dict keyed by category
        (sorted) plus the key "ALL" for the pooled data. Each value is
        {"n": n, "successes": s, "asr": s / n, "lo": lo, "hi": hi} using wilson_interval.

    worst_category(summary) -> str
        The category (excluding "ALL") with the highest UPPER bound "hi". Ties go to the
        alphabetically first. This is the conservative choice when n differs by category.

    two_proportion_z(s1, n1, s2, n2) -> float
        Pooled two-proportion z statistic for p1 - p2:
        p = (s1 + s2) / (n1 + n2), se = sqrt(p(1-p)(1/n1 + 1/n2)), z = (s1/n1 - s2/n2) / se.
        Return 0.0 when se == 0.

    zero_success_upper_bound(n, confidence=0.95) -> float
        If 0 of n attacks succeeded, the one-sided upper bound on the true rate is
        1 - (1 - confidence) ** (1 / n).

    attempts_needed(target_asr, confidence=0.95) -> int
        Smallest n such that n attempts with zero successes demonstrate an ASR below
        target_asr at that confidence: ceil(ln(1 - confidence) / ln(1 - target_asr)).

    asr_at_n(p, n) -> float
        Probability that at least one of n independent attempts succeeds: 1 - (1 - p) ** n.

HINTS
-----
- Use math.sqrt, math.log and math.ceil. No numpy needed.
- Group with a plain dict. Do not forget the pooled "ALL" entry.

WHAT YOU SHOULD LEARN
---------------------
- A point estimate of ASR from 20 attempts is nearly meaningless. 2/20 has a 95% interval of
  about 3% to 30%.
- "We found nothing" needs a number of attempts attached. Zero successes in 100 attempts still
  allows a true rate near 3%.
- Attackers get many tries (best-of-N), so a 1% per-attempt ASR is a ~63% chance of eventual
  success at 100 attempts.
"""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(results: list[tuple[str, bool]]) -> dict[str, dict[str, float]]:
    counts: dict[str, list[int]] = {}
    for cat, ok in results:
        for key in (cat, "ALL"):
            c = counts.setdefault(key, [0, 0])
            c[0] += 1 if ok else 0
            c[1] += 1
    out: dict[str, dict[str, float]] = {}
    for key in sorted(counts, key=lambda k: (k == "ALL", k)):
        s, n = counts[key]
        lo, hi = wilson_interval(s, n)
        out[key] = {"n": n, "successes": s, "asr": s / n, "lo": lo, "hi": hi}
    return out


def worst_category(summary: dict[str, dict[str, float]]) -> str:
    cats = sorted(k for k in summary if k != "ALL")
    best = cats[0]
    for k in cats[1:]:
        if summary[k]["hi"] > summary[best]["hi"]:  # strict, so ties keep the earlier name
            best = k
    return best


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> float:
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (s1 / n1 - s2 / n2) / se


def zero_success_upper_bound(n: int, confidence: float = 0.95) -> float:
    return 1 - (1 - confidence) ** (1 / n)


def attempts_needed(target_asr: float, confidence: float = 0.95) -> int:
    return math.ceil(math.log(1 - confidence) / math.log(1 - target_asr))


def asr_at_n(p: float, n: int) -> float:
    return 1 - (1 - p) ** n


def _run_tests() -> None:
    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(2, 20)
    assert abs(lo - 0.0279) < 2e-3 and abs(hi - 0.3010) < 2e-3
    lo, hi = wilson_interval(0, 50)
    assert lo < 1e-12 and 0.07 < hi < 0.075
    lo, hi = wilson_interval(50, 50)
    assert hi > 1 - 1e-9 and lo > 0.92

    results = (
        [("direct_injection", True)] * 2 + [("direct_injection", False)] * 18
        + [("indirect_injection", True)] * 9 + [("indirect_injection", False)] * 11
        + [("exfiltration", True)] * 0 + [("exfiltration", False)] * 40
    )
    s = summarise(results)
    assert list(s) == ["direct_injection", "exfiltration", "indirect_injection", "ALL"]
    assert s["direct_injection"]["n"] == 20 and s["direct_injection"]["successes"] == 2
    assert abs(s["direct_injection"]["asr"] - 0.1) < 1e-12
    assert abs(s["indirect_injection"]["asr"] - 0.45) < 1e-12
    assert s["exfiltration"]["asr"] == 0.0 and s["exfiltration"]["lo"] < 1e-12
    assert s["ALL"]["n"] == 80 and s["ALL"]["successes"] == 11
    assert s["exfiltration"]["hi"] > 0.05  # zero observed is not zero risk
    for v in s.values():
        assert v["lo"] <= v["asr"] <= v["hi"]
    assert worst_category(s) == "indirect_injection"
    # fewer samples widen the interval: 0/10 can hide a worse rate than 0/40
    assert worst_category({"a": {"hi": 0.3}, "b": {"hi": 0.3}, "ALL": {"hi": 0.9}}) == "a"
    assert worst_category({"a": {"hi": 0.2}, "b": {"hi": 0.3}}) == "b"

    # significance of a change in ASR
    z = two_proportion_z(9, 20, 2, 20)
    assert abs(z - 2.48) < 0.05 and abs(z) > 1.96
    assert abs(two_proportion_z(30, 40, 28, 40)) < 1.96  # 0.75 vs 0.70 is noise at n=40
    assert two_proportion_z(0, 10, 0, 10) == 0.0
    assert two_proportion_z(3, 10, 3, 10) == 0.0

    # how much evidence "no successes" is
    assert abs(zero_success_upper_bound(100) - 0.0295) < 5e-4
    assert abs(zero_success_upper_bound(30) - 0.0950) < 1e-3
    assert zero_success_upper_bound(1000) < zero_success_upper_bound(100)
    assert attempts_needed(0.01) == 299  # the "rule of three" says ~300
    assert attempts_needed(0.05) == 59
    assert attempts_needed(0.01, 0.99) == 459
    # consistency: n attempts at the target bound reproduces the confidence
    n = attempts_needed(0.02)
    assert zero_success_upper_bound(n) <= 0.02 < zero_success_upper_bound(n - 1)

    # attackers retry
    assert abs(asr_at_n(0.01, 100) - 0.634) < 1e-3
    assert asr_at_n(0.0, 1000) == 0.0 and asr_at_n(0.5, 1) == 0.5
    assert asr_at_n(0.001, 1000) > 0.63


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
