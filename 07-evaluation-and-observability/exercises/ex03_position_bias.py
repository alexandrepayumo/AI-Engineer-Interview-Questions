"""Exercise 03 - Detecting position bias in a pairwise LLM judge.

TASK
----
A pairwise judge sees two answers, one in the first slot and one in the second, and says
"first", "second" or "tie". To test it, every pair (A, B) is judged twice: once with A in
the first slot and once with B in the first slot. Each record is a tuple
(v1, v2): v1 = verdict with A first, v2 = verdict with B first.

Implement:

    to_winner(verdict, a_first) -> str
        Map a slot verdict to "A", "B" or "tie". If a_first, "first" means A.
        If not a_first, "first" means B. Raise ValueError for unknown verdicts.
    outcome_counts(runs) -> dict
        Counts over the records: {"A": ..., "B": ..., "tie": ..., "inconsistent": ...}.
        A record is consistent only if both orders name the same winner (tie/tie counts
        as "tie"). Anything else, including tie in one order and a winner in the other,
        is "inconsistent".
    consistency_rate(runs) -> float
        Fraction of records that are consistent.
    first_slot_share(runs) -> float
        Among all non-tie verdicts (both orders pooled), the fraction that picked the
        first slot. 0.5 means no slot preference. Return 0.5 if there are no non-tie verdicts.
    binomial_two_sided_p(k, n) -> float
        Exact two-sided p-value of seeing k or a more extreme count out of n under p = 0.5:
        min(1, 2 * P(X >= max(k, n - k))). Use math.comb. Return 1.0 if n == 0.
    debiased_win_rate(runs) -> float
        Win rate of A among consistent, non-tie records: A / (A + B).
        Return 0.5 if there are none.

HINTS
-----
- Convert both verdicts to winners first, then compare.
- A record where the judge always says "first" is inconsistent in every pair.

WHAT YOU SHOULD LEARN
---------------------
- Swapping the order and keeping only consistent verdicts removes position bias from the
  result, at the cost of discarding data. The consistency rate is a quality metric of the
  judge itself.
- A first-slot share far from 0.5 is a measurable, testable bias, not an impression.
"""
from __future__ import annotations

from math import comb


def to_winner(verdict: str, a_first: bool) -> str:
    raise NotImplementedError("TODO")


def outcome_counts(runs: list[tuple[str, str]]) -> dict:
    raise NotImplementedError("TODO")


def consistency_rate(runs: list[tuple[str, str]]) -> float:
    raise NotImplementedError("TODO")


def first_slot_share(runs: list[tuple[str, str]]) -> float:
    raise NotImplementedError("TODO")


def binomial_two_sided_p(k: int, n: int) -> float:
    raise NotImplementedError("TODO")


def debiased_win_rate(runs: list[tuple[str, str]]) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    assert to_winner("first", True) == "A"
    assert to_winner("second", True) == "B"
    assert to_winner("first", False) == "B"
    assert to_winner("second", False) == "A"
    assert to_winner("tie", False) == "tie"
    try:
        to_winner("left", True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    # mixed example: 4 x A wins, 2 x B wins, 3 x always-first, 1 tie
    runs = (
        [("first", "second")] * 4      # A in slot 1 wins, then A in slot 2 wins: A both times
        + [("second", "first")] * 2    # B both times
        + [("first", "first")] * 3     # picks slot 1 regardless: inconsistent
        + [("tie", "tie")]
    )
    assert outcome_counts(runs) == {"A": 4, "B": 2, "tie": 1, "inconsistent": 3}
    assert abs(consistency_rate(runs) - 0.7) < 1e-12
    # non-tie verdicts: 4*2 + 2*2 + 3*2 = 18, of which "first": 4 + 2 + 6 = 12
    assert abs(first_slot_share(runs) - 12 / 18) < 1e-12
    assert abs(debiased_win_rate(runs) - 4 / 6) < 1e-12

    # exact binomial p-values
    assert binomial_two_sided_p(9, 18) == 1.0
    assert abs(binomial_two_sided_p(18, 18) - 2 * 0.5 ** 18) < 1e-15
    assert abs(binomial_two_sided_p(12, 18) - 2 * 31180 / 2 ** 18) < 1e-12
    assert binomial_two_sided_p(6, 18) == binomial_two_sided_p(12, 18)  # symmetric
    assert binomial_two_sided_p(0, 0) == 1.0

    # a judge that always says "first": zero consistency, share 1.0, tiny p-value
    biased = [("first", "first")] * 50
    assert consistency_rate(biased) == 0.0
    assert first_slot_share(biased) == 1.0
    assert binomial_two_sided_p(100, 100) < 1e-20
    assert debiased_win_rate(biased) == 0.5  # no consistent decisive records

    # a perfectly position-blind judge that prefers A: full consistency
    blind = [("first", "second")] * 30
    assert consistency_rate(blind) == 1.0
    assert first_slot_share(blind) == 0.5
    assert debiased_win_rate(blind) == 1.0

    # tie in one order and a winner in the other is inconsistent
    assert outcome_counts([("tie", "first")])["inconsistent"] == 1
    assert first_slot_share([("tie", "tie")]) == 0.5


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
