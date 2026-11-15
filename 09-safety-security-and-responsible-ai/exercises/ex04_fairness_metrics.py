"""Exercise 04 - Fairness metrics calculator and counterfactual testing.

TASK
----
All predictions and labels are 0/1 lists aligned with a list of group names.

    selection_rates(preds, groups) -> dict[str, float]
        Fraction of positive predictions per group.

    demographic_parity_diff(preds, groups) -> float
        max selection rate - min selection rate.

    disparate_impact_ratio(preds, groups) -> float
        min selection rate / max selection rate. If the max is 0 return 1.0.

    four_fifths_check(preds, groups) -> bool
        True when disparate_impact_ratio >= 0.8 (allow 1e-12 of float slack).

    group_tpr_fpr(preds, labels, groups) -> dict[str, tuple[float, float]]
        (TPR, FPR) per group. A group with no positives or no negatives raises ValueError
        (the rate is undefined, so do not silently return 0).

    equalised_odds_gap(preds, labels, groups) -> float
        max over groups of TPR minus min TPR, or the same for FPR, whichever is larger.

    counterfactual_flip_rate(predict, texts, swaps) -> float
        predict maps a string to 0/1. swaps maps a term to its counterpart, for example
        {"he": "she", "she": "he"}. For each text, replace whole words (case-sensitive,
        regex \\b boundaries, all swaps applied simultaneously, so "he" and "she" do not
        chain) and count a flip when predict changes. Texts where no swap term occurs are
        skipped. Return flips / tested, or 0.0 when nothing was tested.

HINTS
-----
- Build a dict of lists per group first. Use sorted group names for determinism.
- For simultaneous swaps use one re.sub with a function that looks the match up in swaps.

WHAT YOU SHOULD LEARN
---------------------
- Demographic parity ignores ground truth, equalised odds conditions on it. They answer
  different questions and generally cannot both hold when base rates differ.
- The four-fifths rule is a screening heuristic from US employment practice, not a proof
  of fairness or of discrimination.
- Counterfactual testing needs no labels, so it works on an LLM-based screener you do not
  control.
"""
from __future__ import annotations

import re


def _by_group(values: list, groups: list[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for v, g in zip(values, groups):
        out.setdefault(g, []).append(v)
    return out


def selection_rates(preds: list[int], groups: list[str]) -> dict[str, float]:
    raise NotImplementedError("TODO")


def demographic_parity_diff(preds: list[int], groups: list[str]) -> float:
    raise NotImplementedError("TODO")


def disparate_impact_ratio(preds: list[int], groups: list[str]) -> float:
    raise NotImplementedError("TODO")


def four_fifths_check(preds: list[int], groups: list[str]) -> bool:
    raise NotImplementedError("TODO")


def group_tpr_fpr(preds: list[int], labels: list[int],
                  groups: list[str]) -> dict[str, tuple[float, float]]:
    raise NotImplementedError("TODO")


def equalised_odds_gap(preds: list[int], labels: list[int], groups: list[str]) -> float:
    raise NotImplementedError("TODO")


def counterfactual_flip_rate(predict, texts: list[str], swaps: dict[str, str]) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # Group A selects 60 of 100, group B selects 42 of 100
    groups = ["A"] * 100 + ["B"] * 100
    preds = [1] * 60 + [0] * 40 + [1] * 42 + [0] * 58
    rates = selection_rates(preds, groups)
    assert rates == {"A": 0.6, "B": 0.42}
    assert abs(demographic_parity_diff(preds, groups) - 0.18) < 1e-12
    assert abs(disparate_impact_ratio(preds, groups) - 0.7) < 1e-12
    assert four_fifths_check(preds, groups) is False
    # 48 of 100 vs 60 of 100 gives exactly 0.8, which passes
    preds2 = [1] * 60 + [0] * 40 + [1] * 48 + [0] * 52
    assert abs(disparate_impact_ratio(preds2, groups) - 0.8) < 1e-12
    assert four_fifths_check(preds2, groups) is True
    assert disparate_impact_ratio([0, 0, 0, 0], ["A", "A", "B", "B"]) == 1.0
    assert demographic_parity_diff([1, 0], ["A", "B"]) == 1.0
    # works with three groups
    g3 = ["A", "B", "C"] * 10
    p3 = [1, 1, 0] * 10
    assert disparate_impact_ratio(p3, g3) == 0.0

    # Equalised odds: both groups have 50 positives and 50 negatives
    #   A: TPR 0.8 (40/50), FPR 0.1 (5/50)    B: TPR 0.6 (30/50), FPR 0.1 (5/50)
    labels = ([1] * 50 + [0] * 50) * 2
    preds = ([1] * 40 + [0] * 10 + [1] * 5 + [0] * 45
             + [1] * 30 + [0] * 20 + [1] * 5 + [0] * 45)
    assert group_tpr_fpr(preds, labels, groups) == {"A": (0.8, 0.1), "B": (0.6, 0.1)}
    assert abs(equalised_odds_gap(preds, labels, groups) - 0.2) < 1e-12
    # selection rates here are 0.45 and 0.35, so parity and odds give different numbers
    assert abs(demographic_parity_diff(preds, groups) - 0.10) < 1e-12
    # a perfect classifier satisfies equalised odds even when base rates differ
    lab2 = [1] * 30 + [0] * 70 + [1] * 10 + [0] * 90
    assert equalised_odds_gap(lab2, lab2, groups) == 0.0
    assert abs(demographic_parity_diff(lab2, groups) - 0.2) < 1e-12  # parity violated: 0.30 vs 0.10
    try:
        group_tpr_fpr([1, 0], [1, 1], ["A", "A"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a group with no negatives")

    # Counterfactual flips with a deliberately biased toy screener
    def biased(text: str) -> int:
        return 0 if re.search(r"\bshe\b", text) else 1

    texts = ["he has 5 years experience", "she has 5 years experience",
             "the candidate has 5 years experience", "he led a team", "she led a team"]
    swaps = {"he": "she", "she": "he"}
    # 4 texts contain a term, and each flips under the biased screener
    assert counterfactual_flip_rate(biased, texts, swaps) == 1.0

    def fair(text: str) -> int:
        return 1 if "years" in text or "team" in text else 0

    assert counterfactual_flip_rate(fair, texts, swaps) == 0.0
    assert counterfactual_flip_rate(fair, ["no terms here"], swaps) == 0.0
    # simultaneous swap: "he told she" -> "she told he", not "he told he"
    seen: list[str] = []
    counterfactual_flip_rate(lambda t: seen.append(t) or 1, ["he told she"], swaps)
    assert seen == ["she told he", "he told she"] or seen == ["he told she", "she told he"]
    # word boundaries: "the" and "shell" are untouched
    seen.clear()
    counterfactual_flip_rate(lambda t: seen.append(t) or 1, ["the shell he"], swaps)
    assert "the shell she" in seen


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
