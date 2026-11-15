"""Exercise 01 - Retrieval metrics by hand: recall@k, MRR, nDCG, average precision.

TASK
----
Implement these functions (ranked lists contain document ids, best first):

    recall_at_k(ranked, relevant, k) -> float
        |top-k intersect relevant| / |relevant|. Return 0.0 if relevant is empty.
    precision_at_k(ranked, relevant, k) -> float
        |top-k intersect relevant| / k.
    reciprocal_rank(ranked, relevant) -> float
        1 / (rank of first relevant doc, 1-based). 0.0 if none appears.
    mrr(rankings, relevants) -> float
        Mean reciprocal rank over queries.
    dcg_at_k(gains, k) -> float
        sum over i in 0..k-1 of gains[i] / log2(i + 2). If fewer than k gains, use what exists.
    ndcg_at_k(ranked, grades, k, exponential=True) -> float
        grades maps doc id -> graded relevance (int, missing = 0).
        gain = 2**grade - 1 if exponential else grade.
        nDCG = DCG of the ranking / DCG of the ideal ordering (all graded docs sorted by
        grade, descending). Return 0.0 if the ideal DCG is 0.
    average_precision(ranked, relevant) -> float
        Mean of precision@i over the ranks i where a relevant doc appears,
        divided by |relevant| (so missed relevant docs count as zero).

HINTS
-----
- The ideal ordering is built from ALL graded docs, not only the ones you retrieved.
- Use a set for `relevant` so membership tests are O(1).
- The log base is 2 and the first rank is discounted by log2(2) = 1.

WHAT YOU SHOULD LEARN
---------------------
- recall@k gates the generator: if the gold chunk is not in top-k, nothing downstream helps.
- MRR only looks at the first hit; nDCG rewards graded relevance and the position of every hit.
- Exponential gain (2**g - 1) makes a grade-3 doc worth 7x a grade-1 doc, linear gain only 3x.
"""
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def mrr(rankings: list[list[str]], relevants: list[set[str]]) -> float:
    rr = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants)]
    return sum(rr) / len(rr)


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int, exponential: bool = True) -> float:
    def gain(g: int) -> float:
        return (2 ** g - 1) if exponential else float(g)

    actual = dcg_at_k([gain(grades.get(d, 0)) for d in ranked], k)
    # Ideal ordering uses every graded doc, sorted best first.
    ideal_gains = sorted((gain(g) for g in grades.values()), reverse=True)
    ideal = dcg_at_k(ideal_gains, k)
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def _run_tests() -> None:
    ranked = ["a", "b", "c", "d", "e"]

    # recall / precision
    assert recall_at_k(ranked, {"a", "c", "z"}, 3) == 2 / 3
    assert recall_at_k(ranked, {"a", "c", "z"}, 1) == 1 / 3
    assert recall_at_k(ranked, set(), 3) == 0.0
    assert recall_at_k(ranked, {"a"}, 100) == 1.0
    assert precision_at_k(ranked, {"a", "c", "z"}, 3) == 2 / 3
    assert precision_at_k(ranked, {"a", "c", "z"}, 5) == 2 / 5

    # reciprocal rank and MRR
    assert reciprocal_rank(ranked, {"c"}) == 1 / 3
    assert reciprocal_rank(ranked, {"c", "b"}) == 1 / 2
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0
    m = mrr([ranked, ranked, ranked], [{"a"}, {"c"}, {"zzz"}])
    assert abs(m - (1 + 1 / 3 + 0) / 3) < 1e-12

    # DCG: 3/1 + 2/log2(3) + 3/2 + 0/log2(5) + 1/log2(6)
    gains = [3, 2, 3, 0, 1]
    expected = 3 + 2 / math.log2(3) + 3 / 2 + 0 + 1 / math.log2(6)
    assert abs(dcg_at_k(gains, 5) - expected) < 1e-12
    assert abs(dcg_at_k(gains, 2) - (3 + 2 / math.log2(3))) < 1e-12
    assert dcg_at_k(gains, 50) == dcg_at_k(gains, 5)

    # nDCG with linear gain: ranking has grades [3,2,3,0,1], ideal is [3,3,2,1,0]
    grades = {"a": 3, "b": 2, "c": 3, "d": 0, "e": 1}
    ideal = 3 + 3 / math.log2(3) + 2 / 2 + 1 / math.log2(5) + 0
    got = ndcg_at_k(ranked, grades, 5, exponential=False)
    assert abs(got - expected / ideal) < 1e-12
    assert 0.9 < got < 1.0  # about 0.972

    # nDCG with exponential gain punishes the swap of the two top docs more
    swapped = ["b", "a", "c", "d", "e"]
    lin = ndcg_at_k(swapped, grades, 5, exponential=False)
    exp = ndcg_at_k(swapped, grades, 5, exponential=True)
    assert exp < lin < 1.0

    # perfect ranking gives exactly 1; empty-grades gives 0
    assert abs(ndcg_at_k(["a", "c", "b", "e", "d"], grades, 5) - 1.0) < 1e-12
    assert ndcg_at_k(ranked, {}, 5) == 0.0
    # the ideal list includes graded docs that were never retrieved
    assert ndcg_at_k(["a"], {"a": 1, "b": 1}, 2, exponential=False) < 1.0

    # average precision: relevant at ranks 1 and 3, one relevant never retrieved
    ap = average_precision(ranked, {"a", "c", "zzz"})
    assert abs(ap - (1 / 1 + 2 / 3) / 3) < 1e-12
    assert average_precision(ranked, set()) == 0.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
