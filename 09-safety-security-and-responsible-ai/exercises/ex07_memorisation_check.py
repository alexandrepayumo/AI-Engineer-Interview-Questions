"""Exercise 07 - Memorisation checks: n-gram overlap, longest verbatim span, membership AUC.

TASK
----
Two simple tools for auditing whether a model reproduces training text.

    tokenise(text) -> list[str]
        Lower-case, then split into runs of letters, digits and apostrophes
        (regex [a-z0-9']+). Punctuation is dropped.

    ngrams(tokens, n) -> list[tuple[str, ...]]
        All contiguous n-grams in order (empty list if len(tokens) < n). n < 1 raises
        ValueError.

    build_index(docs, n) -> set[tuple[str, ...]]
        The set of all n-grams in all docs. Each doc is tokenised on its own, so n-grams
        never span a document boundary.

    overlap_ratio(output, index, n) -> float
        Fraction of the output's n-grams that are in the index. 0.0 if the output has no
        n-grams.

    longest_verbatim_span(output, doc) -> int
        Length in tokens of the longest contiguous token run shared by both texts.
        (Classic longest-common-substring DP over tokens.)

    flag_memorised(output, index, n, threshold) -> bool
        True when overlap_ratio(...) >= threshold.

    mia_auc(member_losses, nonmember_losses) -> float
        Loss-threshold membership-inference AUC: the probability that a randomly chosen
        member has LOWER loss than a randomly chosen non-member, counting ties as 0.5.
        0.5 means the attack learns nothing.

HINTS
-----
- A set of tuples is a fine index at this scale. Real systems use suffix arrays or Bloom
  filters over billions of tokens.
- For the AUC an O(m * k) double loop is fine.

WHAT YOU SHOULD LEARN
---------------------
- Exact n-gram overlap is the cheapest memorisation probe and has a clear failure mode: it
  misses paraphrase and near-verbatim copies with a single changed token.
- Choosing n is a trade-off. Short n flags common phrases, long n misses edited copies.
- Membership inference scores a model, not a string, and AUC near 0.5 is the healthy
  result. Duplicated training text pushes it up.
"""
from __future__ import annotations

import re


def tokenise(text: str) -> list[str]:
    raise NotImplementedError("TODO")


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    raise NotImplementedError("TODO")


def build_index(docs: list[str], n: int) -> set[tuple[str, ...]]:
    raise NotImplementedError("TODO")


def overlap_ratio(output: str, index: set[tuple[str, ...]], n: int) -> float:
    raise NotImplementedError("TODO")


def longest_verbatim_span(output: str, doc: str) -> int:
    raise NotImplementedError("TODO")


def flag_memorised(output: str, index: set[tuple[str, ...]], n: int, threshold: float) -> bool:
    raise NotImplementedError("TODO")


def mia_auc(member_losses: list[float], nonmember_losses: list[float]) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    assert tokenise("Hello, World! It's 2026.") == ["hello", "world", "it's", "2026"]
    assert tokenise("") == [] and tokenise("!!!") == []

    toks = ["a", "b", "c", "d"]
    assert ngrams(toks, 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ngrams(toks, 4) == [("a", "b", "c", "d")]
    assert ngrams(toks, 5) == []
    try:
        ngrams(toks, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    docs = ["The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs."]
    idx = build_index(docs, 3)
    assert ("the", "quick", "brown") in idx and ("five", "dozen", "liquor") in idx
    assert ("dog", "pack", "my") not in idx          # no n-gram spans two documents
    assert len(idx) == 7 + 6                          # 9-token and 8-token docs

    # verbatim copy
    out = "Sure: the quick brown fox jumps over the lazy dog."
    # 10 tokens give 8 trigrams, and 7 are in the index (only "sure the quick" is new)
    assert abs(overlap_ratio(out, idx, 3) - 7 / 8) < 1e-12
    assert overlap_ratio("completely unrelated sentence about cooking pasta", idx, 3) == 0.0
    assert overlap_ratio("two words", idx, 3) == 0.0     # shorter than n

    # one changed token breaks every n-gram that spans it: the cost of exact matching
    edited = "the quick brown cat jumps over the lazy dog"
    assert overlap_ratio(edited, idx, 3) < overlap_ratio(out, idx, 3)
    assert abs(overlap_ratio(edited, idx, 3) - 4 / 7) < 1e-12   # 7 trigrams, 4 survive
    assert abs(overlap_ratio(edited, build_index(docs, 2), 2) - 6 / 8) < 1e-12
    assert abs(overlap_ratio(edited, build_index(docs, 5), 5) - 1 / 5) < 1e-12
    # shorter n is more forgiving, longer n is stricter
    i2, i5 = build_index(docs, 2), build_index(docs, 5)
    assert overlap_ratio(edited, i2, 2) > overlap_ratio(edited, idx, 3) > overlap_ratio(edited, i5, 5)

    assert flag_memorised(out, idx, 3, 0.8) is True
    assert flag_memorised(edited, idx, 3, 0.8) is False
    assert flag_memorised(edited, idx, 3, 0.4) is True

    assert longest_verbatim_span(out, docs[0]) == 9
    assert longest_verbatim_span(edited, docs[0]) == 5   # "jumps over the lazy dog"
    assert longest_verbatim_span("nothing in common", docs[0]) == 0
    assert longest_verbatim_span("", docs[0]) == 0
    assert longest_verbatim_span("a b c a b c", "c a b") == 3

    # membership inference
    assert abs(mia_auc([1, 2, 3], [2, 3, 4]) - 7 / 9) < 1e-12
    assert mia_auc([1.0, 1.1], [5.0, 6.0]) == 1.0          # members clearly lower
    assert mia_auc([5.0, 6.0], [1.0, 1.1]) == 0.0
    assert mia_auc([2.0, 2.0], [2.0, 2.0]) == 0.5           # all ties: no signal
    assert abs(mia_auc([1.0, 3.0], [2.0, 2.0]) - 0.5) < 1e-12


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
