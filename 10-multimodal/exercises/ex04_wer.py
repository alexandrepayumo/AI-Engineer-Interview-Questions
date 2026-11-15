"""Exercise 04 - Word and character error rate with a full alignment.

TASK
----
Implement these functions (standard library only):

    normalize(text) -> str
        Lowercase, replace every character that is not a letter, digit, apostrophe or
        whitespace with a space, then collapse runs of whitespace and strip.

    _table(ref, hyp) -> list[list[int]]\n        The (len(ref) + 1) x (len(hyp) + 1) Levenshtein table d, where d[i][j] is the edit\n        distance between ref[:i] and hyp[:j]. The tests read d[-1][-1] from it.\n\n    align(ref, hyp) -> list[tuple[str, str | None, str | None]]
        ref and hyp are lists of tokens. Return the edit operations in order as
        (op, ref_token, hyp_token) with op in {"C", "S", "D", "I"} (correct, substitution,
        deletion, insertion). Use the standard Levenshtein table with unit costs. Trace
        back from the bottom-right cell and, when several moves are optimal, prefer in this
        order: diagonal (C or S), then deletion, then insertion. A deletion has
        hyp_token None, an insertion has ref_token None.

    error_counts(ref, hyp) -> dict[str, int]
        {"C": ..., "S": ..., "D": ..., "I": ...} from align.

    wer(ref_text, hyp_text) -> float
        (S + D + I) / number of reference words, after normalize and whitespace
        splitting. ValueError if the reference has no words. Can exceed 1.0.

    cer(ref_text, hyp_text) -> float
        The same ratio over characters (after normalize; spaces count as characters).

HINTS
-----
- Build d[i][j] = edit distance between ref[:i] and hyp[:j]. Row 0 and column 0 are 0..n.
- Total edits equal d[len(ref)][len(hyp)]. Your alignment must reproduce that number.
- Substitutions, deletions and insertions are reported separately because they mean
  different things: deletions often mean dropped speech, insertions often mean hallucination.

WHAT YOU SHOULD LEARN
---------------------
- WER is an edit distance divided by the reference length, so it is not bounded by 1.
- Why normalisation choices (case, punctuation, numbers) change the score more than most
  model upgrades do.
- Why S/D/I breakdowns help you diagnose an ASR system.
"""
from __future__ import annotations

import re


def normalize(text: str) -> str:
    raise NotImplementedError("TODO")


def _table(ref: list[str], hyp: list[str]) -> list[list[int]]:
    raise NotImplementedError("TODO")


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    raise NotImplementedError("TODO")


def error_counts(ref: list[str], hyp: list[str]) -> dict[str, int]:
    raise NotImplementedError("TODO")


def wer(ref_text: str, hyp_text: str) -> float:
    raise NotImplementedError("TODO")


def cer(ref_text: str, hyp_text: str) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    assert normalize("  Hello, WORLD!  It's 3.5 o'clock. ") == "hello world it's 3 5 o'clock"

    # classic example: one substitution and one deletion
    ref = "the cat sat on the mat".split()
    hyp = "the cat sit on mat".split()
    assert error_counts(ref, hyp) == {"C": 4, "S": 1, "D": 1, "I": 0}
    ops = align(ref, hyp)
    assert [o[0] for o in ops].count("D") == 1
    assert ("S", "sat", "sit") in ops
    assert abs(wer("the cat sat on the mat", "the cat sit on mat") - 2 / 6) < 1e-12

    # perfect and fully wrong
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b c", "x y z") == 1.0

    # WER above 1: hallucinated words are insertions
    assert error_counts(["a"], ["a", "b", "c", "d"]) == {"C": 1, "S": 0, "D": 0, "I": 3}
    assert wer("a", "a b c d") == 3.0

    # empty hypothesis: everything deleted, WER = 1
    assert wer("a b c", "") == 1.0
    assert error_counts(["a", "b"], []) == {"C": 0, "S": 0, "D": 2, "I": 0}

    # tie-break: swapped words are two substitutions (diagonal preferred)
    assert error_counts(["a", "b"], ["b", "a"]) == {"C": 0, "S": 2, "D": 0, "I": 0}

    # alignment reproduces the edit distance and consumes both sequences exactly
    cases = [
        ("turn the lights off in the kitchen", "turn lights of in the the kitchen please"),
        ("a b c d e f", "f e d c b a"),
        ("same", "same"),
        ("one two", "three four five six"),
    ]
    for r, h in cases:
        rt, ht = r.split(), h.split()
        ops = align(rt, ht)
        d = _table(rt, ht)[len(rt)][len(ht)]
        assert sum(1 for o in ops if o[0] != "C") == d
        assert [o[1] for o in ops if o[1] is not None] == rt
        assert [o[2] for o in ops if o[2] is not None] == ht

    # total edits are symmetric even though S/D/I roles flip
    a, b = "turn the lights off in the kitchen".split(), "turn lights of in the the kitchen please".split()
    ca, cb = error_counts(a, b), error_counts(b, a)
    assert ca["S"] + ca["D"] + ca["I"] == cb["S"] + cb["D"] + cb["I"]

    # character error rate: kitten -> sitting needs 3 edits over 6 characters
    assert cer("kitten", "sitting") == 0.5
    assert cer("abc", "abc") == 0.0
    # normalisation matters: punctuation and case are ignored, so this is a perfect match
    assert wer("Hello, World!", "hello world") == 0.0
    # but number formatting is not normalised: "10" versus "ten" costs a full substitution
    assert wer("i have ten apples", "i have 10 apples") == 0.25
    try:
        wer("", "anything")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
