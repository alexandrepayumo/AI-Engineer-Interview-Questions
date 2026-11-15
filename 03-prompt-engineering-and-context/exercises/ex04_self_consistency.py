"""Exercise 04 - Self-consistency: answer extraction, normalisation and voting.

TASK
----
    def extract_answer(completion: str) -> str | None
        '''Return the final answer string from a chain-of-thought completion.
        If the text contains "answer:" or "answer is" (case-insensitive), take what
        follows the LAST occurrence up to the end of that line. Otherwise fall back
        to the last number in the text. Return None if neither exists.'''

    def normalise_answer(ans: str) -> str
        '''Canonical form so that "$1,200.00", "1200" and "1200." compare equal.
        Strip whitespace, lowercase, remove "$", "," and a trailing ".".
        If the result parses as a float, format it with no trailing zeros
        ("18.50" -> "18.5", "18.0" -> "18"). Otherwise return the cleaned string.'''

    def majority_vote(completions: list[str]) -> tuple[str | None, float]
        '''Extract + normalise each completion, ignore those with no answer, and
        return (winning answer, agreement) where agreement = votes for winner /
        number of completions that produced an answer. Ties go to the answer that
        appeared first. No answers at all -> (None, 0.0).'''

    def vote_accuracy(p: float, k: int) -> float
        '''Probability that a majority of k independent samples is correct when each
        is correct with probability p and wrong answers never agree with each other
        (worst case for voting: wrong votes are spread out, so correct needs > k/2).
        k must be odd and >= 1, else ValueError.'''

HINTS
-----
- re.findall(r"-?\\d[\\d,]*\\.?\\d*", text) finds numbers like "1,200.50".
- collections.Counter plus first-seen order handles tie-breaking.
- vote_accuracy is a binomial tail: sum over i > k/2 of C(k, i) p^i (1-p)^(k-i).

WHAT YOU SHOULD LEARN
---------------------
Self-consistency only works if equivalent answers are recognised as equal, so
normalisation is half the technique. The agreement fraction doubles as a cheap
confidence signal. The binomial formula shows why it helps when p > 0.5 and
hurts when p < 0.5, and why the gain per extra sample shrinks.
"""
from __future__ import annotations

import re
from collections import Counter
from math import comb

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(completion: str) -> str | None:
    raise NotImplementedError("TODO")


def normalise_answer(ans: str) -> str:
    raise NotImplementedError("TODO")


def majority_vote(completions: list[str]) -> tuple[str | None, float]:
    raise NotImplementedError("TODO")


def vote_accuracy(p: float, k: int) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    assert extract_answer("Step 1 ... so the Answer: 42") == "42"
    assert extract_answer("thinking\nThe answer is $1,200.00\nthanks") == "$1,200.00"
    assert extract_answer("answer: 5\nwait, recheck.\nAnswer: 7") == "7"   # last occurrence wins
    assert extract_answer("we get 3 apples then 12 more") == "12"           # fallback: last number
    assert extract_answer("no digits at all") is None

    assert normalise_answer(" $1,200.00 ") == "1200"
    assert normalise_answer("1200.") == "1200"
    assert normalise_answer("18.50") == "18.5"
    assert normalise_answer("Paris") == "paris"
    assert normalise_answer("-3.0") == "-3"

    runs = [
        "so 6*3 = 18. Answer: 18",
        "the answer is $18.00",
        "Answer: 18.0",
        "Answer: 20",
        "I give up, no digits",
    ]
    assert majority_vote(runs) == ("18", 0.75)
    assert majority_vote(["Answer: a", "Answer: b"]) == ("a", 0.5)       # tie -> first seen
    assert majority_vote(["Answer: b", "Answer: a", "Answer: a"])[0] == "a"
    assert majority_vote(["nothing here"]) == (None, 0.0)
    assert majority_vote([]) == (None, 0.0)

    assert abs(vote_accuracy(0.7, 1) - 0.7) < 1e-12
    assert abs(vote_accuracy(0.7, 3) - 0.784) < 1e-9            # 3*.49*.3 + .343
    assert abs(vote_accuracy(0.7, 5) - 0.83692) < 1e-9
    assert vote_accuracy(0.4, 5) < 0.4                          # voting hurts when p < 0.5
    assert abs(vote_accuracy(0.5, 9) - 0.5) < 1e-12
    gains = [vote_accuracy(0.7, k) for k in (1, 3, 5, 7, 9)]
    assert all(b > a for a, b in zip(gains, gains[1:]))
    assert (gains[1] - gains[0]) > (gains[2] - gains[1]) > (gains[3] - gains[2])  # diminishing returns
    try:
        vote_accuracy(0.7, 4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
