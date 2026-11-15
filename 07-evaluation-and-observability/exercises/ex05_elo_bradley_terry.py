"""Exercise 05 - Ranking models from pairwise battles: Elo and Bradley-Terry.

TASK
----
A battle is a tuple (a, b, s): s is the score of `a` (1.0 = a wins, 0.0 = b wins,
0.5 = tie). Implement:

    expected_score(ra, rb, scale=400.0) -> float
        Elo expected score of a: 1 / (1 + 10 ** ((rb - ra) / scale)).
    elo_ratings(battles, k=32.0, base=1000.0, scale=400.0) -> dict[str, float]
        Process battles IN ORDER. Every player starts at `base`. For each battle
        compute the expected score from the current ratings, then
        ra += k * (s - ea) and rb -= k * (s - ea).
    bradley_terry(battles, iters=500) -> dict[str, float]
        Maximum-likelihood strengths p_i with P(i beats j) = p_i / (p_i + p_j), found with
        the MM update: p_i <- W_i / sum_j (n_ij / (p_i + p_j)), where W_i is the number
        of wins of i (a tie counts 0.5 for each side) and n_ij the number of battles
        between i and j. Start all p_i at 1.0 and rescale after every iteration so the
        geometric mean of the strengths is 1. Every player must have W_i > 0.
    strengths_to_elo(strengths, base=1000.0, scale=400.0) -> dict[str, float]
        rating = base + scale * log10(p_i) (geometric mean of 1 maps to `base`).

HINTS
-----
- Elo is an online algorithm and depends on the order of battles. Bradley-Terry fits all
  battles at once and does not.
- Rescale with exp(mean(log p)) to keep the solution unique.

WHAT YOU SHOULD LEARN
---------------------
- Why leaderboards that aggregate many votes use Bradley-Terry rather than online Elo:
  order independence and a proper likelihood, which also allows confidence intervals.
- A 400-point Elo gap means 10:1 odds; 3:1 odds is about 190 points.
"""
from __future__ import annotations

import math

import numpy as np


def expected_score(ra: float, rb: float, scale: float = 400.0) -> float:
    raise NotImplementedError("TODO")


def elo_ratings(battles: list[tuple[str, str, float]], k: float = 32.0, base: float = 1000.0,
                scale: float = 400.0) -> dict[str, float]:
    raise NotImplementedError("TODO")


def bradley_terry(battles: list[tuple[str, str, float]], iters: int = 500) -> dict[str, float]:
    raise NotImplementedError("TODO")


def strengths_to_elo(strengths: dict[str, float], base: float = 1000.0, scale: float = 400.0) -> dict[str, float]:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # expected score
    assert expected_score(1000, 1000) == 0.5
    assert abs(expected_score(1200, 1000) - 1 / (1 + 10 ** (-0.5))) < 1e-12
    assert abs(expected_score(1000, 1200) + expected_score(1200, 1000) - 1) < 1e-12
    # a 400 point gap is 10:1 odds
    assert abs(expected_score(1400, 1000) - 10 / 11) < 1e-12

    # one game between equals moves each player by k / 2
    r = elo_ratings([("a", "b", 1.0)], k=32)
    assert abs(r["a"] - 1016) < 1e-9 and abs(r["b"] - 984) < 1e-9
    # a tie between equals changes nothing
    r = elo_ratings([("a", "b", 0.5)])
    assert r == {"a": 1000.0, "b": 1000.0}
    # ratings are zero-sum around the base
    r = elo_ratings([("a", "b", 1.0), ("b", "c", 0.0), ("a", "c", 0.5)])
    assert abs(sum(r.values()) - 3000) < 1e-9

    # Elo depends on order, Bradley-Terry does not
    battles = [("a", "b", 1.0)] * 5 + [("b", "a", 1.0)] * 5 + [("a", "b", 1.0)] * 2
    flipped = list(reversed(battles))
    e1, e2 = elo_ratings(battles), elo_ratings(flipped)
    assert abs(e1["a"] - e2["a"]) > 1.0
    b1, b2 = bradley_terry(battles), bradley_terry(flipped)
    assert abs(b1["a"] - b2["a"]) < 1e-9

    # 3 wins out of 4 for a: strength ratio 3, about 190.8 Elo points apart
    bt = bradley_terry([("a", "b", 1.0)] * 3 + [("b", "a", 1.0)])
    assert abs(bt["a"] / bt["b"] - 3.0) < 1e-6
    assert abs(math.exp((np.log(bt["a"]) + np.log(bt["b"])) / 2) - 1.0) < 1e-9
    elo = strengths_to_elo(bt)
    assert abs((elo["a"] - elo["b"]) - 400 * math.log10(3)) < 1e-6
    assert abs((elo["a"] + elo["b"]) / 2 - 1000) < 1e-6

    # recover known strengths from simulated battles
    rng = np.random.default_rng(0)
    true = {"w": 1.0, "x": 2.0, "y": 4.0, "z": 8.0}
    names = list(true)
    sim = []
    for _ in range(6000):
        i, j = rng.choice(4, size=2, replace=False)
        a, b = names[i], names[j]
        pa = true[a] / (true[a] + true[b])
        sim.append((a, b, 1.0 if rng.random() < pa else 0.0))
    est = bradley_terry(sim)
    assert sorted(est, key=est.get) == ["w", "x", "y", "z"]
    for lo, hi in (("w", "x"), ("x", "y"), ("y", "z")):
        assert abs(math.log2(est[hi] / est[lo]) - 1.0) < 0.2  # each step is a doubling

    # ties count half for each side
    bt = bradley_terry([("a", "b", 1.0), ("b", "a", 1.0), ("a", "b", 0.5), ("a", "b", 0.5)])
    assert abs(bt["a"] - bt["b"]) < 1e-9


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
