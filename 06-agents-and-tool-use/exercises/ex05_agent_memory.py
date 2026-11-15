"""Exercise 05 - Agent memory store with recency, relevance and importance scoring.

TASK
----
Implement a MemoryStore in the style of the "Generative Agents" retrieval function.

    MemoryStore(embed, half_life=10.0, weights=(1.0, 1.0, 1.0))
        embed(text) -> np.ndarray (any length). Vectors are L2-normalised on add;
        a zero vector stays zero. weights = (w_recency, w_relevance, w_importance).

    add(text, t, importance=0.5) -> int
        Store a memory with created = last_access = t. Ids count up from 0.

    recency(mem, now) -> float
        0.5 ** ((now - mem.last_access) / half_life). Age is measured from the LAST ACCESS,
        so memories that keep being useful stay fresh.

    relevance(mem, query_vec) -> float
        Cosine similarity between the (normalised) memory and query vectors; 0.0 if the
        query vector is zero.

    score(mem, query_vec, now) -> float
        w_recency * recency + w_relevance * relevance + w_importance * importance.

    retrieve(query, now, k=3, min_score=None) -> list[Memory]
        Embed the query, score every memory, sort by score descending (ties: lower id
        first), keep memories with score >= min_score if it is given, return the top k.
        Side effect: set last_access = now on every returned memory.

    prune(now, min_recency, max_importance) -> int
        Delete memories whose recency is below min_recency AND whose importance is <=
        max_importance. Return how many were deleted.

HINTS
-----
- Score with the access time as it was BEFORE the retrieval touches it.
- Normalise vectors once on add so relevance is a dot product.
- The three terms live on different scales. Weights are how you tune them; some systems
  min-max normalise each term first. Here we use the raw values.

WHAT YOU SHOULD LEARN
---------------------
- Why pure similarity retrieval is not enough for an agent's memory: it ignores age and
  how much a memory matters.
- Exponential decay with a half-life, and why reinforcing on access keeps useful memories.
- How a forgetting policy bounds the store and keeps retrieval fast and clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Memory:
    id: int
    text: str
    vec: np.ndarray
    created: float
    last_access: float
    importance: float


class MemoryStore:
    def __init__(self, embed, half_life: float = 10.0, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.embed = embed
        self.half_life = half_life
        self.weights = weights
        self.memories: list[Memory] = []
        self._next_id = 0

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v  # zero vectors stay zero

    def add(self, text: str, t: float, importance: float = 0.5) -> int:
        raise NotImplementedError("TODO")

    def recency(self, mem: Memory, now: float) -> float:
        raise NotImplementedError("TODO")

    def relevance(self, mem: Memory, query_vec: np.ndarray) -> float:
        raise NotImplementedError("TODO")

    def score(self, mem: Memory, query_vec: np.ndarray, now: float) -> float:
        raise NotImplementedError("TODO")

    def retrieve(self, query: str, now: float, k: int = 3, min_score: float | None = None) -> list[Memory]:
        raise NotImplementedError("TODO")

    def prune(self, now: float, min_recency: float, max_importance: float) -> int:
        raise NotImplementedError("TODO")


def _run_tests() -> None:
    vocab = ["coffee", "tea", "meeting", "python", "deadline", "budget"]

    def embed(text: str) -> np.ndarray:
        words = text.lower().split()
        return np.array([float(words.count(w)) for w in vocab])

    # recency is an exponential with the configured half-life
    store = MemoryStore(embed, half_life=10.0)
    i0 = store.add("coffee", t=0.0, importance=0.4)
    m0 = store.memories[0]
    assert store.recency(m0, 0.0) == 1.0
    assert abs(store.recency(m0, 10.0) - 0.5) < 1e-12
    assert abs(store.recency(m0, 20.0) - 0.25) < 1e-12

    # exact score: recency 0.5 + relevance 1.0 + importance 0.4
    q = store._unit(embed("coffee"))
    assert abs(store.score(m0, q, now=10.0) - 1.9) < 1e-12
    # a query with no vocabulary overlap has zero relevance, and an all-zero query is safe
    assert store.relevance(m0, store._unit(embed("tea"))) == 0.0
    assert store.relevance(m0, store._unit(embed("xyz"))) == 0.0
    assert i0 == 0

    # ranking mixes all three signals
    store = MemoryStore(embed, half_life=10.0)
    a = store.add("python deadline", t=0.0, importance=0.9)   # old but important and relevant
    b = store.add("coffee tea", t=9.0, importance=0.1)        # fresh but irrelevant
    c = store.add("python budget", t=9.0, importance=0.1)     # fresh, relevant, unimportant
    assert (a, b, c) == (0, 1, 2)
    s = 1 / np.sqrt(2)
    q = store._unit(embed("python"))
    sa = 0.5 + s + 0.9
    sc = 0.5 ** 0.1 + s + 0.1
    sb = 0.5 ** 0.1 + 0.0 + 0.1
    assert abs(store.score(store.memories[0], q, 10.0) - sa) < 1e-12
    assert abs(store.score(store.memories[2], q, 10.0) - sc) < 1e-12
    assert abs(store.score(store.memories[1], q, 10.0) - sb) < 1e-12
    got = store.retrieve("python", now=10.0, k=3)
    assert [m.id for m in got] == [0, 2, 1]
    # retrieval reinforced the returned memories
    assert all(m.last_access == 10.0 for m in got)
    assert store.recency(store.memories[0], 10.0) == 1.0

    # k and min_score
    store2 = MemoryStore(embed)
    for i, txt in enumerate(["coffee", "tea", "meeting", "python"]):
        store2.add(txt, t=0.0, importance=0.5)
    assert len(store2.retrieve("coffee", now=0.0, k=10)) == 4
    assert [m.text for m in store2.retrieve("coffee", now=0.0, k=1)] == ["coffee"]
    # scores at now=0: coffee 1+1+0.5 = 2.5, others 1+0+0.5 = 1.5
    assert [m.text for m in store2.retrieve("coffee", now=0.0, k=10, min_score=2.0)] == ["coffee"]
    assert store2.retrieve("coffee", now=0.0, k=10, min_score=3.0) == []

    # ties go to the lower id, and k truncates after sorting
    store3 = MemoryStore(embed)
    for _ in range(3):
        store3.add("tea", t=0.0, importance=0.5)
    assert [m.id for m in store3.retrieve("tea", now=0.0, k=2)] == [0, 1]

    # weights: pure importance, pure recency
    store4 = MemoryStore(embed, weights=(0.0, 0.0, 1.0))
    store4.add("coffee", t=0.0, importance=0.2)
    store4.add("tea", t=0.0, importance=0.9)
    assert store4.retrieve("coffee", now=0.0, k=1)[0].text == "tea"
    store5 = MemoryStore(embed, weights=(1.0, 0.0, 0.0))
    store5.add("coffee", t=0.0, importance=0.9)
    store5.add("tea", t=8.0, importance=0.0)
    assert store5.retrieve("coffee", now=10.0, k=1)[0].text == "tea"

    # a reinforced old memory beats a never-touched one of the same age
    store6 = MemoryStore(embed, weights=(1.0, 0.0, 0.0))
    store6.add("coffee", t=0.0)
    store6.add("tea", t=0.0)
    store6.memories[0].last_access = 15.0
    assert store6.retrieve("anything", now=20.0, k=1)[0].text == "coffee"

    # prune: old AND unimportant goes, old but important stays, young stays
    store7 = MemoryStore(embed, half_life=10.0)
    store7.add("old trivia", t=0.0, importance=0.1)
    store7.add("old but vital", t=0.0, importance=0.9)
    store7.add("recent trivia", t=45.0, importance=0.1)
    removed = store7.prune(now=50.0, min_recency=0.1, max_importance=0.5)
    # recency at age 50 is 0.03125 (<0.1); at age 5 it is 0.707
    assert removed == 1
    assert [m.text for m in store7.memories] == ["old but vital", "recent trivia"]
    assert store7.add("new", t=50.0) == 3  # ids never reused


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
