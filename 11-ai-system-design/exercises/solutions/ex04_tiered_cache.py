"""Exercise 04 - Two-tier response cache (exact + semantic): hit rates, cost and wrong hits.

TASK
----
Implement a cache with two tiers and a simulator.

    normalise(text) -> str
        Lower-case, strip, and collapse runs of whitespace to a single space.

    class TieredCache:
        __init__(self, exact_capacity: int, semantic_capacity: int, threshold: float)
        lookup(self, text: str, emb: np.ndarray) -> tuple[str, str | None]
            Returns ("exact", answer) if normalise(text) is in the exact tier, else
            ("semantic", answer) if the best cosine similarity against the semantic tier
            is >= threshold (answer of that best entry), else ("miss", None).
            An exact hit refreshes that key's recency (LRU).
        insert(self, text: str, emb: np.ndarray, answer: str) -> None
            Store in both tiers. The exact tier is LRU with exact_capacity entries. The
            semantic tier is FIFO with semantic_capacity entries (oldest dropped first).
            Embeddings are normalised to unit length on insert and on lookup.

    simulate(queries, cache, llm_cost, embed_cost) -> dict
        queries is a list of (text, emb, intent). The correct answer for a query is its
        intent string. For each query: call cache.lookup. On a miss, "call the LLM" (the
        answer is the intent) and insert. Costs: an exact hit costs 0. Every other query
        pays embed_cost for the embedding, and a miss also pays llm_cost.
        Return {"exact_hit_rate", "semantic_hit_rate", "miss_rate", "wrong_hit_rate",
                "cost", "baseline_cost"}.
        wrong_hit_rate = fraction of all queries served a stale or wrong answer (a hit
        whose answer != intent). baseline_cost = len(queries) * llm_cost (no cache).

    expected_cost_per_query(exact_rate, semantic_rate, llm_cost, embed_cost) -> float
        Closed form: (1 - exact_rate) * embed_cost + (1 - exact_rate - semantic_rate) * llm_cost

HINTS
-----
- collections.OrderedDict gives LRU with move_to_end and popitem(last=False).
- Keep the semantic tier as parallel Python lists, or a 2D array rebuilt on insert. The
  tests use only a few dozen entries.
- Cosine similarity of unit vectors is a dot product.
- A semantic hit does not refresh anything. Only inserts and exact hits change state.

WHAT YOU SHOULD LEARN
---------------------
- Exact caching is free and safe. Semantic caching costs an embedding call per lookup and
  can serve a wrong answer, so the threshold is a precision knob you tune on labelled pairs.
- The cost saving is a function of hit rates, and a wrong hit is a quality incident, not
  a saving.
- How capacity and eviction policy change hit rates for a skewed query stream.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TieredCache:
    def __init__(self, exact_capacity: int, semantic_capacity: int, threshold: float):
        self.exact_capacity = exact_capacity
        self.semantic_capacity = semantic_capacity
        self.threshold = threshold
        self._exact: OrderedDict[str, str] = OrderedDict()
        self._sem_vecs: list[np.ndarray] = []
        self._sem_answers: list[str] = []

    def lookup(self, text: str, emb: np.ndarray) -> tuple[str, str | None]:
        key = normalise(text)
        if key in self._exact:
            self._exact.move_to_end(key)  # refresh LRU recency
            return "exact", self._exact[key]
        if self._sem_vecs:
            sims = np.vstack(self._sem_vecs) @ _unit(emb)
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                return "semantic", self._sem_answers[best]
        return "miss", None

    def insert(self, text: str, emb: np.ndarray, answer: str) -> None:
        key = normalise(text)
        self._exact[key] = answer
        self._exact.move_to_end(key)
        while len(self._exact) > self.exact_capacity:
            self._exact.popitem(last=False)  # evict least recently used
        self._sem_vecs.append(_unit(emb))
        self._sem_answers.append(answer)
        while len(self._sem_vecs) > self.semantic_capacity:
            self._sem_vecs.pop(0)  # FIFO
            self._sem_answers.pop(0)


def simulate(
    queries: list[tuple[str, np.ndarray, str]],
    cache: TieredCache,
    llm_cost: float,
    embed_cost: float,
) -> dict:
    n = len(queries)
    counts = {"exact": 0, "semantic": 0, "miss": 0}
    wrong = 0
    cost = 0.0
    for text, emb, intent in queries:
        tier, answer = cache.lookup(text, emb)
        counts[tier] += 1
        if tier == "exact":
            pass  # free
        elif tier == "semantic":
            cost += embed_cost
        else:
            cost += embed_cost + llm_cost
            cache.insert(text, emb, intent)
        if tier != "miss" and answer != intent:
            wrong += 1
    return {
        "exact_hit_rate": counts["exact"] / n,
        "semantic_hit_rate": counts["semantic"] / n,
        "miss_rate": counts["miss"] / n,
        "wrong_hit_rate": wrong / n,
        "cost": cost,
        "baseline_cost": n * llm_cost,
    }


def expected_cost_per_query(
    exact_rate: float, semantic_rate: float, llm_cost: float, embed_cost: float
) -> float:
    return (1 - exact_rate) * embed_cost + (1 - exact_rate - semantic_rate) * llm_cost


def _run_tests() -> None:
    assert normalise("  What   IS\tX ") == "what is x"

    A, B, C = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.8, 0.6])
    queries = [
        ("What is X", A, "A"),
        ("  what is x ", A, "A"),                    # same after normalising
        ("tell me about X", np.array([0.98, 0.2]), "A"),  # paraphrase, cos ~ 0.98
        ("what is Y", B, "B"),
        ("is X maybe Z", C, "C"),                    # near A (0.8) but a different intent
    ]
    llm, emb = 0.01, 0.0001

    strict = simulate(queries, TieredCache(10, 10, 0.95), llm, emb)
    assert strict["exact_hit_rate"] == 0.2 and strict["semantic_hit_rate"] == 0.2
    assert abs(strict["miss_rate"] - 0.6) < 1e-12 and strict["wrong_hit_rate"] == 0.0
    assert abs(strict["cost"] - (4 * emb + 3 * llm)) < 1e-12  # 4 non-exact lookups, 3 misses
    assert abs(strict["baseline_cost"] - 0.05) < 1e-12
    assert strict["cost"] < strict["baseline_cost"]

    loose = simulate(queries, TieredCache(10, 10, 0.75), llm, emb)
    # C is served A's answer: a cheaper but wrong response
    assert loose["semantic_hit_rate"] == 0.4 and loose["wrong_hit_rate"] == 0.2
    assert loose["cost"] < strict["cost"]

    # the closed form agrees with the simulation
    for res in (strict, loose):
        closed = expected_cost_per_query(res["exact_hit_rate"], res["semantic_hit_rate"], llm, emb)
        assert abs(res["cost"] / len(queries) - closed) < 1e-12

    # semantic tier is bypassed on exact hits and never runs on an empty cache
    c = TieredCache(2, 2, 0.9)
    assert c.lookup("hello", A) == ("miss", None)
    c.insert("hello", A, "H")
    assert c.lookup("HELLO", B) == ("exact", "H")      # exact beats a dissimilar embedding
    assert c.lookup("something else", A) == ("semantic", "H")
    assert c.lookup("something else", B) == ("miss", None)

    # exact tier is LRU
    c = TieredCache(2, 10, 0.99)
    c.insert("a", A, "1")
    c.insert("b", B, "2")
    assert c.lookup("a", np.array([0.6, -0.8]))[0] == "exact"   # refresh a, so b is oldest
    c.insert("c", np.array([-1.0, 0.0]), "3")                    # evicts b
    assert c.lookup("b", np.array([0.6, -0.8]))[0] == "miss"
    assert c.lookup("a", np.array([0.6, -0.8]))[0] == "exact"
    assert c.lookup("c", np.array([0.6, -0.8]))[0] == "exact"

    # semantic tier is FIFO
    c = TieredCache(10, 2, 0.99)
    c.insert("a", A, "1")
    c.insert("b", B, "2")
    c.insert("c", np.array([-1.0, 0.0]), "3")  # drops the entry for A
    assert c.lookup("x", A) == ("miss", None)
    assert c.lookup("y", B) == ("semantic", "2")

    # skewed stream: a bigger cache raises the hit rate and cuts cost
    rng = np.random.default_rng(0)
    centres = [_unit(rng.normal(size=8)) for _ in range(40)]
    ranks = np.arange(1, 41)
    probs = (1 / ranks) / (1 / ranks).sum()
    stream = []
    for i in rng.choice(40, size=600, p=probs):
        v = centres[i] + 0.02 * rng.normal(size=8)
        stream.append((f"q{i} v{rng.integers(0, 3)}", v, f"intent{i}"))
    small = simulate(stream, TieredCache(5, 5, 0.97), llm, emb)
    large = simulate(stream, TieredCache(200, 200, 0.97), llm, emb)
    hit_small = small["exact_hit_rate"] + small["semantic_hit_rate"]
    hit_large = large["exact_hit_rate"] + large["semantic_hit_rate"]
    assert hit_large > hit_small + 0.1
    assert large["cost"] < small["cost"] < small["baseline_cost"]
    assert large["wrong_hit_rate"] < 0.02  # tight threshold on well-separated intents


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
