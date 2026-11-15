"""Exercise 07 - A mini HNSW: layered greedy graph search measured against brute force.

TASK
----
Implement a small Hierarchical Navigable Small World index (squared L2 distance).

    class MiniHNSW(M=8, ef_construction=64, seed=0)
        .add(vec) -> int
            Insert one vector, return its id (0, 1, 2, ...). Steps:
            1. Draw a level: level = int(-log(U) * mL) with mL = 1 / log(M), U uniform in (0, 1]
               from the index's own rng (np.random.default_rng(seed)).
            2. The first vector becomes the entry point. Otherwise greedy-descend (ef = 1)
               from the entry point's top layer down to layer level + 1.
            3. For each layer from min(level, top_layer) down to 0: run
               _search_layer(vec, entry_points, ef_construction, layer), connect the new node
               to its M nearest candidates (M0 = 2*M on layer 0), add the reverse edges,
               and if a neighbour now has more than its maximum degree, keep only its closest
               links. Use the whole candidate list as entry points for the next layer down.
            4. If level is above the current top layer, the new node becomes the entry point.
        ._search_layer(q, entries, ef, layer) -> list[tuple[float, int]]
            Beam search inside one layer. Returns up to ef (distance, id) pairs, nearest first.
            Maintain a candidate min-heap and a result max-heap; stop when the closest
            unexpanded candidate is farther than the worst result (and the result is full).
        .search(q, k, ef) -> list[int]
            Greedy descent (ef = 1) through the upper layers, then _search_layer on layer 0
            with ef = max(ef, k). Return the ids of the k nearest found.
        .dist_calls: int
            Counter incremented once for every distance evaluated (use it to compare cost).

    brute_force(x, q, k) -> list[int]
        Exact k nearest ids.

HINTS
-----
- graph[layer][node] is a list of neighbour ids. Upper layers only contain nodes whose level
  is at least that layer.
- heapq is a min-heap. For the result max-heap push (-distance, id).
- Keep a visited set per _search_layer call.
- Increase ef at query time to buy recall with latency. That is the main HNSW tuning knob.

WHAT YOU SHOULD LEARN
---------------------
- Why a handful of long-range links in sparse upper layers makes greedy search logarithmic.
- Recall depends on ef, and cost (distance evaluations) grows with ef.
- Why searching with ef = 1 usually gets stuck in a local minimum.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def brute_force(x: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    d = ((x - q) ** 2).sum(1)
    return np.argsort(d, kind="stable")[:k].tolist()


class MiniHNSW:
    def __init__(self, M: int = 8, ef_construction: int = 64, seed: int = 0) -> None:
        self.M = M
        self.M0 = 2 * M
        self.efc = ef_construction
        self.mL = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.vecs: list[np.ndarray] = []
        self.graph: list[dict[int, list[int]]] = []  # graph[layer][node] -> neighbours
        self.levels: list[int] = []
        self.entry: int | None = None
        self.dist_calls = 0

    def _dist(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        self.dist_calls += len(ids)
        return ((np.stack([self.vecs[i] for i in ids]) - q) ** 2).sum(1)

    def _search_layer(
        self, q: np.ndarray, entries: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        visited = set(entries)
        d0 = self._dist(q, entries)
        cand = [(float(d), i) for d, i in zip(d0, entries)]  # min-heap of frontier
        heapq.heapify(cand)
        res = [(-d, i) for d, i in cand]  # max-heap (negated) of best ef found
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        while cand:
            d, c = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break  # closest frontier node is worse than our worst result
            nbrs = [n for n in self.graph[layer].get(c, []) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for nd, n in zip(self._dist(q, nbrs), nbrs):
                nd = float(nd)
                if len(res) < ef or nd < -res[0][0]:
                    heapq.heappush(cand, (nd, n))
                    heapq.heappush(res, (-nd, n))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-nd, i) for nd, i in res)

    def _shrink(self, node: int, layer: int, max_deg: int) -> None:
        nb = self.graph[layer][node]
        if len(nb) > max_deg:
            d = self._dist(self.vecs[node], nb)
            keep = np.argsort(d, kind="stable")[:max_deg]
            self.graph[layer][node] = [nb[j] for j in keep]

    def add(self, vec: np.ndarray) -> int:
        nid = len(self.vecs)
        self.vecs.append(np.asarray(vec, dtype=np.float64))
        level = int(-math.log(1.0 - self.rng.random()) * self.mL)  # 1-U is in (0, 1]
        self.levels.append(level)
        while len(self.graph) <= level:
            self.graph.append({})
        for layer in range(level + 1):
            self.graph[layer][nid] = []
        if self.entry is None:
            self.entry = nid
            return nid
        top = self.levels[self.entry]
        ep = [self.entry]
        for layer in range(top, level, -1):  # greedy descent above the node's level
            ep = [self._search_layer(self.vecs[nid], ep, 1, layer)[0][1]]
        for layer in range(min(level, top), -1, -1):
            found = self._search_layer(self.vecs[nid], ep, self.efc, layer)
            max_deg = self.M0 if layer == 0 else self.M
            chosen = [i for _, i in found[: self.M]]
            self.graph[layer][nid] = list(chosen)
            for n in chosen:  # reverse edges, prune overfull neighbours
                self.graph[layer][n].append(nid)
                self._shrink(n, layer, max_deg)
            ep = [i for _, i in found]
        if level > top:
            self.entry = nid
        return nid

    def search(self, q: np.ndarray, k: int, ef: int) -> list[int]:
        ep = [self.entry]
        for layer in range(self.levels[self.entry], 0, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]
        found = self._search_layer(q, ep, max(ef, k), 0)
        return [i for _, i in found[:k]]


def _recall(index: MiniHNSW, x: np.ndarray, queries: np.ndarray, k: int, ef: int) -> float:
    hits = 0
    for q in queries:
        hits += len(set(index.search(q, k, ef)) & set(brute_force(x, q, k)))
    return hits / (k * len(queries))


def _run_tests() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(800, 12))
    queries = rng.normal(size=(40, 12))

    idx = MiniHNSW(M=8, ef_construction=64, seed=0)
    for i, v in enumerate(x):
        assert idx.add(v) == i
    assert len(idx.vecs) == 800

    # structural invariants
    for layer, g in enumerate(idx.graph):
        cap = idx.M0 if layer == 0 else idx.M
        for node, nbrs in g.items():
            assert len(nbrs) <= cap
            assert node not in nbrs
            assert idx.levels[node] >= layer  # upper layers hold only high-level nodes
            assert all(n in g for n in nbrs)  # links stay inside the layer
    assert len(idx.graph[0]) == 800
    assert len(idx.graph) >= 2 and len(idx.graph[1]) < len(idx.graph[0]) / 2
    assert idx.levels[idx.entry] == len(idx.graph) - 1

    # brute force sanity
    assert brute_force(x, x[5], 1) == [5]

    # a stored vector finds itself
    assert idx.search(x[123], 1, ef=16) == [123]

    # recall rises with ef, and is high at ef=64
    r_small = _recall(idx, x, queries, 10, 10)
    r_mid = _recall(idx, x, queries, 10, 40)
    r_big = _recall(idx, x, queries, 10, 200)
    assert r_small <= r_mid + 1e-9 and r_mid <= r_big + 1e-9
    assert r_mid >= 0.85, r_mid
    assert r_big >= 0.97, r_big

    # cost: a moderate-ef query touches far fewer vectors than brute force
    idx.dist_calls = 0
    for q in queries:
        idx.search(q, 10, 40)
    per_query = idx.dist_calls / len(queries)
    assert per_query < 0.6 * len(x), per_query
    # and cost grows with ef
    idx.dist_calls = 0
    for q in queries:
        idx.search(q, 10, 10)
    assert idx.dist_calls / len(queries) < per_query

    # results come back nearest first
    out = idx.search(queries[0], 10, 64)
    d = [((x[i] - queries[0]) ** 2).sum() for i in out]
    assert d == sorted(d)

    # determinism: same seed -> same graph
    idx2 = MiniHNSW(M=8, ef_construction=64, seed=0)
    for v in x:
        idx2.add(v)
    assert idx2.levels == idx.levels and idx2.graph[0] == idx.graph[0]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
