"""Exercise 03 - IVF index: k-means partitioning and nprobe vs recall.

TASK
----
Implement:

    kmeans(x, k, iters, seed) -> np.ndarray
        Lloyd's algorithm on (n, d) float data with squared-L2 distance. Initialise by
        choosing k distinct rows with np.random.default_rng(seed).choice(n, k, replace=False).
        An empty cluster must be re-seeded with a random data row (use the same rng).
        Return centroids of shape (k, d).

    exact_search(x, q, k) -> np.ndarray
        Ids (row numbers) of the k nearest rows of x to q by squared L2, nearest first.

    class IVFIndex(nlist, seed=0)
        .fit_add(x)            train centroids with kmeans(x, nlist, 15, seed), then assign
                               every vector to its nearest centroid and store inverted lists
                               (one array of row ids per cell).
        .search(q, k, nprobe) -> tuple[np.ndarray, int]
                               find the nprobe nearest centroids to q, scan only the vectors
                               in those cells, return (top-k ids nearest first, number of
                               vectors scanned). If fewer than k candidates exist, return
                               what you have.

    recall_vs_exact(index, x, queries, k, nprobe) -> tuple[float, float]
        Mean recall@k against exact_search, and mean fraction of the database scanned.

HINTS
-----
- Squared distance: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2. Use a matrix multiply, not loops.
- Probing centroids in order of distance means probe sets are nested: nprobe=3 includes
  everything nprobe=2 scanned. That is why recall cannot decrease as nprobe grows.
- Ties are unlikely with random floats; use np.argsort on the candidate distances.

WHAT YOU SHOULD LEARN
---------------------
- IVF trades recall for speed through one knob: nprobe. nprobe = nlist is exact search.
- Cell-boundary misses: a true neighbour can sit in a cell whose centroid is not among
  the closest nprobe to the query.
- Scanned fraction is roughly nprobe / nlist when cells are balanced.
"""
from __future__ import annotations

import numpy as np


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances between rows of a (m, d) and rows of b (n, d)."""
    return (a * a).sum(1)[:, None] - 2.0 * a @ b.T + (b * b).sum(1)[None, :]


def kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    cent = x[rng.choice(n, k, replace=False)].astype(np.float64)
    for _ in range(iters):
        assign = _sqdist(x, cent).argmin(axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members) == 0:
                cent[c] = x[rng.integers(n)]  # re-seed empty cluster
            else:
                cent[c] = members.mean(axis=0)
    return cent


def exact_search(x: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    d = _sqdist(q[None, :], x)[0]
    return np.argsort(d)[:k]


class IVFIndex:
    def __init__(self, nlist: int, seed: int = 0) -> None:
        self.nlist = nlist
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.lists: list[np.ndarray] = []
        self.x: np.ndarray | None = None

    def fit_add(self, x: np.ndarray) -> None:
        self.x = x
        self.centroids = kmeans(x, self.nlist, 15, self.seed)
        assign = _sqdist(x, self.centroids).argmin(axis=1)
        self.lists = [np.where(assign == c)[0] for c in range(self.nlist)]

    def search(self, q: np.ndarray, k: int, nprobe: int) -> tuple[np.ndarray, int]:
        cd = _sqdist(q[None, :], self.centroids)[0]
        cells = np.argsort(cd)[:nprobe]  # nearest centroids first -> nested probe sets
        cand = np.concatenate([self.lists[c] for c in cells])
        if len(cand) == 0:
            return cand, 0
        d = _sqdist(q[None, :], self.x[cand])[0]
        order = np.argsort(d)[:k]
        return cand[order], int(len(cand))


def recall_vs_exact(
    index: IVFIndex, x: np.ndarray, queries: np.ndarray, k: int, nprobe: int
) -> tuple[float, float]:
    recalls, fracs = [], []
    for q in queries:
        truth = set(exact_search(x, q, k).tolist())
        ids, scanned = index.search(q, k, nprobe)
        recalls.append(len(truth & set(ids.tolist())) / k)
        fracs.append(scanned / len(x))
    return float(np.mean(recalls)), float(np.mean(fracs))


def _run_tests() -> None:
    rng = np.random.default_rng(42)
    # 16 well-separated-ish Gaussian blobs in 16 dims, 3200 points
    centers = rng.normal(scale=3.0, size=(16, 16))
    x = np.vstack([c + rng.normal(scale=1.5, size=(200, 16)) for c in centers])
    queries = np.vstack([c + rng.normal(scale=1.5, size=(5, 16)) for c in centers])

    # kmeans basics
    cent = kmeans(x, 16, 10, seed=1)
    assert cent.shape == (16, 16)
    assert np.allclose(cent, kmeans(x, 16, 10, seed=1))  # deterministic
    # inertia must beat a single global mean by a wide margin
    inertia = _sqdist(x, cent).min(axis=1).sum()
    single = ((x - x.mean(0)) ** 2).sum()
    assert inertia < 0.5 * single

    # exact search
    ids = exact_search(x, x[7], 5)
    assert ids[0] == 7 and len(ids) == 5
    dists = ((x[ids] - x[7]) ** 2).sum(1)
    assert np.all(np.diff(dists) >= 0)

    # index structure
    idx = IVFIndex(nlist=16, seed=0)
    idx.fit_add(x)
    sizes = np.array([len(l) for l in idx.lists])
    assert sizes.sum() == len(x)  # every vector in exactly one cell
    assert len(set(np.concatenate(idx.lists).tolist())) == len(x)

    # nprobe = nlist is exact search
    r_full, frac_full = recall_vs_exact(idx, x, queries, 10, nprobe=16)
    assert r_full == 1.0 and abs(frac_full - 1.0) < 1e-12

    # recall is non-decreasing in nprobe (nested probe sets) and scan cost grows
    recalls, fracs = [], []
    for nprobe in (1, 2, 4, 8, 16):
        r, f = recall_vs_exact(idx, x, queries, 10, nprobe)
        recalls.append(r)
        fracs.append(f)
    assert all(b >= a - 1e-12 for a, b in zip(recalls, recalls[1:]))
    assert all(b > a for a, b in zip(fracs, fracs[1:]))
    assert recalls[0] < 1.0  # one cell is not enough for overlapping blobs
    assert fracs[0] < 0.2  # and it scanned far less than the whole database
    # the self-query always finds itself when its own cell is probed
    found, _ = idx.search(x[100], 1, nprobe=1)
    assert found[0] == 100


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
