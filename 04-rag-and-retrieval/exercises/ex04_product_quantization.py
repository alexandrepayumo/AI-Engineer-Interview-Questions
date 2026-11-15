"""Exercise 04 - Product quantisation: encode, decode, asymmetric distance, memory math.

TASK
----
Implement:

    pq_memory_bytes(n, d, m, nbits=8) -> dict
        Memory for n vectors of dimension d. Return a dict with keys
        "float32_bytes" (n*d*4), "pq_code_bytes" (n*m*nbits/8, assume nbits divisible by 8),
        "codebook_bytes" (m * 2**nbits * (d//m) * 4) and "ratio" (float32_bytes divided by
        pq_code_bytes).

    class ProductQuantizer(m, nbits=4, seed=0)
        Splits a d-dim vector into m contiguous sub-vectors (d must be divisible by m) and
        learns 2**nbits centroids per sub-space.
        .fit(x, iters=10)       learn codebooks with k-means per sub-space (a small k-means
                                helper is provided).
        .encode(x) -> np.ndarray   (n, m) uint8 codes: nearest centroid id per sub-space.
        .decode(codes) -> np.ndarray   (n, d) reconstruction: concatenated centroids.
        .adc_table(q) -> np.ndarray    (m, 2**nbits): table[j, c] = squared L2 distance between
                                sub-vector j of the (un-quantised) query and centroid c of
                                sub-space j.
        .adc_distances(table, codes) -> np.ndarray   (n,): sum over j of table[j, codes[:, j]].

HINTS
-----
- Asymmetric distance computation (ADC) keeps the query exact and quantises only the
  database. Building the table costs m * 2**nbits sub-distances once per query; scoring each
  database vector then costs m table lookups and adds, with no d-dimensional arithmetic.
- ADC distance must equal ||q - decode(code)||^2 exactly (up to float error).
- Fancy indexing: table[np.arange(m), codes] has shape (n, m) when codes is (n, m)
  and the arange is broadcast as table[np.arange(m)[None, :], codes].

WHAT YOU SHOULD LEARN
---------------------
- Where 32x-ish memory savings come from, and what the codebooks cost.
- Why PQ is lossy: nearby vectors can share a code, so you rerank candidates with exact vectors.
- Why scoring is fast: lookups into a per-query table instead of d multiply-adds per vector.
"""
from __future__ import annotations

import numpy as np


def _kmeans(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """Small Lloyd k-means helper (provided)."""
    n = x.shape[0]
    cent = x[rng.choice(n, k, replace=False)].astype(np.float64)
    for _ in range(iters):
        d = (x * x).sum(1)[:, None] - 2 * x @ cent.T + (cent * cent).sum(1)[None, :]
        assign = d.argmin(1)
        for c in range(k):
            mem = x[assign == c]
            cent[c] = mem.mean(0) if len(mem) else x[rng.integers(n)]
    return cent


def pq_memory_bytes(n: int, d: int, m: int, nbits: int = 8) -> dict:
    raise NotImplementedError("TODO")


class ProductQuantizer:
    def __init__(self, m: int, nbits: int = 4, seed: int = 0) -> None:
        self.m = m
        self.nbits = nbits
        self.ksub = 2 ** nbits
        self.seed = seed
        self.dsub = 0
        self.codebooks: np.ndarray | None = None  # (m, ksub, dsub)

    def fit(self, x: np.ndarray, iters: int = 10) -> None:
        raise NotImplementedError("TODO")

    def encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def decode(self, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def adc_table(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def adc_distances(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")


def _run_tests() -> None:
    # memory math: 1M vectors, d=768, m=96 sub-vectors, 8 bits each
    mem = pq_memory_bytes(1_000_000, 768, 96, 8)
    assert mem["float32_bytes"] == 3_072_000_000
    assert mem["pq_code_bytes"] == 96_000_000
    assert mem["codebook_bytes"] == 96 * 256 * 8 * 4  # 786,432 bytes, negligible
    assert mem["ratio"] == 32.0
    assert pq_memory_bytes(10, 64, 8, 8)["ratio"] == 32.0

    rng = np.random.default_rng(7)
    # structured data: 32 dims, low-rank signal plus noise, 3000 vectors
    basis = rng.normal(size=(6, 32))
    x = rng.normal(size=(3000, 6)) @ basis + 0.1 * rng.normal(size=(3000, 32))
    pq = ProductQuantizer(m=8, nbits=4, seed=0)
    pq.fit(x[:2000])
    assert pq.codebooks.shape == (8, 16, 4)

    codes = pq.encode(x)
    assert codes.shape == (3000, 8) and codes.dtype == np.uint8 and codes.max() < 16
    rec = pq.decode(codes)
    assert rec.shape == x.shape
    err = ((x - rec) ** 2).sum(1).mean()
    var = ((x - x.mean(0)) ** 2).sum(1).mean()
    assert err < 0.5 * var  # quantisation removes a good chunk of variance

    # decode(encode(.)) is a fixed point of encode
    assert np.array_equal(pq.encode(rec), codes)

    # ADC equals the exact distance to the reconstruction
    q = rng.normal(size=(6,)) @ basis + 0.1 * rng.normal(size=32)
    table = pq.adc_table(q)
    assert table.shape == (8, 16)
    adc = pq.adc_distances(table, codes)
    direct = ((rec - q) ** 2).sum(1)
    assert np.allclose(adc, direct)

    # Retrieval quality: PQ top-50 candidates contain most of the exact top-10 (the reason
    # we shortlist with PQ and then rerank with exact vectors)
    exact_top = np.argsort(((x - q) ** 2).sum(1))[:10]
    pq_top = np.argsort(adc)[:50]
    overlap = len(set(exact_top.tolist()) & set(pq_top.tolist())) / 10
    assert overlap >= 0.7
    # and reranking those 50 candidates exactly recovers at least as many
    rer = pq_top[np.argsort(((x[pq_top] - q) ** 2).sum(1))[:10]]
    assert len(set(exact_top.tolist()) & set(rer.tolist())) / 10 >= overlap - 0.31


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
