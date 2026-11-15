"""Exercise 06 - Embedding storage and vector index sizing.

TASK
----
Sizes use decimal units: 1 GB = 1e9 bytes. Implement:

    raw_bytes(n_vectors, dim, bytes_per_dim=4) -> int
        n_vectors * dim * bytes_per_dim (fp32 = 4, fp16 = 2, int8 = 1).
    hnsw_bytes(n_vectors, dim, m=16, bytes_per_dim=4, link_bytes=4) -> int
        Stored vectors plus layer-0 graph links: n_vectors * 2 * m * link_bytes.
        (Upper layers add a few percent, ignored here.)
    pq_bytes(n_vectors, dim, subvectors, bits=8, centroid_bytes=4) -> int
        Product-quantisation codes plus the codebooks.
        codes     = n_vectors * subvectors * bits / 8 (bits must be a multiple of 8)
        codebooks = subvectors * 2**bits * (dim / subvectors) * centroid_bytes
        Raise ValueError if dim is not divisible by subvectors or bits % 8 != 0.
    vectors_for_corpus(documents, chunks_per_document, vectors_per_chunk=1) -> int
        Total vectors to index (documents * chunks * vectors per chunk).
    nodes_needed(index_bytes, ram_per_node_bytes, replicas=2, headroom=0.7) -> tuple[int, int]
        Usable RAM per node = ram_per_node_bytes * headroom. shards = ceil(index_bytes /
        usable). Return (shards, shards * replicas). Round before ceil to 9 decimals.
    monthly_index_cost(total_nodes, price_per_node_hour, hours_per_month=730) -> float
    recall_note(raw, compressed) -> float
        Compression ratio raw / compressed.

HINTS
-----
- Work in plain integers where possible, so 100,000,000 * 1,024 * 4 is exactly 409,600,000,000.
- Replication multiplies cost, not shard count. Replicas give availability and throughput.
- Filtering, metadata and the text itself are not in these numbers. Real sizing adds them.

WHAT YOU SHOULD LEARN
---------------------
- Why 100M vectors at 1,024 dimensions needs ~410 GB raw, and what fp16, int8, PQ and
  lower-dimension (Matryoshka) embeddings each buy you.
- That chunking multiplies the vector count: 10M documents at 8 chunks is 80M vectors.
- That the graph overhead of HNSW is small next to the vectors at high dimension, but
  dominates at low dimension.
"""
from __future__ import annotations

import math


def raw_bytes(n_vectors: int, dim: int, bytes_per_dim: int = 4) -> int:
    return n_vectors * dim * bytes_per_dim


def hnsw_bytes(
    n_vectors: int, dim: int, m: int = 16, bytes_per_dim: int = 4, link_bytes: int = 4
) -> int:
    return raw_bytes(n_vectors, dim, bytes_per_dim) + n_vectors * 2 * m * link_bytes


def pq_bytes(
    n_vectors: int, dim: int, subvectors: int, bits: int = 8, centroid_bytes: int = 4
) -> int:
    if dim % subvectors != 0 or bits % 8 != 0:
        raise ValueError("dim must divide by subvectors and bits by 8")
    codes = n_vectors * subvectors * bits // 8
    codebooks = subvectors * (2 ** bits) * (dim // subvectors) * centroid_bytes
    return codes + codebooks


def vectors_for_corpus(documents: int, chunks_per_document: int, vectors_per_chunk: int = 1) -> int:
    return documents * chunks_per_document * vectors_per_chunk


def nodes_needed(
    index_bytes: float, ram_per_node_bytes: float, replicas: int = 2, headroom: float = 0.7
) -> tuple[int, int]:
    usable = ram_per_node_bytes * headroom
    shards = math.ceil(round(index_bytes / usable, 9))
    return shards, shards * replicas


def monthly_index_cost(
    total_nodes: int, price_per_node_hour: float, hours_per_month: int = 730
) -> float:
    return total_nodes * price_per_node_hour * hours_per_month


def recall_note(raw: float, compressed: float) -> float:
    return raw / compressed


def _run_tests() -> None:
    GB = 1_000_000_000
    n = 100_000_000
    assert raw_bytes(n, 1024) == 409_600_000_000
    assert raw_bytes(n, 1024, 2) == 204_800_000_000
    assert raw_bytes(n, 1024, 1) == 102_400_000_000
    assert raw_bytes(n, 256) == 102_400_000_000       # Matryoshka-style truncation to 256 dims

    # HNSW M=16 adds 128 bytes per vector for layer-0 links
    assert hnsw_bytes(n, 1024) == 409_600_000_000 + 12_800_000_000
    overhead_hi = (hnsw_bytes(n, 1024) - raw_bytes(n, 1024)) / raw_bytes(n, 1024)
    overhead_lo = (hnsw_bytes(n, 64) - raw_bytes(n, 64)) / raw_bytes(n, 64)
    assert abs(overhead_hi - 0.03125) < 1e-9 and abs(overhead_lo - 0.5) < 1e-9
    assert hnsw_bytes(1000, 8, m=32) == 1000 * 8 * 4 + 1000 * 2 * 32 * 4

    # PQ: 64 one-byte codes per vector plus codebooks of 64 * 256 * 16 floats
    pq = pq_bytes(n, 1024, 64)
    assert pq == n * 64 + 64 * 256 * 16 * 4 == 6_400_000_000 + 1_048_576
    assert abs(recall_note(raw_bytes(n, 1024), pq) - 64.0) < 0.02
    assert pq_bytes(10, 8, 4, bits=16) == 10 * 4 * 2 + 4 * 65536 * 2 * 4
    for bad in ((10, 10, 3, 8), (10, 8, 4, 4)):
        try:
            pq_bytes(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # chunking multiplies the vector count
    assert vectors_for_corpus(10_000_000, 8) == 80_000_000
    assert vectors_for_corpus(1_000, 5, 3) == 15_000

    # nodes: 422.4 GB HNSW index on 128 GB nodes with 70% usable RAM
    idx = hnsw_bytes(n, 1024)
    shards, total = nodes_needed(idx, 128 * GB)
    assert idx == 422_400_000_000
    assert (shards, total) == (5, 10)               # 422.4 / 89.6 = 4.71 -> 5 shards
    # int8 halves again: 102.4 GB + links on the same nodes
    shards8, total8 = nodes_needed(hnsw_bytes(n, 1024, bytes_per_dim=1), 128 * GB)
    assert (shards8, total8) == (2, 4)
    # exact fit does not round up through float noise
    assert nodes_needed(70.0, 100.0, replicas=1, headroom=0.7) == (1, 1)
    assert nodes_needed(70.1, 100.0, replicas=3, headroom=0.7) == (2, 6)

    # money: 10 nodes at $4 per hour
    assert abs(monthly_index_cost(10, 4.0) - 29_200.0) < 1e-6
    assert abs(monthly_index_cost(4, 4.0) - 11_680.0) < 1e-6
    assert monthly_index_cost(0, 4.0) == 0.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
