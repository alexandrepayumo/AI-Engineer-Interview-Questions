"""Exercise 05 - ColBERT-style late interaction (MaxSim) scoring.

TASK
----
Implement:

    maxsim_score(Q, D) -> float
        Q: (nq, d) query token embeddings, D: (nd, d) document token embeddings
        (rows are assumed L2-normalised). Score = sum over query tokens i of
        max over doc tokens j of Q[i] . D[j].

    rank_docs(Q, docs) -> list[int]
        docs is a list of (nd_i, d) arrays (different lengths). Return doc indices sorted
        by MaxSim score, best first. Ties broken by lower index.

    mean_pool_score(Q, D) -> float
        The single-vector baseline: cosine between mean(Q) and mean(D).

    token_contributions(Q, D) -> np.ndarray
        (nq,) array: for each query token, its best match score in D. This is what you
        inspect to see WHY a document matched.

    storage_bytes(n_docs, avg_tokens, dim, bytes_per_value=2) -> tuple[int, int]
        (multi_vector_bytes, single_vector_bytes) for storing every token vs one vector
        per document.

HINTS
-----
- One matrix product S = Q @ D.T has shape (nq, nd); MaxSim is S.max(axis=1).sum().
- MaxSim is asymmetric: the sum runs over QUERY tokens only.
- Document length does not enter the formula: padding a document with irrelevant tokens
  cannot lower its score (it can only add candidates for the max).

WHAT YOU SHOULD LEARN
---------------------
- Late interaction keeps token-level matching while still allowing pre-computed document
  vectors.
- Where a single pooled vector loses information that MaxSim keeps (and vice versa).
- The storage cost: roughly avg_tokens times more than one vector per document.
"""
from __future__ import annotations

import numpy as np


def maxsim_score(Q: np.ndarray, D: np.ndarray) -> float:
    S = Q @ D.T  # (nq, nd) all token-token similarities
    return float(S.max(axis=1).sum())  # best doc token per query token, summed


def rank_docs(Q: np.ndarray, docs: list[np.ndarray]) -> list[int]:
    scores = [maxsim_score(Q, D) for D in docs]
    return sorted(range(len(docs)), key=lambda i: (-scores[i], i))


def mean_pool_score(Q: np.ndarray, D: np.ndarray) -> float:
    a, b = Q.mean(axis=0), D.mean(axis=0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def token_contributions(Q: np.ndarray, D: np.ndarray) -> np.ndarray:
    return (Q @ D.T).max(axis=1)


def storage_bytes(n_docs: int, avg_tokens: int, dim: int, bytes_per_value: int = 2) -> tuple[int, int]:
    multi = n_docs * avg_tokens * dim * bytes_per_value
    single = n_docs * dim * bytes_per_value
    return multi, single


def _run_tests() -> None:
    e = np.eye(8)
    s2 = 1 / np.sqrt(2)

    Q = np.stack([e[0], e[1]])  # query tokens: e0, e1

    # Doc A has exactly the two query tokens plus two unrelated ones.
    A = np.stack([e[0], e[1], e[5], e[6]])
    # Doc B has four copies of the blend (e0 + e1)/sqrt(2).
    B = np.stack([(e[0] + e[1]) * s2] * 4)

    assert abs(maxsim_score(Q, A) - 2.0) < 1e-12
    assert abs(maxsim_score(Q, B) - 2 * s2) < 1e-12  # 1.4142...
    assert rank_docs(Q, [B, A]) == [1, 0]  # MaxSim prefers the doc with exact token matches

    # The pooled baseline gets it the other way round: B's mean equals the query mean.
    assert abs(mean_pool_score(Q, B) - 1.0) < 1e-12
    assert mean_pool_score(Q, A) < mean_pool_score(Q, B)
    # cosine of (0.5, 0.5, 0, ...) with 0.25 on e0, e1, e5, e6 = 0.25 / (0.7071 * 0.5)
    assert abs(mean_pool_score(Q, A) - 0.25 / (np.sqrt(0.5) * 0.5)) < 1e-12

    # Per-token explanation
    assert np.allclose(token_contributions(Q, A), [1.0, 1.0])
    assert np.allclose(token_contributions(Q, B), [s2, s2])

    # Padding a document with irrelevant tokens never lowers its score
    padded = np.vstack([A, np.stack([e[7]] * 10)])
    assert maxsim_score(Q, padded) >= maxsim_score(Q, A)
    assert abs(maxsim_score(Q, padded) - maxsim_score(Q, A)) < 1e-12

    # Invariant to document token order
    perm = np.random.default_rng(0).permutation(len(A))
    assert abs(maxsim_score(Q, A[perm]) - maxsim_score(Q, A)) < 1e-12

    # Asymmetry: swapping roles changes the score (sum is over the first argument's tokens)
    assert abs(maxsim_score(A, Q) - 2.0) < 1e-12  # tokens e0,e1 match; e5,e6 contribute 0
    assert abs(maxsim_score(A, Q) - maxsim_score(Q, A)) < 1e-12  # equal here by construction
    assert maxsim_score(A, B) != maxsim_score(B, A)

    # Ties break by index
    assert rank_docs(Q, [A, A.copy(), B]) == [0, 1, 2]

    # Random sanity: score bounded by number of query tokens for unit vectors
    rng = np.random.default_rng(1)
    Qr = rng.normal(size=(5, 16))
    Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
    Dr = rng.normal(size=(30, 16))
    Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)
    assert maxsim_score(Qr, Dr) <= 5 + 1e-9

    # Storage: 1M docs, 200 tokens, dim 128, fp16
    multi, single = storage_bytes(1_000_000, 200, 128, 2)
    assert multi == 51_200_000_000 and single == 256_000_000
    assert multi // single == 200


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
