"""Exercise 03 - Few-shot example selection with MMR.

TASK
----
Pick k few-shot examples that are relevant to the query but not redundant with
each other, using Maximal Marginal Relevance over bag-of-words vectors.

    def bow_vectors(texts: list[str]) -> tuple[np.ndarray, list[str]]
        '''Lowercase, tokenise with re.findall(r"[a-z0-9]+", text.lower()).
        Return (matrix, vocab). vocab is sorted alphabetically; matrix has shape
        (len(texts), len(vocab)) with raw counts, each row L2-normalised
        (an all-zero row stays all zeros).'''

    def mmr_select(query: np.ndarray, cands: np.ndarray, k: int, lam: float) -> list[int]
        '''Greedy MMR. Vectors are already L2-normalised so a dot product is cosine.
        score(i) = lam * sim(query, cands[i]) - (1 - lam) * max_{j in selected} sim(cands[i], cands[j])
        (the redundancy term is 0 while nothing is selected). Pick the highest score
        each round, ties to the lowest index. Return indices in selection order.
        k larger than the number of candidates returns all of them.'''

    def select_examples(query: str, pool: list[str], k: int, lam: float) -> list[int]
        '''Vectorise query + pool together, run mmr_select, then reorder the chosen
        indices so the MOST similar example to the query comes LAST (recency bias:
        the final demonstration pulls hardest). Others stay in ascending similarity.'''

HINTS
-----
- Build the vocab from query + pool in select_examples so the spaces match.
- sims = cands @ query gives all query similarities at once; cands @ cands.T the pairwise ones.
- lam = 1 ignores diversity (plain top-k). lam = 0 ignores relevance.

WHAT YOU SHOULD LEARN
---------------------
Pure nearest-neighbour selection tends to return near-duplicates, which wastes
context and over-represents one label. MMR trades a little relevance for
coverage with one knob. The final ordering step shows that selection and ordering
are separate decisions.
"""
from __future__ import annotations

import re

import numpy as np


def bow_vectors(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    toks = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
    vocab = sorted({w for ts in toks for w in ts})
    index = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(texts), len(vocab)))
    for r, ts in enumerate(toks):
        for w in ts:
            m[r, index[w]] += 1.0
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # keep zero rows as zeros
    return m / norms, vocab


def mmr_select(query: np.ndarray, cands: np.ndarray, k: int, lam: float) -> list[int]:
    n = cands.shape[0]
    k = min(k, n)
    rel = cands @ query          # relevance of each candidate
    pair = cands @ cands.T       # pairwise similarity
    selected: list[int] = []
    for _ in range(k):
        best_i, best_score = -1, -np.inf
        for i in range(n):
            if i in selected:
                continue
            red = max(pair[i, j] for j in selected) if selected else 0.0
            score = lam * rel[i] - (1 - lam) * red
            if score > best_score + 1e-12:  # strict improvement keeps the lowest index on ties
                best_i, best_score = i, score
        selected.append(best_i)
    return selected


def select_examples(query: str, pool: list[str], k: int, lam: float) -> list[int]:
    mat, _ = bow_vectors([query] + pool)
    q, cands = mat[0], mat[1:]
    chosen = mmr_select(q, cands, k, lam)
    sims = cands @ q
    # ascending similarity, most similar last; stable on ties by selection order
    return sorted(chosen, key=lambda i: sims[i])


def _run_tests() -> None:
    m, vocab = bow_vectors(["Red red blue", "blue green", ""])
    assert vocab == ["blue", "green", "red"]
    assert np.allclose(m[0], np.array([1, 0, 2]) / np.sqrt(5))
    assert np.allclose(np.linalg.norm(m[1]), 1.0)
    assert np.allclose(m[2], 0.0)

    q = np.array([1.0, 0.0])
    a = np.array([0.9, np.sqrt(1 - 0.81)])
    b = np.array([0.89, np.sqrt(1 - 0.89 ** 2)])   # near-duplicate of a
    c = np.array([0.8, -0.6])
    cands = np.stack([a, b, c])
    assert mmr_select(q, cands, 2, 1.0) == [0, 1]    # lam=1 -> plain top-k
    assert mmr_select(q, cands, 2, 0.5) == [0, 2]    # diversity prefers c over the duplicate b
    assert mmr_select(q, cands, 10, 0.5) == [0, 2, 1]
    assert mmr_select(q, cands, 0, 0.5) == []

    # hand check of the second-round scores at lam=0.5: b -> 0.5*0.89 - 0.5*(a.b), c -> 0.5*0.8 - 0.5*(a.c)
    sc_b = 0.5 * 0.89 - 0.5 * float(a @ b)
    sc_c = 0.5 * 0.8 - 0.5 * float(a @ c)
    assert sc_c > sc_b and abs(sc_b - (-0.0549)) < 1e-3 and abs(sc_c - 0.1709) < 1e-3

    # ties go to the lowest index
    same = np.stack([q, q, q])
    assert mmr_select(q, same, 2, 1.0) == [0, 1]

    pool = [
        "refund my order",                 # 0
        "refund my order please",          # 1 near-duplicate of 0
        "how do I reset my password",      # 2
        "cancel my subscription today",    # 3
    ]
    query = "I want a refund for my order"
    top = select_examples(query, pool, 2, 1.0)
    assert set(top) == {0, 1}
    diverse = select_examples(query, pool, 2, 0.3)
    assert len(set(diverse) & {0, 1}) == 1       # only one of the duplicates survives
    # most similar example is last
    mat, _ = bow_vectors([query] + pool)
    sims = mat[1:] @ mat[0]
    assert sims[diverse[-1]] == max(sims[i] for i in diverse)
    # all candidates returned when k is large
    assert sorted(select_examples(query, pool, 10, 0.5)) == [0, 1, 2, 3]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
