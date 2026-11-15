"""Exercise 01 - Broadcasting and einsum drills: replace loops with array code.

TASK
----
Implement each function below WITHOUT Python loops over data (no `for` over rows,
batches, heads or positions). Every function has a slow loop reference defined
just above the tests, and the tests check your output against it.

    pairwise_sq_dists(x, y) -> (n, m)
        x is (n, d), y is (m, d). Entry [i, j] = sum_k (x[i, k] - y[j, k]) ** 2.
        Use |x|^2 + |y|^2 - 2 x.y so no (n, m, d) temporary is built. Clip at 0.
    split_heads(x, n_heads) -> (B, h, T, dh)
        x is (B, T, D) with D = h * dh. Head i owns channels [i*dh, (i+1)*dh).
    merge_heads(x) -> (B, T, D)
        The inverse of split_heads.
    attention_scores(q, k) -> (B, h, Tq, Tk)
        q is (B, h, Tq, dh), k is (B, h, Tk, dh). Return q . k / sqrt(dh) using einsum.
    outer_sum(a, b) -> (n, m)
        a is (B, n), b is (B, m). Return sum over the batch of outer(a[b], b[b]).
    batched_bilinear(x, w, y) -> (B,)
        x is (B, i), w is (i, j), y is (B, j). Entry b = x[b] @ w @ y[b], with ONE einsum.
    cosine_matrix(x, y, eps=1e-12) -> (n, m)
        Cosine similarity between every row of x and every row of y. A zero row must
        give similarity 0, not NaN.
    gather_logprobs(logp, ids) -> (B, T)
        logp is (B, T, V), ids is (B, T) ints. Entry [b, t] = logp[b, t, ids[b, t]].
    masked_mean(x, lengths) -> (B, D)
        x is (B, T, D), lengths is (B,). Mean over the first lengths[b] positions of
        each sequence, ignoring padding. A length of 0 returns a zero vector.
    one_hot(ids, v) -> ids.shape + (v,) float array, built with broadcasting.

HINTS
-----
- Broadcasting aligns shapes from the RIGHT. Insert axes with [:, None] or [..., None].
- reshape splits or merges axes in memory order, so it can only split the LAST axis
  cleanly. To get heads in front of time, reshape first and then transpose.
- einsum subscripts: letters repeated across operands are multiplied, letters missing
  from the output are summed. 'bhqd,bhkd->bhqk' sums over d.
- np.take_along_axis(a, idx, axis) needs idx with the same number of dimensions as a.

WHAT YOU SHOULD LEARN
---------------------
- Reading and writing shapes before writing code, which is most of numpy fluency.
- Why the head split is reshape then transpose, and why the merge must transpose first.
- The three standard ways to select per-row values: fancy indexing, take_along_axis,
  one-hot multiplication. Cross-entropy implementations use the first two.
- Guarding degenerate inputs (zero vectors, empty sequences) without branching.
"""
from __future__ import annotations

import math

import numpy as np


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, so one matmul and two row-norm vectors.
    d2 = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * (x @ y.T)
    # Rounding can leave tiny negatives where the true distance is 0.
    return np.maximum(d2, 0.0)


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    b, t, d = x.shape
    # Split the channel axis into (heads, head_dim), THEN move heads in front of time.
    return x.reshape(b, t, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    b, h, t, dh = x.shape
    # Undo the transpose first so each position's heads are adjacent, then flatten.
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)


def attention_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    dh = q.shape[-1]
    # d is repeated and absent from the output, so it is summed.
    return np.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(dh)


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Batch index repeated and dropped: the sum over the batch is the contraction.
    return np.einsum("bi,bj->ij", a, b)


def batched_bilinear(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", x, w, y)


def cosine_matrix(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Clamp the norm, not the result: a zero row becomes 0 / eps = 0, so similarity is 0.
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), eps)
    return xn @ yn.T


def gather_logprobs(logp: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # take_along_axis needs ids with a trailing axis; drop it again afterwards.
    return np.take_along_axis(logp, ids[..., None], axis=-1)[..., 0]


def masked_mean(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    t = x.shape[1]
    mask = np.arange(t)[None, :] < lengths[:, None]          # (B, T) bool
    total = (x * mask[..., None]).sum(axis=1)                # padding contributes 0
    return total / np.maximum(lengths, 1)[:, None]           # length 0 -> zero vector


def one_hot(ids: np.ndarray, v: int) -> np.ndarray:
    # (..., 1) == (v,) broadcasts to (..., v).
    return (ids[..., None] == np.arange(v)).astype(np.float64)


# ----------------------------------------------------------------- loop references

def _loop_sq_dists(x, y):
    out = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            out[i, j] = sum((x[i, k] - y[j, k]) ** 2 for k in range(x.shape[1]))
    return out


def _loop_scores(q, k):
    b, h, tq, dh = q.shape
    tk = k.shape[2]
    out = np.zeros((b, h, tq, tk))
    for bi in range(b):
        for hi in range(h):
            for i in range(tq):
                for j in range(tk):
                    out[bi, hi, i, j] = np.dot(q[bi, hi, i], k[bi, hi, j]) / math.sqrt(dh)
    return out


def _loop_masked_mean(x, lengths):
    out = np.zeros((x.shape[0], x.shape[2]))
    for bi in range(x.shape[0]):
        n = int(lengths[bi])
        if n > 0:
            out[bi] = x[bi, :n].mean(axis=0)
    return out


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # pairwise distances
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    assert np.allclose(pairwise_sq_dists(x, y), _loop_sq_dists(x, y), atol=1e-9)
    d_self = pairwise_sq_dists(x, x)
    assert np.all(d_self >= 0.0) and np.allclose(np.diag(d_self), 0.0, atol=1e-9)
    big = rng.normal(size=(6, 4)) * 1e3 + 1e6  # large offset stresses cancellation
    assert np.all(pairwise_sq_dists(big, big) >= 0.0)

    # heads
    z = rng.normal(size=(2, 5, 8))
    s = split_heads(z, 2)
    assert s.shape == (2, 2, 5, 4)
    assert np.array_equal(s[:, 0], z[:, :, 0:4]) and np.array_equal(s[:, 1], z[:, :, 4:8])
    assert np.array_equal(merge_heads(s), z)
    assert not np.array_equal(z.reshape(2, 2, 5, 4), s), "reshape alone mixes time and heads"
    z3 = rng.normal(size=(3, 7, 12))
    assert np.array_equal(merge_heads(split_heads(z3, 3)), z3)
    assert np.array_equal(merge_heads(split_heads(z3, 1)), z3)

    # attention scores
    q, k = rng.normal(size=(2, 3, 4, 5)), rng.normal(size=(2, 3, 6, 5))
    sc = attention_scores(q, k)
    assert sc.shape == (2, 3, 4, 6)
    assert np.allclose(sc, _loop_scores(q, k), atol=1e-9)

    # outer sum
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 4))
    ref = sum(np.outer(a[i], b[i]) for i in range(7))
    assert np.allclose(outer_sum(a, b), ref) and outer_sum(a, b).shape == (3, 4)

    # bilinear
    xb, w, yb = rng.normal(size=(6, 3)), rng.normal(size=(3, 5)), rng.normal(size=(6, 5))
    ref = np.array([xb[i] @ w @ yb[i] for i in range(6)])
    assert np.allclose(batched_bilinear(xb, w, yb), ref)

    # cosine
    c = cosine_matrix(x, y)
    assert c.shape == (5, 4) and np.all(np.abs(c) <= 1.0 + 1e-9)
    assert np.allclose(c[1, 2], x[1] @ y[2] / (np.linalg.norm(x[1]) * np.linalg.norm(y[2])))
    assert np.allclose(np.diag(cosine_matrix(x, x)), 1.0)
    xz = np.vstack([x, np.zeros((1, 3))])
    cz = cosine_matrix(xz, y)
    assert np.all(np.isfinite(cz)) and np.all(cz[-1] == 0.0)
    assert np.allclose(cosine_matrix(3.0 * x, y), c)  # scale invariance

    # gather
    logp = rng.normal(size=(3, 4, 9))
    ids = rng.integers(0, 9, size=(3, 4))
    g = gather_logprobs(logp, ids)
    assert g.shape == (3, 4)
    for bi in range(3):
        for ti in range(4):
            assert g[bi, ti] == logp[bi, ti, ids[bi, ti]]

    # masked mean
    xm = rng.normal(size=(4, 6, 3))
    lengths = np.array([6, 3, 1, 0])
    xm_poison = xm.copy()
    for bi, n in enumerate(lengths):
        xm_poison[bi, n:] = 1e9  # padding values must not leak into the mean
    mm = masked_mean(xm_poison, lengths)
    assert np.allclose(mm, _loop_masked_mean(xm, lengths))
    assert np.all(mm[3] == 0.0)

    # one hot
    oh = one_hot(np.array([[0, 2], [1, 1]]), 3)
    assert oh.shape == (2, 2, 3) and np.array_equal(oh.sum(axis=-1), np.ones((2, 2)))
    assert np.array_equal(oh[0, 1], np.array([0.0, 0.0, 1.0]))
    assert one_hot(np.array(4), 5).shape == (5,)


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
