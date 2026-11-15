"""Exercise 04 - Sliding-window and sparse attention mask builders.

TASK
----
Build boolean attention masks (True = the query may attend to the key) and use them in
a masked attention function. Rows are queries (i), columns are keys (j).

    causal_mask(t) -> (t, t) bool
        j <= i.
    sliding_window_mask(t, w) -> (t, t) bool
        Causal, and only the w most recent positions including the token itself:
        i - w < j <= i. w = 1 means self only.
    sink_window_mask(t, w, n_sink) -> (t, t) bool
        Sliding window plus the first n_sink positions as "attention sinks"
        (still causal: sinks are visible only when j <= i).
    strided_mask(t, stride) -> (t, t) bool
        Causal and (i - j < stride or (i - j) % stride == 0): a local band plus every
        stride-th earlier token, the Sparse Transformer "strided" pattern.
    key_padding_mask(lengths, t) -> (B, t) bool
        True for real tokens, False for padding, per sequence.
    combine_masks(structural, lengths) -> (B, t, t) bool
        structural is (t, t). Result is structural AND the key is not padding.
    masked_attention(q, k, v, mask) -> (B, T, d)
        Scaled dot-product attention. mask is bool and broadcasts to (B, T, T). A query
        with NO allowed key returns the zero vector (not NaN).
    allowed_pairs_window(t, w) -> int
        Number of True entries in sliding_window_mask(t, w), by closed form, no array.
    receptive_field(mask, layers) -> (t, t) bool
        Which input positions can influence each position after stacking `layers`
        attention layers that all use `mask`.

HINTS
-----
- Build i and j with np.arange and compare i[:, None] with j[None, :]. One broadcast
  gives the whole matrix. Let d = i - j and write every mask as a condition on d.
- For the closed form, row i allows min(i + 1, w) keys. Sum it for i = 0..t-1.
- Masked softmax: scores -> where(mask, scores, -inf), subtract the row max (use 0 when
  the row max is -inf), exp, divide by the row sum only where the sum is positive.
- Reachability after L layers is repeated boolean matrix multiplication:
  (m_int @ m_int) > 0.

WHAT YOU SHOULD LEARN
---------------------
- Every attention pattern is a predicate on (i, j). Seeing masks that way makes new
  patterns (dilated, block, local-global) easy to write.
- Why a window of w tokens still gives a long range: the receptive field grows by
  w - 1 per layer.
- Why sinks help streaming inference: the first tokens soak up attention mass, so
  evicting them from a rolling cache changes outputs a lot.
- The fully masked row trap, which produces NaN in a naive implementation.
"""
from __future__ import annotations

import numpy as np


def _distance(t: int) -> np.ndarray:
    """d[i, j] = i - j, shape (t, t)."""
    idx = np.arange(t)
    return idx[:, None] - idx[None, :]


def causal_mask(t: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def sliding_window_mask(t: int, w: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def sink_window_mask(t: int, w: int, n_sink: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def strided_mask(t: int, stride: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def key_padding_mask(lengths: np.ndarray, t: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def combine_masks(structural: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def masked_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def allowed_pairs_window(t: int, w: int) -> int:
    raise NotImplementedError("TODO")


def receptive_field(mask: np.ndarray, layers: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def _loop_attention(q, k, v, allowed):
    b, t, d = q.shape
    out = np.zeros_like(v)
    for bi in range(b):
        for i in range(t):
            js = [j for j in range(t) if allowed[bi, i, j]]
            if not js:
                continue
            s = np.array([q[bi, i] @ k[bi, j] / np.sqrt(d) for j in js])
            w = np.exp(s - s.max())
            w /= w.sum()
            out[bi, i] = sum(wi * v[bi, j] for wi, j in zip(w, js))
    return out


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # causal and window shapes by hand
    assert np.array_equal(causal_mask(3), np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool))
    w3 = sliding_window_mask(6, 3)
    assert np.array_equal(np.flatnonzero(w3[0]), [0])
    assert np.array_equal(np.flatnonzero(w3[1]), [0, 1])
    assert np.array_equal(np.flatnonzero(w3[4]), [2, 3, 4])
    assert np.array_equal(sliding_window_mask(5, 1), np.eye(5, dtype=bool))
    assert np.array_equal(sliding_window_mask(5, 5), causal_mask(5))
    assert np.array_equal(sliding_window_mask(5, 99), causal_mask(5))
    assert not np.any(np.triu(w3, k=1)), "window must stay causal"

    # closed form
    for t in (1, 2, 5, 17):
        for w in (1, 2, 3, 5, 40):
            assert allowed_pairs_window(t, w) == sliding_window_mask(t, w).sum(), (t, w)
    assert allowed_pairs_window(4096, 512) < causal_mask(4096).sum() / 4  # sparsity win

    # sinks
    sk = sink_window_mask(8, 2, 2)
    assert np.array_equal(np.flatnonzero(sk[5]), [0, 1, 4, 5])
    assert np.array_equal(np.flatnonzero(sk[1]), [0, 1])        # sink not visible from the future
    assert np.array_equal(np.flatnonzero(sk[0]), [0])
    assert np.array_equal(sink_window_mask(8, 3, 0), sliding_window_mask(8, 3))
    assert not np.any(np.triu(sk, k=1))

    # strided
    st = strided_mask(10, 3)
    assert np.array_equal(np.flatnonzero(st[9]), [0, 3, 6, 7, 8, 9])
    assert np.array_equal(strided_mask(7, 1), causal_mask(7))
    assert not np.any(np.triu(st, k=1))

    # padding
    kp = key_padding_mask(np.array([3, 5]), 5)
    assert np.array_equal(kp, np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool))
    cm = combine_masks(causal_mask(5), np.array([3, 5]))
    assert cm.shape == (2, 5, 5)
    assert np.array_equal(np.flatnonzero(cm[0, 4]), [0, 1, 2])
    assert np.array_equal(np.flatnonzero(cm[1, 4]), [0, 1, 2, 3, 4])

    # attention against the loop reference
    b, t, d = 2, 7, 4
    q, k, v = (rng.normal(size=(b, t, d)) for _ in range(3))
    mask3 = combine_masks(sliding_window_mask(t, 3), np.array([7, 5]))
    out = masked_attention(q, k, v, mask3)
    assert out.shape == (b, t, d)
    assert np.allclose(out, _loop_attention(q, k, v, mask3), atol=1e-9)
    # a (t, t) mask broadcasts over the batch
    assert np.allclose(masked_attention(q, k, v, causal_mask(t)),
                       _loop_attention(q, k, v, np.broadcast_to(causal_mask(t), (b, t, t))), atol=1e-9)

    # fully masked rows give zeros, not NaN. With length 0 every key is padding.
    only_pad = combine_masks(causal_mask(3), np.array([0]))
    z = masked_attention(q[:1, :3], k[:1, :3], v[:1, :3], only_pad)
    assert np.all(z == 0.0) and not np.any(np.isnan(z))

    # one allowed key returns that key's value exactly
    one = masked_attention(q, k, v, sliding_window_mask(t, 1))
    assert np.allclose(one, v)

    # information flow: values outside the window cannot change the output
    v2 = v.copy()
    v2[:, 0] += 100.0                       # position 0 is outside the window for i >= 3 when w = 3
    base = masked_attention(q, k, v, sliding_window_mask(t, 3))
    pert = masked_attention(q, k, v2, sliding_window_mask(t, 3))
    assert np.allclose(base[:, 3:], pert[:, 3:])
    assert not np.allclose(base[:, :3], pert[:, :3])
    # causality: changing the future never changes the past
    v3 = v.copy()
    v3[:, 5:] = 0.0
    assert np.allclose(masked_attention(q, k, v, causal_mask(t))[:, :5],
                       masked_attention(q, k, v3, causal_mask(t))[:, :5])

    # receptive field grows by (w - 1) per layer
    rf1 = receptive_field(sliding_window_mask(30, 3), 1)
    assert np.array_equal(rf1, sliding_window_mask(30, 3))
    rf4 = receptive_field(sliding_window_mask(30, 3), 4)
    assert np.flatnonzero(rf4[20])[0] == 20 - 4 * 2
    assert np.flatnonzero(rf4[3])[0] == 0
    assert receptive_field(causal_mask(10), 3).sum() == causal_mask(10).sum()
    # with sinks, every late token reaches position 0 in a single layer
    assert receptive_field(sink_window_mask(30, 3, 1), 1)[25, 0]
    assert not receptive_field(sliding_window_mask(30, 3), 1)[25, 0]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
