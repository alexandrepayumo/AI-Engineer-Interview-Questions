"""Exercise 4 - Grouped-query attention, checked against multi-head attention.

TASK
----
Implement

    gqa_attention(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, causal=True) -> ndarray (n, d_model)

with x of shape (n, d_model), Wq (d_model, n_heads*hd), Wk and Wv (d_model, n_kv_heads*hd),
Wo (n_heads*hd, d_model). Query head h uses key/value head h // (n_heads // n_kv_heads).
n_kv_heads == n_heads is ordinary multi-head attention; n_kv_heads == 1 is multi-query.
Raise ValueError if n_heads is not divisible by n_kv_heads.

Also implement

    kv_cache_elems_per_token(n_kv_heads, head_dim) -> int    # K and V elements per layer

HINTS
-----
- Reshape projections to (n, heads, hd) and move heads first.
- np.repeat(K, group_size, axis=head_axis) expands KV heads to match query heads. Real
  kernels avoid the copy, but repeating is the clearest correct version.
- Causal mask: set scores[i, j] = -inf for j > i before the softmax.

WHAT YOU SHOULD LEARN
---------------------
- GQA is MHA with tied K/V projections across groups, nothing more.
- MHA and MQA are the two ends of one dial.
- The cache cost is n_kv_heads * head_dim per token per layer, independent of n_heads.
"""
from __future__ import annotations

import numpy as np


def kv_cache_elems_per_token(n_kv_heads: int, head_dim: int) -> int:
    """K plus V elements stored per token per layer."""
    return 2 * n_kv_heads * head_dim


def gqa_attention(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                  Wo: np.ndarray, n_heads: int, n_kv_heads: int, causal: bool = True) -> np.ndarray:
    """Grouped-query self-attention for one sequence."""
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    n, _ = x.shape
    hd = Wq.shape[1] // n_heads
    group = n_heads // n_kv_heads
    q = (x @ Wq).reshape(n, n_heads, hd).transpose(1, 0, 2)       # (H, n, hd)
    k = (x @ Wk).reshape(n, n_kv_heads, hd).transpose(1, 0, 2)    # (G, n, hd)
    v = (x @ Wv).reshape(n, n_kv_heads, hd).transpose(1, 0, 2)
    k = np.repeat(k, group, axis=0)                               # (H, n, hd): head h -> group h // group
    v = np.repeat(v, group, axis=0)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)               # (H, n, n)
    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    out = (w @ v).transpose(1, 0, 2).reshape(n, n_heads * hd)     # concatenate heads
    return out @ Wo


def _reference_mha(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """Independent per-head loop implementation used only by the tests."""
    n = x.shape[0]
    hd = Wq.shape[1] // n_heads
    outs = []
    for h in range(n_heads):
        sl = slice(h * hd, (h + 1) * hd)
        q, k, v = x @ Wq[:, sl], x @ Wk[:, sl], x @ Wv[:, sl]
        s = q @ k.T / np.sqrt(hd)
        if causal:
            for i in range(n):
                s[i, i + 1:] = -np.inf
        e = np.exp(s - s.max(-1, keepdims=True))
        outs.append((e / e.sum(-1, keepdims=True)) @ v)
    return np.concatenate(outs, axis=-1) @ Wo


def _run_tests() -> None:
    rng = np.random.default_rng(1)
    n, d, H, hd = 6, 16, 4, 4
    x = rng.normal(size=(n, d))
    Wq = rng.normal(size=(d, H * hd)) * 0.3
    Wo = rng.normal(size=(H * hd, d)) * 0.3

    # 1. groups == heads reproduces plain MHA.
    Wk = rng.normal(size=(d, H * hd)) * 0.3
    Wv = rng.normal(size=(d, H * hd)) * 0.3
    out = gqa_attention(x, Wq, Wk, Wv, Wo, H, H)
    assert out.shape == (n, d)
    assert np.allclose(out, _reference_mha(x, Wq, Wk, Wv, Wo, H), atol=1e-10)

    # 2. GQA equals MHA whose K/V weights are tiled across each group.
    for G in (1, 2):
        Wk_g = rng.normal(size=(d, G * hd)) * 0.3
        Wv_g = rng.normal(size=(d, G * hd)) * 0.3
        rep = H // G
        Wk_full = np.concatenate([Wk_g[:, (h // rep) * hd:(h // rep + 1) * hd] for h in range(H)], axis=1)
        Wv_full = np.concatenate([Wv_g[:, (h // rep) * hd:(h // rep + 1) * hd] for h in range(H)], axis=1)
        got = gqa_attention(x, Wq, Wk_g, Wv_g, Wo, H, G)
        want = _reference_mha(x, Wq, Wk_full, Wv_full, Wo, H)
        assert np.allclose(got, want, atol=1e-10), f"G={G}"

    # 3. Causality: editing a future token leaves earlier outputs unchanged.
    Wk2 = rng.normal(size=(d, 2 * hd)) * 0.3
    Wv2 = rng.normal(size=(d, 2 * hd)) * 0.3
    base = gqa_attention(x, Wq, Wk2, Wv2, Wo, H, 2)
    x2 = x.copy()
    x2[4:] += 5.0
    changed = gqa_attention(x2, Wq, Wk2, Wv2, Wo, H, 2)
    assert np.allclose(base[:4], changed[:4], atol=1e-10)
    assert not np.allclose(base[4:], changed[4:])

    # 4. Non-causal differs from causal; first token output is just its own value mix in causal mode.
    nc = gqa_attention(x, Wq, Wk2, Wv2, Wo, H, 2, causal=False)
    assert not np.allclose(nc, base)

    # 5. Invalid grouping and cache arithmetic.
    try:
        gqa_attention(x, Wq, Wk2, Wv2, Wo, H, 3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert kv_cache_elems_per_token(8, 128) == 2048
    assert kv_cache_elems_per_token(64, 128) == 8 * kv_cache_elems_per_token(8, 128)


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
