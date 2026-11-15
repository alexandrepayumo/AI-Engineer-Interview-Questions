"""Exercise 07 - Find the planted bugs in a transformer block.

TASK
----
`buggy_attention` and `buggy_block` below look plausible, run without errors and return
arrays of the right shape. Each contains bugs in regions marked REGION 1 to REGION 4:
three regions in `buggy_attention`, one in `buggy_block`. Every region holds exactly
one bug. Do not edit the buggy code. Instead write corrected versions:

    fixed_attention(x, wq, wk, wv, wo, n_heads) -> (B, T, D)
        Causal multi-head self-attention. x is (B, T, D), the four weights are (D, D),
        D = n_heads * dh. Head i uses channels [i*dh, (i+1)*dh) of q, k and v.
    fixed_block(x, p, n_heads) -> (B, T, D)
        Pre-norm transformer block:
            h   = x + attention(layer_norm(x))
            out = h + mlp(layer_norm(h))
        p holds wq, wk, wv, wo, ln1_g, ln1_b, ln2_g, ln2_b, w1, b1, w2, b2.
        It must use fixed_attention and the provided _layer_norm and _mlp helpers.

The tests are the real lesson. `_run_tests` contains a set of named checks. Each one is
a property or a tiny oracle that exposes a class of bug without needing a reference
implementation of the whole model. The tests first check your fixed code, then assert
that the buggy code FAILS at least the checks named below, so you can see which check
catches which bug.

HINTS
-----
- Read every region and ask "what would this look like on paper?" Check the divisor, the
  order of mask and softmax, the axes of transpose and reshape, and the residual wiring.
- A mask applied after softmax zeroes the future weights, but the softmax denominator
  was computed over the future scores too. Which checks notice that?
- With q = k = 0 every allowed score is equal, so attention is a running average of v.
  That oracle is easy to compute by hand and it is blind to the scale, so it isolates
  mask and head-merge bugs.
- The scale bug only shows when n_heads > 1 (with one head, D equals dh).
- A block whose weights are all zero should be the identity, which catches wiring bugs.

WHAT YOU SHOULD LEARN
---------------------
- A debugging order: shapes, invariants (rows sum to 1, causality), tiny oracles, then a
  slow loop reference.
- Why bugs that keep shapes intact are the dangerous ones.
- How to turn "something is wrong" into a test that names the bug class.
"""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _mlp(x: np.ndarray, p: dict) -> np.ndarray:
    return _gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


# ------------------------------------------------------------------ buggy reference code

def buggy_attention(x, wq, wk, wv, wo, n_heads):
    b, t, d = x.shape
    dh = d // n_heads

    def split(a):
        return a.reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)

    q, k, v = split(x @ wq), split(x @ wk), split(x @ wv)

    # REGION 1
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)

    # REGION 2
    probs = _softmax(scores)
    probs = probs * np.tril(np.ones((t, t)))

    out = probs @ v

    # REGION 3
    out = out.reshape(b, t, d)

    return out @ wo


def buggy_block(x, p, n_heads):
    a = buggy_attention(_layer_norm(x, p["ln1_g"], p["ln1_b"]),
                        p["wq"], p["wk"], p["wv"], p["wo"], n_heads)
    h = x + a

    # REGION 4
    out = _mlp(_layer_norm(h, p["ln2_g"], p["ln2_b"]), p)

    return out


# ------------------------------------------------------------------ your fixes

def fixed_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray,
                    wo: np.ndarray, n_heads: int) -> np.ndarray:
    b, t, d = x.shape
    dh = d // n_heads

    def split(a):
        return a.reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)

    q, k, v = split(x @ wq), split(x @ wk), split(x @ wv)
    # FIX 1: the scale is sqrt(d_head), not sqrt(d_model). It matters only when n_heads > 1.
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    # FIX 2: mask BEFORE softmax with -inf, so the surviving weights renormalise to 1.
    mask = np.tril(np.ones((t, t), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    probs = _softmax(scores)
    out = probs @ v                                    # (b, h, t, dh)
    # FIX 3: bring time back in front of heads BEFORE flattening, or heads and positions mix.
    out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
    return out @ wo


def fixed_block(x: np.ndarray, p: dict, n_heads: int) -> np.ndarray:
    a = fixed_attention(_layer_norm(x, p["ln1_g"], p["ln1_b"]),
                        p["wq"], p["wk"], p["wv"], p["wo"], n_heads)
    h = x + a
    # FIX 4: the MLP branch is a residual update too. Dropping "h +" discards the whole stream.
    return h + _mlp(_layer_norm(h, p["ln2_g"], p["ln2_b"]), p)


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    def make_params(d: int, seed: int, zero: bool = False) -> dict:
        r = np.random.default_rng(seed)
        s = 0.0 if zero else 0.4
        return {
            "wq": r.normal(size=(d, d)) * s, "wk": r.normal(size=(d, d)) * s,
            "wv": r.normal(size=(d, d)) * s, "wo": r.normal(size=(d, d)) * s,
            "ln1_g": np.ones(d) + 0.1 * r.normal(size=d), "ln1_b": 0.1 * r.normal(size=d),
            "ln2_g": np.ones(d) + 0.1 * r.normal(size=d), "ln2_b": 0.1 * r.normal(size=d),
            "w1": r.normal(size=(d, 4 * d)) * s, "b1": np.zeros(4 * d) if zero else 0.1 * r.normal(size=4 * d),
            "w2": r.normal(size=(4 * d, d)) * s, "b2": np.zeros(d) if zero else 0.1 * r.normal(size=d),
        }

    def loop_attention(x, wq, wk, wv, wo, n_heads):
        """Slow oracle: one head, one query, one key at a time."""
        b, t, d = x.shape
        dh = d // n_heads
        q, k, v = x @ wq, x @ wk, x @ wv
        out = np.zeros((b, t, d))
        for bi in range(b):
            for h in range(n_heads):
                sl = slice(h * dh, (h + 1) * dh)
                for i in range(t):
                    s = np.array([q[bi, i, sl] @ k[bi, j, sl] / np.sqrt(dh) for j in range(i + 1)])
                    w = np.exp(s - s.max())
                    w /= w.sum()
                    out[bi, i, sl] = sum(w[j] * v[bi, j, sl] for j in range(i + 1))
        return out @ wo

    def attention_checks(fn) -> list[str]:
        """Names of the checks that `fn` fails."""
        failed = []
        d, h, t = 8, 2, 6
        p = make_params(d, 1)
        x = rng.normal(size=(2, t, d))
        args = (p["wq"], p["wk"], p["wv"], p["wo"], h)

        # 1. Causality: changing the future must not change the past.
        x2 = x.copy()
        x2[:, 3:] += rng.normal(size=x2[:, 3:].shape)
        if not np.allclose(fn(x, *args)[:, :3], fn(x2, *args)[:, :3]):
            failed.append("causality")

        # 2. Rows of attention weights sum to 1, so identical value rows come back unchanged.
        row = rng.normal(size=(1, 1, d))
        x_const = np.repeat(row, t, axis=1)
        eye = np.eye(d)
        out = fn(x_const, p["wq"], p["wk"], eye, eye, h)
        if not np.allclose(out, x_const):
            failed.append("constant_rows")

        # 3. q = k = 0 gives equal scores, so output is the running mean of v (hand oracle).
        zero = np.zeros((d, d))
        out = fn(x, zero, zero, eye, eye, h)
        running_mean = np.cumsum(x, axis=1) / np.arange(1, t + 1)[None, :, None]
        if not np.allclose(out, running_mean):
            failed.append("uniform_attention")

        # 4. Full comparison with the slow loop, for two head counts.
        for heads in (2, 4):
            if not np.allclose(fn(x, p["wq"], p["wk"], p["wv"], p["wo"], heads),
                               loop_attention(x, p["wq"], p["wk"], p["wv"], p["wo"], heads)):
                failed.append(f"matches_reference_h{heads}")
        return failed

    def block_checks(fn, attn) -> list[str]:
        failed = []
        d, h, t = 8, 2, 5
        x = rng.normal(size=(2, t, d))

        # Zero weights: attention and MLP both output 0, so the block is the identity.
        p0 = make_params(d, 2, zero=True)
        if not np.allclose(fn(x, p0, h), x):
            failed.append("identity_when_zero")

        # Full comparison with a block assembled from the slow attention oracle.
        p = make_params(d, 3)
        a = loop_attention(_layer_norm(x, p["ln1_g"], p["ln1_b"]), p["wq"], p["wk"], p["wv"], p["wo"], h)
        hmid = x + a
        want = hmid + _mlp(_layer_norm(hmid, p["ln2_g"], p["ln2_b"]), p)
        if not np.allclose(fn(x, p, h), want):
            failed.append("matches_reference_block")
        return failed

    # ---- your code must pass every check
    assert attention_checks(fixed_attention) == [], attention_checks(fixed_attention)
    assert block_checks(fixed_block, fixed_attention) == [], block_checks(fixed_block, fixed_attention)

    # ---- and the checks must really detect the planted bugs
    bad = attention_checks(buggy_attention)
    assert "causality" in bad              # REGION 2: the softmax denominator still sees the future
    assert "constant_rows" in bad          # catches REGION 2 (rows no longer sum to 1)
    assert "uniform_attention" in bad      # catches REGION 2 and REGION 3 (mask, head merge)
    assert "matches_reference_h2" in bad and "matches_reference_h4" in bad  # REGION 1 (scale)
    bad_block = block_checks(buggy_block, buggy_attention)
    assert "identity_when_zero" in bad_block      # catches REGION 4 (dropped residual)
    assert "matches_reference_block" in bad_block

    # shapes survive the bugs, which is why shape asserts alone never find them
    x = rng.normal(size=(2, 5, 8))
    p = make_params(8, 4)
    assert buggy_block(x, p, 2).shape == fixed_block(x, p, 2).shape == x.shape

    # one more property for the fixed code: with one head the scale bug cannot hide anywhere,
    # and the result must equal the loop oracle exactly as before
    assert np.allclose(fixed_attention(x, p["wq"], p["wk"], p["wv"], p["wo"], 1),
                       loop_attention(x, p["wq"], p["wk"], p["wv"], p["wo"], 1))


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
