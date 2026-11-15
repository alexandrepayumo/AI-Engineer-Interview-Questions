"""Exercise 6 - Top-k Mixture-of-Experts router with a load-balancing loss.

TASK
----
Implement

    topk_route(x, Wg, k) -> (idx, gates, probs)
        x (n, d), Wg (d, E). probs = softmax(x @ Wg) over experts, shape (n, E).
        idx (n, k): the k highest-probability experts per token (descending).
        gates (n, k): the probs of those experts renormalised to sum to 1 per token.

    load_balancing_loss(probs, idx, n_experts) -> float
        Switch-style auxiliary loss:  E * sum_i f_i * P_i  where
        f_i = fraction of the n*k routing assignments that went to expert i,
        P_i = mean router probability of expert i over tokens.
        Equals 1.0 for perfectly uniform routing and E for total collapse onto one expert (k=1).

    moe_forward(x, Wg, experts, k) -> ndarray (n, d)
        experts is a list of (W_in (d, f), W_out (f, d)); each expert is relu(x @ W_in) @ W_out.
        Output per token = sum over its chosen experts of gate * expert(x).

    moe_param_counts(d_model, d_ff, n_experts, k, gated=True, n_shared=0) -> (total, active)
        MLP parameters stored vs. used per token (router weights d_model * n_experts included
        in both; shared experts are always active).

HINTS
-----
- np.argsort(-probs, axis=1, kind="stable")[:, :k] gives the top-k indices.
- Loop over experts, not tokens, in moe_forward: gather the tokens routed to expert e,
  run them as one batch, scatter results back with their gates.

WHAT YOU SHOULD LEARN
---------------------
- How routing, gating and batching by expert actually work.
- Why the balance loss multiplies a hard count f_i with a differentiable P_i.
- The memory-versus-compute split of MoE: total parameters stored, active parameters used.
"""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def topk_route(x: np.ndarray, Wg: np.ndarray, k: int):
    """Return (idx, gates, probs) for top-k routing."""
    raise NotImplementedError("TODO")


def load_balancing_loss(probs: np.ndarray, idx: np.ndarray, n_experts: int) -> float:
    """Switch-transformer style auxiliary loss."""
    raise NotImplementedError("TODO")


def moe_forward(x: np.ndarray, Wg: np.ndarray, experts: list, k: int) -> np.ndarray:
    """Sparse MoE layer, batched by expert."""
    raise NotImplementedError("TODO")


def moe_param_counts(d_model: int, d_ff: int, n_experts: int, k: int,
                     gated: bool = True, n_shared: int = 0) -> tuple[int, int]:
    """(total, active) MLP-side parameters of one MoE layer."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    rng = np.random.default_rng(2)
    n, d, E, f, k = 12, 8, 4, 16, 2
    x = rng.normal(size=(n, d))
    Wg = rng.normal(size=(d, E))

    idx, gates, probs = topk_route(x, Wg, k)
    assert idx.shape == (n, k) and gates.shape == (n, k) and probs.shape == (n, E)
    assert np.allclose(probs.sum(1), 1.0)
    assert np.allclose(gates.sum(1), 1.0)
    # idx is descending in probability and distinct per token.
    top = np.take_along_axis(probs, idx, axis=1)
    assert np.all(top[:, 0] >= top[:, 1])
    assert np.all(idx[:, 0] != idx[:, 1])
    assert np.all(top[:, 1] >= np.sort(probs, axis=1)[:, -2] - 1e-15)

    # Balance loss: uniform -> 1, collapse -> E.
    uni = np.full((8, 4), 0.25)
    assert abs(load_balancing_loss(uni, np.arange(8).reshape(8, 1) % 4, 4) - 1.0) < 1e-12
    collapsed = np.zeros((8, 4))
    collapsed[:, 0] = 1.0
    assert abs(load_balancing_loss(collapsed, np.zeros((8, 1), dtype=int), 4) - 4.0) < 1e-12
    # Real router outputs are never below the balanced optimum of 1 (up to rounding)
    # when routing is top-1 and probs are consistent with it.
    i1, _, p1 = topk_route(x, Wg, 1)
    assert load_balancing_loss(p1, i1, E) >= 1.0 - 1e-9
    # A skewed router has a higher loss than a zero-weight (uniform) router.
    zero_idx, _, zero_p = topk_route(x, np.zeros((d, E)), 1)
    assert abs(load_balancing_loss(zero_p, np.arange(n).reshape(n, 1) % E, E) - 1.0) < 1e-12
    assert load_balancing_loss(p1, i1, E) > 1.0 - 1e-9 and zero_idx.shape == (n, 1)

    # moe_forward equals a slow per-token reference.
    experts = [(rng.normal(size=(d, f)) * 0.3, rng.normal(size=(f, d)) * 0.3) for _ in range(E)]
    got = moe_forward(x, Wg, experts, k)
    want = np.zeros_like(x)
    for t in range(n):
        for j in range(k):
            W_in, W_out = experts[idx[t, j]]
            want[t] += gates[t, j] * (np.maximum(x[t] @ W_in, 0.0) @ W_out)
    assert np.allclose(got, want, atol=1e-12)

    # With k = E and identical experts, the output equals one dense expert (gates sum to 1).
    same = [experts[0]] * E
    dense = np.maximum(x @ experts[0][0], 0.0) @ experts[0][1]
    assert np.allclose(moe_forward(x, Wg, same, E), dense, atol=1e-12)

    # Parameter accounting: a DeepSeek-V3-like layer shape (fine-grained experts + 1 shared).
    total, active = moe_param_counts(7168, 2048, 256, 8, gated=True, n_shared=1)
    assert total == 257 * 3 * 7168 * 2048 + 7168 * 256
    assert active == 9 * 3 * 7168 * 2048 + 7168 * 256
    assert 25 < total / active < 30
    # Mixtral-like: 8 experts, top-2, no shared expert -> active is ~1/4 of the expert weights.
    t2, a2 = moe_param_counts(4096, 14336, 8, 2)
    assert abs((a2 - 4096 * 8) / (t2 - 4096 * 8) - 0.25) < 1e-12


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
