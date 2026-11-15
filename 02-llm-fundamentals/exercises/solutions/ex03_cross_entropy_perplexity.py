"""Exercise 3 - Cross-entropy, perplexity and bits-per-byte from raw logits.

TASK
----
Implement, using numpy only and without overflow for logits as large as 1e4:

    log_softmax(logits)                         -> ndarray, same shape, last axis normalised
    cross_entropy(logits, targets, ignore_index=-100) -> float
        mean negative log-likelihood in nats over positions whose target != ignore_index;
        logits shape (n, V), targets shape (n,) of ints. Raise ValueError if no position counts.
    perplexity(logits, targets, ignore_index=-100) -> float   # exp(cross_entropy)
    bits_per_byte(total_nll_nats, n_bytes)      -> float      # total NLL / (n_bytes * ln 2)
    cross_entropy_grad(logits, targets, ignore_index=-100) -> ndarray
        d(mean loss)/d(logits): (softmax - onehot) / n_counted, zero rows for ignored positions

HINTS
-----
- log_softmax(z) = (z - max z) - log(sum(exp(z - max z))).
- Never compute softmax and then log; log(0) = -inf for confident wrong predictions.

WHAT YOU SHOULD LEARN
---------------------
- The log-sum-exp trick and why every framework fuses softmax with the loss.
- Perplexity as the effective branching factor: uniform over V tokens has perplexity V.
- Why bits-per-byte allows comparison between models with different tokenisers.
- The clean gradient softmax - onehot that makes LM training numerically friendly.
"""
from __future__ import annotations

import math

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)  # largest entry becomes 0
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -100) -> float:
    """Mean NLL in nats over non-ignored positions."""
    keep = targets != ignore_index
    if not keep.any():
        raise ValueError("no positions to score")
    logp = log_softmax(logits[keep])
    nll = -logp[np.arange(logp.shape[0]), targets[keep]]
    return float(nll.mean())


def perplexity(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -100) -> float:
    """exp of the mean NLL."""
    return float(math.exp(cross_entropy(logits, targets, ignore_index)))


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Tokeniser-independent compression metric."""
    return total_nll_nats / (n_bytes * math.log(2))


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -100) -> np.ndarray:
    """Gradient of the mean loss with respect to the logits."""
    keep = targets != ignore_index
    n = int(keep.sum())
    if n == 0:
        raise ValueError("no positions to score")
    grad = np.exp(log_softmax(logits))          # softmax
    rows = np.arange(logits.shape[0])
    grad[rows[keep], targets[keep]] -= 1.0      # minus one-hot at the target
    grad[~keep] = 0.0                           # ignored rows contribute nothing
    return grad / n


def _run_tests() -> None:
    rng = np.random.default_rng(0)
    # Uniform logits: loss = ln V, perplexity = V.
    V = 50
    z = np.zeros((7, V))
    t = rng.integers(0, V, size=7)
    assert abs(cross_entropy(z, t) - math.log(V)) < 1e-12
    assert abs(perplexity(z, t) - V) < 1e-9

    # log_softmax rows exponentiate to 1 and survive huge logits.
    big = np.array([[10000.0, 0.0, -10000.0], [1e4, 1e4, 1e4]])
    ls = log_softmax(big)
    assert np.all(np.isfinite(ls))
    assert np.allclose(np.exp(ls).sum(-1), 1.0)
    assert np.allclose(ls[1], -math.log(3))

    # Confident and right: ~0 loss. Confident and wrong: ~ the logit gap, finite.
    z2 = np.array([[1000.0, 0.0, 0.0]])
    assert cross_entropy(z2, np.array([0])) < 1e-12
    assert abs(cross_entropy(z2, np.array([1])) - 1000.0) < 1e-9

    # Hand-checked value: logits (0, ln 3) -> p = (0.25, 0.75).
    z3 = np.array([[0.0, math.log(3.0)]])
    assert abs(cross_entropy(z3, np.array([1])) - (-math.log(0.75))) < 1e-12
    assert abs(perplexity(z3, np.array([1])) - 4 / 3) < 1e-12

    # ignore_index: padded positions do not change the mean.
    zz = rng.normal(size=(5, 11))
    tt = rng.integers(0, 11, size=5)
    base = cross_entropy(zz, tt)
    z_pad = np.vstack([zz, rng.normal(size=(3, 11))])
    t_pad = np.concatenate([tt, np.full(3, -100)])
    assert abs(cross_entropy(z_pad, t_pad) - base) < 1e-12
    try:
        cross_entropy(zz, np.full(5, -100))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # bits per byte: 1000 tokens at 2.0 nats each over 4000 bytes.
    bpb = bits_per_byte(1000 * 2.0, 4000)
    assert abs(bpb - 0.5 / math.log(2)) < 1e-12
    # Same text, a tokeniser with twice as many tokens at half the loss/token gives the same bpb.
    assert abs(bits_per_byte(2000 * 1.0, 4000) - bpb) < 1e-12

    # Gradient: matches central finite differences, rows sum to zero, ignored rows are zero.
    g = cross_entropy_grad(z_pad, t_pad)
    assert g.shape == z_pad.shape
    assert np.allclose(g.sum(-1), 0.0, atol=1e-12)
    assert np.all(g[5:] == 0.0)
    eps = 1e-6
    num = np.zeros_like(z_pad)
    for i in range(z_pad.shape[0]):
        for j in range(z_pad.shape[1]):
            zp, zm = z_pad.copy(), z_pad.copy()
            zp[i, j] += eps
            zm[i, j] -= eps
            num[i, j] = (cross_entropy(zp, t_pad) - cross_entropy(zm, t_pad)) / (2 * eps)
    assert np.allclose(g, num, atol=1e-7)


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
