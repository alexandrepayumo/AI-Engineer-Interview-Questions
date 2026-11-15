"""Exercise 7 - Initialisation, normalisation, dropout, and the InfoNCE loss.

TASK
    xavier_init(fan_in, fan_out, rng) -> np.ndarray
        Normal weights of shape (fan_in, fan_out), variance 2 / (fan_in + fan_out).
    he_init(fan_in, fan_out, rng) -> np.ndarray
        Normal weights of shape (fan_in, fan_out), variance 2 / fan_in.
    activation_stds(weights_fn, activation, depth, width, seed=0, batch=64) -> list[float]
        Push a standard-normal batch of shape (batch, width) through `depth` layers
        h = activation(h @ W), where W = weights_fn(width, width, rng). Return the
        standard deviation of h after each layer. `activation` is "relu" or "tanh".
    layer_norm(x, gamma, beta, eps=1e-5) -> np.ndarray
        Normalise over the LAST axis (per row), then scale by gamma and shift by beta.
        Use the biased variance (divide by the feature count).
    rms_norm(x, gamma, eps=1e-6) -> np.ndarray
        x / sqrt(mean(x ** 2) + eps) over the last axis, times gamma. No mean removal.
    batch_norm_train(x, gamma, beta, running_mean, running_var, momentum=0.1, eps=1e-5)
            -> tuple[y, new_running_mean, new_running_var]
        Normalise each feature over the batch axis (biased variance). Update the
        running statistics as (1 - momentum) * old + momentum * batch_stat, using
        the UNBIASED batch variance (n / (n - 1) factor) for the running variance.
    batch_norm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5) -> np.ndarray
        Normalise with the stored running statistics (no dependence on the batch).
    dropout(x, p, rng, train=True) -> tuple[np.ndarray, np.ndarray]
        Inverted dropout. Returns (out, mask). In train mode drop each entry with
        probability p and scale the survivors by 1 / (1 - p). In eval mode, return x
        unchanged with an all-ones mask.
    info_nce(queries, keys, temperature=0.07) -> float
        L2-normalise both, logits = q @ k.T / temperature, mean cross-entropy where
        row i's correct class is column i (in-batch negatives).

HINTS
    - He init exists because ReLU zeroes about half the signal: the factor 2 in
      2 / fan_in compensates for that halving of the second moment.
    - Subtract the row max before exponentiating in info_nce.
    - Use np.random.default_rng(seed) and pass the same rng to every weight draw.

WHAT YOU SHOULD LEARN
    - Variance-preserving init, measured: signals vanish or blow up exponentially with
      depth when the scale is wrong.
    - The axis difference between LayerNorm (per token) and BatchNorm (per feature
      across the batch), and why that makes BatchNorm batch-dependent.
    - Train/eval mode differences for BatchNorm and dropout.
    - InfoNCE as softmax cross-entropy over similarities.
"""
from __future__ import annotations

import numpy as np


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def activation_stds(weights_fn, activation: str, depth: int, width: int, seed: int = 0, batch: int = 64):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(batch, width))
    stds = []
    for _ in range(depth):
        W = weights_fn(width, width, rng)
        h = h @ W
        h = np.maximum(h, 0.0) if activation == "relu" else np.tanh(h)
        stds.append(float(h.std()))
    return stds


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return gamma * x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def batch_norm_train(x, gamma, beta, running_mean, running_var, momentum=0.1, eps=1e-5):
    n = x.shape[0]
    mu = x.mean(axis=0)
    var = x.var(axis=0)  # biased, used to normalise
    y = gamma * (x - mu) / np.sqrt(var + eps) + beta
    new_mean = (1 - momentum) * running_mean + momentum * mu
    new_var = (1 - momentum) * running_var + momentum * var * n / (n - 1)  # unbiased for the running estimate
    return y, new_mean, new_var


def batch_norm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    return gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta


def dropout(x: np.ndarray, p: float, rng: np.random.Generator, train: bool = True):
    if not train or p == 0.0:
        return x, np.ones_like(x)
    keep = (rng.uniform(size=x.shape) >= p).astype(x.dtype)
    mask = keep / (1.0 - p)  # inverted dropout: rescale now so inference needs nothing
    return x * mask, mask


def info_nce(queries: np.ndarray, keys: np.ndarray, temperature: float = 0.07) -> float:
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    k = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    logits = q @ k.T / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_probs)))


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # init variances
    w = he_init(400, 300, rng)
    assert w.shape == (400, 300) and abs(w.var() - 2 / 400) < 5e-4
    w = xavier_init(400, 300, rng)
    assert abs(w.var() - 2 / 700) < 5e-4

    # signal propagation through 30 ReLU layers of width 256
    he = activation_stds(he_init, "relu", 30, 256)
    xa = activation_stds(xavier_init, "relu", 30, 256)
    small = activation_stds(lambda i, o, r: r.normal(0, 0.01, (i, o)), "relu", 30, 256)
    large = activation_stds(lambda i, o, r: r.normal(0, 0.2, (i, o)), "relu", 30, 256)
    assert 0.3 < he[-1] / he[0] < 3.0, (he[0], he[-1])  # He keeps the scale
    assert xa[-1] < 0.01 * he[-1], (xa[-1], he[-1])  # Xavier under-scales ReLU nets
    assert small[-1] < 1e-20 or small[-1] == 0.0  # vanishes
    assert large[-1] > 1e6  # explodes
    # tanh with Xavier stays in a sane band
    tx = activation_stds(xavier_init, "tanh", 30, 256)
    assert 0.05 < tx[-1] < 1.0

    # LayerNorm: per-row mean 0 / variance 1 with gamma = 1, beta = 0
    x = rng.normal(5.0, 3.0, size=(6, 32))
    g, b = np.ones(32), np.zeros(32)
    y = layer_norm(x, g, b)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-9)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-3)
    # batch independence: a row normalises identically alone or inside a batch
    assert np.allclose(layer_norm(x[2:3], g, b), y[2:3])
    # gamma and beta act per feature
    assert np.allclose(layer_norm(x, 2 * g, b + 1.0), 2 * y + 1.0)
    # hand check: row [1, 2, 3] -> mean 2, var 2/3
    hc = layer_norm(np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3), eps=0.0)
    assert np.allclose(hc, [[-1.2247449, 0.0, 1.2247449]], atol=1e-6)

    # RMSNorm: unit RMS, and it does NOT remove the mean
    r = rms_norm(x, g)
    assert np.allclose(np.sqrt(np.mean(r ** 2, axis=-1)), 1.0, atol=1e-4)
    assert np.all(np.abs(r.mean(axis=-1)) > 0.1)
    assert np.allclose(rms_norm(np.array([[3.0, 4.0]]), np.ones(2), eps=0.0), [[3 / np.sqrt(12.5), 4 / np.sqrt(12.5)]])

    # BatchNorm: per-feature stats over the batch; output for a row depends on the others
    xb = rng.normal(2.0, 4.0, size=(64, 8))
    gb, bb = np.ones(8), np.zeros(8)
    yb, rm, rv = batch_norm_train(xb, gb, bb, np.zeros(8), np.ones(8))
    assert np.allclose(yb.mean(axis=0), 0.0, atol=1e-9) and np.allclose(yb.var(axis=0), 1.0, atol=1e-3)
    other = xb.copy()
    other[1:] = rng.normal(size=(63, 8))
    y_other, _, _ = batch_norm_train(other, gb, bb, np.zeros(8), np.ones(8))
    assert not np.allclose(yb[0], y_other[0])  # same input row, different output
    # running statistics move toward the batch statistics with the given momentum
    assert np.allclose(rm, 0.1 * xb.mean(axis=0))
    assert np.allclose(rv, 0.9 * 1.0 + 0.1 * xb.var(axis=0, ddof=1))
    # eval mode: batch independent, uses running stats only
    ye = batch_norm_eval(xb[:1], gb, bb, rm, rv)
    assert np.allclose(ye, batch_norm_eval(xb, gb, bb, rm, rv)[:1])
    # after many updates the running stats converge to the data statistics
    rm2, rv2 = np.zeros(8), np.ones(8)
    for _ in range(200):
        _, rm2, rv2 = batch_norm_train(xb, gb, bb, rm2, rv2)
    assert np.allclose(rm2, xb.mean(axis=0), atol=1e-6) and np.allclose(rv2, xb.var(axis=0, ddof=1), atol=1e-6)

    # dropout
    xd = np.ones((2000, 50))
    out, mask = dropout(xd, 0.3, np.random.default_rng(1))
    assert abs((mask == 0).mean() - 0.3) < 0.01
    assert np.allclose(np.unique(mask), [0.0, 1.0 / 0.7])
    assert abs(out.mean() - 1.0) < 0.01  # expectation preserved
    assert np.array_equal(out, xd * mask)
    ev, em = dropout(xd, 0.3, np.random.default_rng(1), train=False)
    assert np.array_equal(ev, xd) and np.all(em == 1.0)
    assert np.array_equal(dropout(xd, 0.0, np.random.default_rng(1))[0], xd)

    # InfoNCE: hand check, identity queries/keys with T = 1 -> log(1 + e^-1)
    I = np.eye(2)
    assert abs(info_nce(I, I, temperature=1.0) - np.log(1 + np.exp(-1.0))) < 1e-12
    # aligned pairs with a low temperature give ~0 loss; shuffled pairs are much worse
    base = np.linalg.qr(np.random.default_rng(2).normal(size=(16, 16)))[0][:8]
    assert info_nce(base, base, 0.05) < 1e-6
    assert info_nce(base, base[::-1], 0.05) > 5.0
    # uninformative (identical) keys give log(batch size)
    same = np.tile(base[:1], (8, 1))
    assert abs(info_nce(base, same, 0.05) - np.log(8)) < 1e-9
    # scale invariance from the normalisation
    assert abs(info_nce(3 * base, 0.1 * base, 0.05) - info_nce(base, base, 0.05)) < 1e-9


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
