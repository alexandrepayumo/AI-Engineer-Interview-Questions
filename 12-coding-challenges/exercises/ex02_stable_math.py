"""Exercise 02 - Numerically stable log-sum-exp, cross-entropy and logsigmoid.

TASK
----
Implement these with plain numpy so they survive extreme inputs (logits of +-1000,
float32, rows that are entirely -inf).

    logsumexp(x, axis=-1, keepdims=False)
        log(sum(exp(x))) along axis. A row of all -inf returns -inf (not NaN).
        The output dtype matches the input dtype.
    log_softmax(x, axis=-1)
        x - logsumexp(x, axis, keepdims=True).
    softplus(x)
        log(1 + exp(x)). No overflow for large x, no loss of precision for very negative x.
    logsigmoid(x)
        log(1 / (1 + exp(-x))) = -softplus(-x).
    cross_entropy(logits, targets) -> float
        Mean negative log-likelihood. logits is (N, V), targets is (N,) ints.
        Do not form probabilities and then take a log.
    cross_entropy_grad(logits, targets) -> (N, V)
        Gradient of the MEAN cross-entropy with respect to logits: (softmax - onehot) / N.
    bce_with_logits(z, y) -> float
        Mean binary cross-entropy where z are raw scores and y in {0, 1} (same shape).
        Use max(z, 0) - z*y + log1p(exp(-|z|)).

HINTS
-----
- Subtract the max before exp. Then the largest term is exp(0) = 1, so the sum is
  at least 1 and the log is safe. Add the max back afterwards.
- If the max itself is -inf, subtracting it gives nan. Replace non-finite maxima with 0.
- log1p(t) is accurate when t is tiny, which is exactly the case for exp(-|x|) with large |x|.
- Wrap intentional log(0) in np.errstate(divide="ignore") so a legitimate -inf does not warn.

WHAT YOU SHOULD LEARN
---------------------
- Why the naive formulas overflow: exp overflows at about 709.78 in float64 and 88.72
  in float32.
- The max-shift identity: logsumexp(x) = m + logsumexp(x - m).
- Why training code takes raw logits into the loss instead of probabilities.
- How to test numerics: compare against high precision, check invariances, and run a
  finite-difference gradient check.
"""
from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    raise NotImplementedError("TODO")


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    raise NotImplementedError("TODO")


def softplus(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def logsigmoid(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # logsumexp: matches the naive formula where the naive formula is safe
    x = rng.normal(size=(4, 6))
    assert np.allclose(logsumexp(x), np.log(np.exp(x).sum(axis=-1)))
    assert logsumexp(x, axis=0).shape == (6,)
    assert logsumexp(x, axis=1, keepdims=True).shape == (4, 1)

    # extreme values: the naive version overflows, the stable one does not
    big = np.array([1000.0, 1000.0])
    with np.errstate(all="ignore"):
        assert np.isinf(np.log(np.exp(big).sum()))
    assert abs(logsumexp(big) - (1000.0 + np.log(2.0))) < 1e-12
    assert abs(logsumexp(np.array([-1000.0, -1000.0])) - (-1000.0 + np.log(2.0))) < 1e-12
    assert abs(logsumexp(np.array([0.0, -1000.0])) - 0.0) < 1e-12

    # shift invariance: logsumexp(x + c) = logsumexp(x) + c
    assert np.allclose(logsumexp(x + 123.0), logsumexp(x) + 123.0)

    # -inf handling
    rows = np.array([[-np.inf, -np.inf, -np.inf], [0.0, -np.inf, 0.0]])
    out = logsumexp(rows)
    assert out[0] == -np.inf and abs(out[1] - np.log(2.0)) < 1e-12
    assert not np.any(np.isnan(log_softmax(rows[1:])))
    assert log_softmax(rows[1:])[0, 1] == -np.inf

    # dtype is preserved, and float32 survives a logit that would overflow exp
    x32 = np.array([[100.0, 100.0, 90.0]], dtype=np.float32)
    assert logsumexp(x32).dtype == np.float32
    assert np.all(np.isfinite(logsumexp(x32)))
    assert abs(float(logsumexp(x32)[0]) - (100.0 + np.log(2.0 + np.exp(-10.0)))) < 1e-4

    # log_softmax rows exponentiate to a distribution
    ls = log_softmax(rng.normal(size=(5, 7)) * 50)
    assert np.allclose(np.exp(ls).sum(axis=-1), 1.0)

    # softplus / logsigmoid
    xs = np.array([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    sp = softplus(xs)
    assert np.all(np.isfinite(sp)) and sp[0] == 0.0 and sp[-1] == 1000.0
    assert abs(sp[3] - np.log(2.0)) < 1e-15
    assert abs(sp[1] - np.exp(-50.0)) < 1e-30  # naive log(1 + exp(-50)) rounds to 0
    with np.errstate(all="ignore"):
        assert np.log(1.0 + np.exp(-50.0)) == 0.0
    ls_ = logsigmoid(xs)
    assert ls_[-1] == -0.0 or abs(ls_[-1]) < 1e-300
    assert abs(ls_[0] - (-1000.0)) < 1e-9
    assert abs(ls_[3] + np.log(2.0)) < 1e-15
    # identity: logsigmoid(x) - logsigmoid(-x) = x
    assert np.allclose(logsigmoid(xs) - logsigmoid(-xs), xs)

    # cross entropy
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    targets = np.array([0, 2])
    expected = (-np.log(np.exp(2) / (np.exp(2) + np.exp(1) + 1)) + np.log(3.0)) / 2
    assert abs(cross_entropy(logits, targets) - expected) < 1e-12
    # uniform logits over V classes give log V no matter how large the constant
    assert abs(cross_entropy(np.full((3, 10), 5000.0), np.array([0, 4, 9])) - np.log(10.0)) < 1e-9
    # confident and right -> ~0; confident and wrong -> large but finite
    sure = np.array([[1000.0, 0.0]])
    assert cross_entropy(sure, np.array([0])) == 0.0
    assert abs(cross_entropy(sure, np.array([1])) - 1000.0) < 1e-9

    # gradient: rows sum to zero and match central finite differences
    lg = rng.normal(size=(3, 5))
    tg = np.array([1, 4, 0])
    grad = cross_entropy_grad(lg, tg)
    assert np.allclose(grad.sum(axis=-1), 0.0, atol=1e-12)
    num = np.zeros_like(lg)
    h = 1e-6
    for i in range(lg.shape[0]):
        for j in range(lg.shape[1]):
            plus, minus = lg.copy(), lg.copy()
            plus[i, j] += h
            minus[i, j] -= h
            num[i, j] = (cross_entropy(plus, tg) - cross_entropy(minus, tg)) / (2 * h)
    assert np.allclose(grad, num, atol=1e-7)
    assert np.all(np.isfinite(cross_entropy_grad(np.array([[1000.0, -1000.0]]), np.array([1]))))

    # bce with logits
    z = np.array([-1000.0, -2.0, 0.0, 3.0, 1000.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    sig = 1.0 / (1.0 + np.exp(-z[1:4]))
    ref_mid = np.mean(-(y[1:4] * np.log(sig) + (1 - y[1:4]) * np.log(1 - sig)))
    assert abs(bce_with_logits(z[1:4], y[1:4]) - ref_mid) < 1e-12
    assert np.isfinite(bce_with_logits(z, y))
    assert abs(bce_with_logits(np.array([1000.0]), np.array([0.0])) - 1000.0) < 1e-9
    assert bce_with_logits(np.array([1000.0]), np.array([1.0])) == 0.0
    # agrees with the logsigmoid form
    via_ls = -np.mean(y * logsigmoid(z) + (1 - y) * logsigmoid(-z))
    assert abs(bce_with_logits(z, y) - via_ls) < 1e-9


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
