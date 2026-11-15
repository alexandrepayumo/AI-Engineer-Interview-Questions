"""Exercise 1 - Logistic regression with gradient descent, from scratch.

TASK
    Implement binary logistic regression with full-batch gradient descent.

    sigmoid(z: np.ndarray) -> np.ndarray
        Numerically stable logistic function. Must not overflow for z = -1000 or 1000.
    bce_loss(w, b, X, y, l2=0.0) -> float
        Mean binary cross-entropy of p = sigmoid(X @ w + b) against y in {0, 1},
        plus (l2 / 2) * sum(w ** 2). Do not penalise b. Must be stable for huge logits
        (work from the logits, not from clipped probabilities).
    gradients(w, b, X, y, l2=0.0) -> tuple[np.ndarray, float]
        Analytic gradient of bce_loss with respect to (w, b).
    fit(X, y, lr=0.5, steps=500, l2=0.0) -> tuple[np.ndarray, float, list[float]]
        Start from w = 0, b = 0. Run `steps` gradient-descent steps.
        Return (w, b, losses) where losses[i] is the loss BEFORE step i.
    predict_proba(w, b, X) -> np.ndarray
    predict(w, b, X, threshold=0.5) -> np.ndarray of ints in {0, 1}

HINTS
    - The gradient of the mean BCE with respect to the logits is (p - y) / n.
      Then dw = X.T @ (p - y) / n + l2 * w and db = mean(p - y).
    - Stable BCE from logits z: mean(max(z, 0) - z * y + log(1 + exp(-|z|))).
    - Stable sigmoid: use exp(-z) for z >= 0 and exp(z) / (1 + exp(z)) for z < 0.

WHAT YOU SHOULD LEARN
    - Cross-entropy is the negative log-likelihood of a Bernoulli model, and its
      gradient through the sigmoid collapses to (p - y).
    - Why implementations work from logits (log-sum-exp style) rather than from
      probabilities.
    - How to verify an analytic gradient against finite differences.
    - What L2 regularisation does to the learned weights.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    # For z >= 0, exp(-z) is in (0, 1]; for z < 0, exp(z) is in (0, 1). Never overflows.
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_loss(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
    z = X @ w + b
    # softplus(z) - y * z, written so that exp() only sees non-positive numbers
    per_example = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(per_example.mean() + 0.5 * l2 * np.sum(w ** 2))


def gradients(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[np.ndarray, float]:
    n = X.shape[0]
    p = sigmoid(X @ w + b)
    err = p - y  # dLoss/dlogit (before the 1/n)
    dw = X.T @ err / n + l2 * w
    db = float(err.mean())
    return dw, db


def fit(X: np.ndarray, y: np.ndarray, lr: float = 0.5, steps: int = 500, l2: float = 0.0):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses: list[float] = []
    for _ in range(steps):
        losses.append(bce_loss(w, b, X, y, l2))
        dw, db = gradients(w, b, X, y, l2)
        w = w - lr * dw
        b = b - lr * db
    return w, b, losses


def predict_proba(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w + b)


def predict(w: np.ndarray, b: float, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(w, b, X) >= threshold).astype(int)


def _make_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=[-1.5, -1.0], scale=1.0, size=(n // 2, 2))
    X1 = rng.normal(loc=[1.5, 1.0], scale=1.0, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    return X, y


def _run_tests() -> None:
    # sigmoid: values and stability
    assert np.allclose(sigmoid(np.array([0.0])), [0.5])
    big = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(big)) and big[0] == 0.0 and big[1] == 1.0
    assert np.allclose(sigmoid(np.array([2.0])) + sigmoid(np.array([-2.0])), [1.0])

    # loss at w = 0 is log 2 regardless of labels
    X, y = _make_data()
    assert abs(bce_loss(np.zeros(2), 0.0, X, y) - np.log(2)) < 1e-12

    # hand-checked tiny case: one example x = 2, y = 1, w = 0.5, b = 0 -> z = 1
    Xs, ys = np.array([[2.0]]), np.array([1.0])
    expected = np.log(1 + np.exp(-1.0))  # -log(sigmoid(1)) = 0.3133
    assert abs(bce_loss(np.array([0.5]), 0.0, Xs, ys) - expected) < 1e-12
    dw, db = gradients(np.array([0.5]), 0.0, Xs, ys)
    p = 1 / (1 + np.exp(-1.0))
    assert abs(dw[0] - (p - 1) * 2.0) < 1e-12 and abs(db - (p - 1)) < 1e-12

    # stable for enormous logits
    assert np.isfinite(bce_loss(np.array([1e4]), 0.0, np.array([[1.0], [-1.0]]), np.array([0.0, 1.0])))

    # analytic gradient equals central finite differences (with L2)
    rng = np.random.default_rng(1)
    w0, b0 = rng.normal(size=2), 0.3
    for l2 in (0.0, 0.1):
        dw, db = gradients(w0, b0, X, y, l2)
        eps = 1e-6
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            num = (bce_loss(w0 + e, b0, X, y, l2) - bce_loss(w0 - e, b0, X, y, l2)) / (2 * eps)
            assert abs(num - dw[i]) < 1e-6, (num, dw[i])
        num_b = (bce_loss(w0, b0 + eps, X, y, l2) - bce_loss(w0, b0 - eps, X, y, l2)) / (2 * eps)
        assert abs(num_b - db) < 1e-6

    # training: loss decreases monotonically for a small enough learning rate, accuracy is high
    w, b, losses = fit(X, y, lr=0.5, steps=300)
    assert len(losses) == 300
    assert all(losses[i + 1] <= losses[i] + 1e-12 for i in range(len(losses) - 1))
    acc = (predict(w, b, X) == y).mean()
    assert acc > 0.9, acc
    assert predict_proba(w, b, X).min() >= 0.0 and predict_proba(w, b, X).max() <= 1.0

    # L2 shrinks the weight norm
    w_reg, _, _ = fit(X, y, lr=0.5, steps=300, l2=0.5)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w)

    # threshold behaviour: raising the threshold cannot add positives
    assert predict(w, b, X, 0.9).sum() <= predict(w, b, X, 0.5).sum()


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
