"""Exercise 3 - A two-layer MLP with hand-written backprop and a gradient check.

TASK
    Implement a tanh MLP classifier (input -> hidden -> classes) in numpy.

    init_params(d_in, d_hidden, n_classes, seed=0) -> dict
        Keys "W1" (d_in, d_hidden), "b1" (d_hidden,), "W2" (d_hidden, n_classes),
        "b2" (n_classes,). Weights ~ N(0, 1/fan_in) from np.random.default_rng(seed),
        biases zero.
    softmax(logits: np.ndarray) -> np.ndarray
        Row-wise, numerically stable (subtract the row max).
    forward(params, X) -> tuple[np.ndarray, dict]
        Returns (logits, cache). Keep in `cache` whatever backward needs.
    loss_and_dlogits(logits, y) -> tuple[float, np.ndarray]
        Mean cross-entropy against integer labels y, and d loss / d logits.
    backward(params, cache, dlogits) -> dict
        Gradients with the same keys and shapes as `params`.
    numerical_grads(params, X, y, eps=1e-5) -> dict
        Central finite-difference gradients of the mean loss for EVERY parameter.
    sgd_step(params, grads, lr) -> None   (in-place update)

HINTS
    - The gradient of mean cross-entropy w.r.t. logits is (softmax - onehot) / n.
    - Linear layer: given dOut, dW = X.T @ dOut, db = dOut.sum(0), dX = dOut @ W.T.
    - tanh backward: dpre = dhidden * (1 - h**2), where h = tanh(pre) was cached.
    - For numerical_grads, perturb one entry at a time and restore it afterwards.

WHAT YOU SHOULD LEARN
    - Backprop as cached activations plus vector-Jacobian products, layer by layer.
    - The softmax + cross-entropy gradient (p - y) and why it is the right shape.
    - How a finite-difference check catches the classic bugs (missing 1/n, wrong
      transpose, forgetting a bias).
"""
from __future__ import annotations

import numpy as np


def init_params(d_in: int, d_hidden: int, n_classes: int, seed: int = 0) -> dict:
    raise NotImplementedError("TODO")


def softmax(logits: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    raise NotImplementedError("TODO")


def loss_and_dlogits(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    raise NotImplementedError("TODO")


def backward(params: dict, cache: dict, dlogits: np.ndarray) -> dict:
    raise NotImplementedError("TODO")


def numerical_grads(params: dict, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> dict:
    raise NotImplementedError("TODO")


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = rng.integers(0, 4, size=8)
    params = init_params(3, 5, 4, seed=1)
    assert params["W1"].shape == (3, 5) and params["W2"].shape == (5, 4)
    assert np.all(params["b1"] == 0) and np.all(params["b2"] == 0)

    # softmax: rows sum to one, stable for huge logits, shift invariant
    s = softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(s)) and np.allclose(s.sum(axis=1), 1.0)
    assert np.allclose(s[1], 1 / 3)

    # initial loss with near-uniform predictions is close to log(n_classes)
    logits, cache = forward(params, X)
    assert logits.shape == (8, 4)
    loss, dlogits = loss_and_dlogits(logits, y)
    assert abs(loss - np.log(4)) < 0.6
    # each row of dlogits sums to zero (softmax rows sum to 1, one-hot sums to 1)
    assert np.allclose(dlogits.sum(axis=1), 0.0)

    # hand-checked cross-entropy: logits [0, ln 3] -> p = [0.25, 0.75]; label 1 -> -ln 0.75
    l1, d1 = loss_and_dlogits(np.array([[0.0, np.log(3.0)]]), np.array([1]))
    assert abs(l1 - (-np.log(0.75))) < 1e-12
    assert np.allclose(d1, [[0.25, -0.25]])

    # the main event: analytic gradients equal finite differences for every parameter
    grads = backward(params, cache, dlogits)
    num = numerical_grads(params, X, y)
    for k in params:
        assert grads[k].shape == params[k].shape, k
        err = np.max(np.abs(grads[k] - num[k]))
        assert err < 1e-7, (k, err)

    # training reduces the loss on a small separable problem (XOR-like quadrants)
    rng = np.random.default_rng(3)
    Xt = rng.uniform(-1, 1, size=(200, 2))
    yt = ((Xt[:, 0] > 0) ^ (Xt[:, 1] > 0)).astype(int)
    p = init_params(2, 16, 2, seed=2)
    first = None
    for step in range(1500):
        lg, ch = forward(p, Xt)
        ls, dl = loss_and_dlogits(lg, yt)
        if first is None:
            first = ls
        sgd_step(p, backward(p, ch, dl), lr=0.5)
    final_logits, _ = forward(p, Xt)
    acc = (final_logits.argmax(axis=1) == yt).mean()
    assert ls < 0.5 * first and acc > 0.9, (first, ls, acc)


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
