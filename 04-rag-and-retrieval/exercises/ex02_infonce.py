"""Exercise 02 - InfoNCE with in-batch and hard negatives, loss and gradients in numpy.

TASK
----
Implement:

    l2_normalize(x) -> np.ndarray
        Row-wise L2 normalisation of a 2-D array (add no epsilon trick that changes
        unit-norm rows; zero rows may stay zero).

    info_nce(q, docs, labels, tau) -> tuple[float, np.ndarray, np.ndarray]
        q:      (B, D) query embeddings
        docs:   (N, D) candidate document embeddings (N >= B)
        labels: (B,) int array, labels[i] = row of `docs` that is the positive for query i
        tau:    temperature (> 0)
        logits = q @ docs.T / tau, loss = mean over queries of cross-entropy of
        softmax(logits[i]) against labels[i].
        Returns (loss, dq, ddocs) where dq has shape (B, D) and ddocs has shape (N, D)
        and are the exact gradients of the loss with respect to q and docs
        (the inputs are used as given: no normalisation inside this function).

    in_batch_labels(batch_size) -> np.ndarray
        The labels for pure in-batch negatives: positive of query i is doc i.

    recall_of_positive(q, docs, labels) -> float
        Fraction of queries whose positive doc has the highest dot product.

HINTS
-----
- Let P = softmax(logits, axis=1) and Y be the one-hot matrix of labels.
  Then dlogits = (P - Y) / B, dq = dlogits @ docs / tau, ddocs = dlogits.T @ q / tau.
- Subtract the row max before exp for numerical stability.
- Any doc that is not a positive for some query still receives gradient: it is pushed away
  from every query in proportion to its softmax probability. That is how negatives work.

WHAT YOU SHOULD LEARN
---------------------
- Why a batch of B pairs gives B*(B-1) negatives for free, and why batch size matters.
- Uniform logits give loss = log(N): the "chance" baseline you should see at step 0.
- Lower temperature sharpens the softmax so the hardest negatives dominate the gradient.
- Hard negatives are just extra rows in `docs`: same code, much more informative gradient.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def in_batch_labels(batch_size: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def info_nce(
    q: np.ndarray, docs: np.ndarray, labels: np.ndarray, tau: float
) -> tuple[float, np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO")


def recall_of_positive(q: np.ndarray, docs: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # l2_normalize
    x = rng.normal(size=(5, 8))
    n = l2_normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.allclose(l2_normalize(np.zeros((2, 3))), 0.0)

    # Chance baseline: identical docs -> uniform softmax -> loss = log(N)
    q = l2_normalize(rng.normal(size=(4, 6)))
    docs_same = np.tile(l2_normalize(rng.normal(size=(1, 6))), (4, 1))
    loss, _, _ = info_nce(q, docs_same, in_batch_labels(4), tau=0.1)
    assert abs(loss - np.log(4)) < 1e-9

    # Perfect, well separated embeddings at low temperature -> loss near 0
    eye = np.eye(4)
    loss, dq, dd = info_nce(eye, eye, in_batch_labels(4), tau=0.05)
    assert loss < 1e-6
    assert np.abs(dq).max() < 1e-6 and np.abs(dd).max() < 1e-6

    # Known value: B=2, q = d = I2, tau = 1. logits = [[1,0],[0,1]]
    # loss = -log(e / (e + 1)) = log(1 + e^-1)
    loss, _, _ = info_nce(np.eye(2), np.eye(2), in_batch_labels(2), tau=1.0)
    assert abs(loss - np.log(1 + np.exp(-1))) < 1e-12

    # Numerical gradient check on random data, with extra hard negatives (N > B)
    B, N, D = 5, 9, 7
    q = rng.normal(size=(B, D))
    docs = rng.normal(size=(N, D))
    labels = rng.integers(0, N, size=B)
    tau = 0.7
    loss, dq, dd = info_nce(q, docs, labels, tau)
    assert dq.shape == q.shape and dd.shape == docs.shape
    eps = 1e-6
    for arr, grad, which in ((q, dq, "q"), (docs, dd, "d")):
        for _ in range(12):
            i, j = rng.integers(arr.shape[0]), rng.integers(arr.shape[1])
            old = arr[i, j]
            arr[i, j] = old + eps
            lp = info_nce(q, docs, labels, tau)[0]
            arr[i, j] = old - eps
            lm = info_nce(q, docs, labels, tau)[0]
            arr[i, j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grad[i, j]) < 1e-6, (which, i, j, num, grad[i, j])

    # Gradient descent on the query pulls it toward its positive and the loss falls
    q = l2_normalize(rng.normal(size=(6, 10)))
    d = l2_normalize(rng.normal(size=(6, 10)))
    lab = in_batch_labels(6)
    start, _, _ = info_nce(q, d, lab, 0.1)
    cur = q.copy()
    for _ in range(50):
        _, g, _ = info_nce(cur, d, lab, 0.1)
        cur = l2_normalize(cur - 0.5 * g)
    end, _, _ = info_nce(cur, d, lab, 0.1)
    assert end < start
    assert recall_of_positive(cur, d, lab) == 1.0

    # Adding a hard negative that is nearly identical to the positive raises the loss
    q1 = np.array([[1.0, 0.0]])
    pos = np.array([[1.0, 0.0]])
    easy = np.array([[-1.0, 0.0]])
    hard = np.array([[0.99, 0.14]])
    lab1 = np.array([0])
    base = info_nce(q1, np.vstack([pos, easy]), lab1, 0.1)[0]
    withhard = info_nce(q1, np.vstack([pos, hard]), lab1, 0.1)[0]
    assert withhard > base * 1000  # the easy negative contributes ~2e-9, the hard one ~0.3

    # Lower temperature -> larger gradient share on the hardest negative
    docs2 = np.vstack([pos, hard, easy])
    g_hot = info_nce(q1, docs2, lab1, 1.0)[2]
    g_cold = info_nce(q1, docs2, lab1, 0.05)[2]
    share_hot = np.linalg.norm(g_hot[1]) / np.linalg.norm(g_hot[2])
    share_cold = np.linalg.norm(g_cold[1]) / np.linalg.norm(g_cold[2])
    assert share_cold > share_hot


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
