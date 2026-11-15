"""Exercise 03 - Bradley-Terry reward model trained with the pairwise logistic loss.

TASK
----
Implement (all numpy, no autograd):

    sigmoid(x) -> np.ndarray
        Numerically stable logistic function (no overflow warnings for x = +-1000).

    bt_prob(r_w, r_l) -> np.ndarray
        Bradley-Terry probability that the first item wins: sigmoid(r_w - r_l).

    bt_loss_grad(w, Xw, Xl, l2=0.0) -> tuple[float, np.ndarray]
        Linear reward r(x) = x @ w. Xw and Xl are (N, D) features of the chosen and rejected
        responses. Loss = mean_i -log sigmoid((Xw_i - Xl_i) @ w) + 0.5 * l2 * |w|^2.
        Return (loss, gradient wrt w). Use np.logaddexp(0, -d) for the stable log-loss.

    train_reward_model(Xw, Xl, lr, steps, l2=0.0) -> np.ndarray
        Full-batch gradient descent from w = 0. Returns the final weights.

    pairwise_accuracy(w, Xw, Xl) -> float
        Fraction of pairs where the chosen response scores strictly higher.

    fit_scores_from_comparisons(n_items, comparisons, lr, steps, l2=0.0) -> np.ndarray
        Tabular Bradley-Terry: one free score per item. comparisons is a list of
        (winner_index, loser_index), repeated pairs allowed. Minimise the mean pairwise logistic
        loss (plus 0.5 * l2 * |s|^2) by gradient descent from zeros. Return scores centred to
        mean 0 (only differences are identifiable).

HINTS
-----
- With d = (Xw - Xl) @ w, dloss/dw = -mean(sigmoid(-d)[:, None] * (Xw - Xl), axis=0) + l2 * w.
- At w = 0 every pair has d = 0, so the loss is log 2 and each pair's weight sigmoid(0) = 0.5.
- In the tabular version, each comparison pushes the winner up and the loser down by the
  same amount (gradient +-sigmoid(-d)).

WHAT YOU SHOULD LEARN
---------------------
- A reward model is a classifier on score differences: only differences matter, so rewards are
  defined up to an additive constant.
- A reward gap of 1 means ~73% preference, a gap of 2 means ~88%.
- On separable data the unregularised loss keeps shrinking by growing |w|, so you need
  regularisation, early stopping or more data. This is the root of reward over-optimisation.
- The Bradley-Terry fit on pairwise comparisons is the same math as Elo ratings.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def bt_prob(r_w: np.ndarray, r_l: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def bt_loss_grad(w: np.ndarray, Xw: np.ndarray, Xl: np.ndarray, l2: float = 0.0) -> tuple[float, np.ndarray]:
    raise NotImplementedError("TODO")


def train_reward_model(Xw: np.ndarray, Xl: np.ndarray, lr: float, steps: int, l2: float = 0.0) -> np.ndarray:
    raise NotImplementedError("TODO")


def pairwise_accuracy(w: np.ndarray, Xw: np.ndarray, Xl: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def fit_scores_from_comparisons(
    n_items: int, comparisons: list[tuple[int, int]], lr: float, steps: int, l2: float = 0.0
) -> np.ndarray:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # sigmoid
    assert abs(float(sigmoid(np.array(0.0))) - 0.5) < 1e-12
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = sigmoid(np.array([-1000.0, 1000.0]))
    assert v[0] == 0.0 and v[1] == 1.0
    assert abs(float(bt_prob(np.array(1.0), np.array(0.0))) - 0.7310585786) < 1e-9
    assert abs(float(bt_prob(np.array(2.0), np.array(0.0))) - 0.8807970780) < 1e-9
    # shift invariance: adding a constant to both rewards changes nothing
    assert np.allclose(bt_prob(np.array([3.0]) + 10, np.array([1.0]) + 10), bt_prob(np.array([3.0]), np.array([1.0])))

    rng = np.random.default_rng(0)

    # Loss at zero weights is log 2 and the gradient matches finite differences
    Xw = rng.normal(size=(40, 5))
    Xl = rng.normal(size=(40, 5))
    loss0, _ = bt_loss_grad(np.zeros(5), Xw, Xl)
    assert abs(loss0 - np.log(2)) < 1e-12
    w = rng.normal(size=5)
    for l2 in (0.0, 0.3):
        loss, g = bt_loss_grad(w, Xw, Xl, l2)
        for j in range(5):
            e = np.zeros(5); e[j] = 1e-6
            num = (bt_loss_grad(w + e, Xw, Xl, l2)[0] - bt_loss_grad(w - e, Xw, Xl, l2)[0]) / 2e-6
            assert abs(num - g[j]) < 1e-7

    # Synthetic preferences drawn from a true linear reward through the BT model
    D, N = 6, 4000
    w_true = np.array([1.5, -1.0, 0.5, 0.0, 2.0, -0.5])
    A = rng.normal(size=(N, D))
    B = rng.normal(size=(N, D))
    p_a = bt_prob(A @ w_true, B @ w_true)
    a_wins = rng.random(N) < p_a
    Xw = np.where(a_wins[:, None], A, B)
    Xl = np.where(a_wins[:, None], B, A)
    w_hat = train_reward_model(Xw, Xl, lr=0.5, steps=400)
    cos = (w_hat @ w_true) / (np.linalg.norm(w_hat) * np.linalg.norm(w_true))
    assert cos > 0.98
    assert np.linalg.norm(w_hat - w_true) < 0.35  # BT-generated labels: the MLE is near the truth
    acc = pairwise_accuracy(w_hat, Xw, Xl)
    bayes = float(np.mean((A @ w_true > B @ w_true) == a_wins))  # accuracy of the true reward
    assert abs(acc - bayes) < 0.02 and 0.75 < acc < 0.95
    # Training lowered the loss
    assert bt_loss_grad(w_hat, Xw, Xl)[0] < np.log(2) - 0.1

    # Separable data: unregularised weights keep growing, L2 keeps them bounded
    sep_w = rng.normal(size=(50, 3)) + np.array([2.0, 0.0, 0.0])
    sep_l = rng.normal(size=(50, 3)) - np.array([2.0, 0.0, 0.0])
    sep_w[:, 0] = np.abs(sep_w[:, 0]) + 0.5
    sep_l[:, 0] = -np.abs(sep_l[:, 0]) - 0.5
    n_short = np.linalg.norm(train_reward_model(sep_w, sep_l, 0.5, 200))
    n_long = np.linalg.norm(train_reward_model(sep_w, sep_l, 0.5, 2000))
    n_reg = np.linalg.norm(train_reward_model(sep_w, sep_l, 0.5, 2000, l2=0.1))
    assert n_long > n_short * 1.3
    assert n_reg < n_long / 2

    # Tabular Bradley-Terry recovers true score differences from expected win counts
    true = np.array([2.0, 0.0, -2.0])
    comps = []
    for i in range(3):
        for j in range(i + 1, 3):
            p = float(bt_prob(true[i], true[j]))
            n_pair = 400
            wins_i = int(round(n_pair * p))
            comps += [(i, j)] * wins_i + [(j, i)] * (n_pair - wins_i)
    s = fit_scores_from_comparisons(3, comps, lr=2.0, steps=3000)
    assert abs(s.mean()) < 1e-12
    assert np.allclose(s, true, atol=0.1)
    assert list(np.argsort(-s)) == [0, 1, 2]
    # A single comparison: winner above loser, symmetric around zero
    s1 = fit_scores_from_comparisons(2, [(0, 1)], lr=1.0, steps=10)
    assert s1[0] > 0 > s1[1] and abs(s1[0] + s1[1]) < 1e-12


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
