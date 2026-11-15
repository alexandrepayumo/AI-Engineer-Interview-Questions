"""Exercise 04 - DPO loss, gradients and the implicit reward on toy log-probabilities.

TASK
----
Implement (inputs are sequence log-probabilities, i.e. the sum of token log-probs of a response):

    log_sigmoid(x) -> np.ndarray
        Stable log(sigmoid(x)) = -logaddexp(0, -x).

    implicit_reward(logp, ref_logp, beta) -> np.ndarray
        beta * (logp - ref_logp). The "reward" a DPO-trained policy implicitly assigns.

    dpo_loss(pi_w, pi_l, ref_w, ref_l, beta) -> tuple[float, np.ndarray, np.ndarray]
        Arrays of shape (B,): policy and reference log-probs of chosen (w) and rejected (l)
        responses. z = beta * ((pi_w - ref_w) - (pi_l - ref_l)). Loss = mean(-log_sigmoid(z)).
        Returns (loss, d_loss/d_pi_w, d_loss/d_pi_l) with the gradients of the MEAN loss.

    preference_stats(pi_w, pi_l, ref_w, ref_l, beta) -> dict[str, float]
        "accuracy": fraction of pairs with implicit reward of chosen > rejected,
        "margin": mean implicit reward of chosen minus rejected,
        "chosen_reward": mean implicit reward of chosen,
        "rejected_reward": mean implicit reward of rejected.

    softmax(logits) -> np.ndarray
        Stable softmax over a 1-D array.

    train_tabular_dpo(ref_logits, pairs, beta, lr, steps) -> np.ndarray
        A one-prompt "policy" over K candidate responses: pi = softmax(theta), theta initialised
        to ref_logits. pairs is a list of (chosen_index, rejected_index). Full-batch gradient
        descent on the DPO loss over theta (log-prob of response i is theta_i - logsumexp(theta),
        so d logp_i / d theta = onehot(i) - pi). Return the final theta.

HINTS
-----
- dLoss/dz = -sigmoid(-z) per pair, so d/d pi_w = -beta * sigmoid(-z) / B and
  d/d pi_l = +beta * sigmoid(-z) / B.
- At pi = ref, z = 0, loss = log 2 and the gradient weight is exactly 1/2.
- Pairs that are already ranked correctly by a large margin get a vanishing gradient. That is
  the built-in "weight by how wrong the implicit reward is".

WHAT YOU SHOULD LEARN
---------------------
- DPO is a logistic loss on a difference of log-ratios. No reward model, no sampling.
- The reference log-probs enter only as constants: they anchor the policy (the KL constraint).
- The implicit reward is determined only up to a prompt-dependent constant, so only differences
  between responses to the same prompt are meaningful.
- DPO can push the chosen log-prob down as long as the rejected one falls faster: check the
  chosen reward, not just the margin.
"""
from __future__ import annotations

import numpy as np


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def implicit_reward(logp: np.ndarray, ref_logp: np.ndarray, beta: float) -> np.ndarray:
    raise NotImplementedError("TODO")


def dpo_loss(
    pi_w: np.ndarray, pi_l: np.ndarray, ref_w: np.ndarray, ref_l: np.ndarray, beta: float
) -> tuple[float, np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO")


def preference_stats(
    pi_w: np.ndarray, pi_l: np.ndarray, ref_w: np.ndarray, ref_l: np.ndarray, beta: float
) -> dict[str, float]:
    raise NotImplementedError("TODO")


def softmax(logits: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def train_tabular_dpo(
    ref_logits: np.ndarray, pairs: list[tuple[int, int]], beta: float, lr: float, steps: int
) -> np.ndarray:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # log_sigmoid
    assert abs(float(log_sigmoid(np.array(0.0))) + np.log(2)) < 1e-12
    assert abs(float(log_sigmoid(np.array(-1000.0))) + 1000.0) < 1e-9
    assert abs(float(log_sigmoid(np.array(1000.0)))) < 1e-12

    # At the reference the loss is log 2 and the gradient weight is 1/2
    pw = np.array([-10.0, -12.0, -9.0])
    pl = np.array([-11.0, -10.0, -9.5])
    loss, dw, dl = dpo_loss(pw, pl, pw, pl, beta=0.1)
    assert abs(loss - np.log(2)) < 1e-12
    assert np.allclose(dw, -0.1 * 0.5 / 3) and np.allclose(dl, 0.1 * 0.5 / 3)

    # Known value: chosen log-ratio +2, rejected -2, beta 0.5 -> z = 0.5 * 4 = 2
    loss, dw, dl = dpo_loss(np.array([-1.0]), np.array([-5.0]), np.array([-3.0]), np.array([-3.0]), 0.5)
    assert abs(loss - np.log1p(np.exp(-2.0))) < 1e-12
    sg = 1.0 / (1.0 + np.exp(2.0))
    assert abs(dw[0] + 0.5 * sg) < 1e-12 and abs(dl[0] - 0.5 * sg) < 1e-12

    # Finite-difference check of both gradients
    rng = np.random.default_rng(1)
    pw, pl = rng.normal(size=6) - 5, rng.normal(size=6) - 5
    rw, rl = rng.normal(size=6) - 5, rng.normal(size=6) - 5
    beta = 0.3
    _, dw, dl = dpo_loss(pw, pl, rw, rl, beta)
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-6
        nw = (dpo_loss(pw + e, pl, rw, rl, beta)[0] - dpo_loss(pw - e, pl, rw, rl, beta)[0]) / 2e-6
        nl = (dpo_loss(pw, pl + e, rw, rl, beta)[0] - dpo_loss(pw, pl - e, rw, rl, beta)[0]) / 2e-6
        assert abs(nw - dw[i]) < 1e-8 and abs(nl - dl[i]) < 1e-8
    # chosen and rejected gradients are equal and opposite
    assert np.allclose(dw, -dl)

    # Already well separated pairs get almost no gradient; wrongly ranked pairs get a lot
    _, g_good, _ = dpo_loss(np.array([0.0]), np.array([-50.0]), np.array([0.0]), np.array([0.0]), 0.5)
    _, g_bad, _ = dpo_loss(np.array([-50.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.5)
    assert abs(g_good[0]) < 1e-9 and abs(g_bad[0]) > 0.49

    # preference stats
    st = preference_stats(np.array([-1.0, -4.0]), np.array([-3.0, -2.0]), np.array([-2.0, -2.0]), np.array([-2.0, -2.0]), 1.0)
    assert st["accuracy"] == 0.5
    assert abs(st["chosen_reward"] - 0.5 * (1.0 - 2.0)) < 1e-12
    assert abs(st["rejected_reward"] - 0.5 * (-1.0 + 0.0)) < 1e-12
    assert abs(st["margin"] - (st["chosen_reward"] - st["rejected_reward"])) < 1e-12

    # Implicit reward of the KL-optimal policy equals r - beta*log Z, so differences match r.
    beta = 0.5
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([1.0, 0.0, -1.0])
    unnorm = ref * np.exp(r / beta)
    Z = unnorm.sum()
    pi_star = unnorm / Z
    ir = implicit_reward(np.log(pi_star), np.log(ref), beta)
    assert np.allclose(ir, r - beta * np.log(Z))
    assert np.allclose(np.diff(ir), np.diff(r))

    # Tabular DPO: three candidates, preference order 0 > 1 > 2, uniform reference
    ref_logits = np.zeros(3)
    pairs = [(0, 1), (1, 2), (0, 2)]
    theta = train_tabular_dpo(ref_logits, pairs, beta=0.5, lr=0.5, steps=200)
    pi = softmax(theta)
    assert pi[0] > pi[1] > pi[2]
    assert pi[0] > 1 / 3 > pi[2]
    lp, ref_lp = np.log(pi), np.log(softmax(ref_logits))
    w_i, l_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    stats = preference_stats(lp[w_i], lp[l_i], ref_lp[w_i], ref_lp[l_i], 0.5)
    assert stats["accuracy"] == 1.0 and stats["margin"] > 0.5
    # the loss went down compared with the reference
    start = dpo_loss(ref_lp[w_i], ref_lp[l_i], ref_lp[w_i], ref_lp[l_i], 0.5)[0]
    end = dpo_loss(lp[w_i], lp[l_i], ref_lp[w_i], ref_lp[l_i], 0.5)[0]
    assert end < start
    # Contradictory data (0>1 and 1>0 equally often) leaves the policy at the reference
    theta_c = train_tabular_dpo(ref_logits, [(0, 1), (1, 0)], beta=0.5, lr=0.5, steps=100)
    assert np.allclose(softmax(theta_c), 1 / 3, atol=1e-9)
    # The chosen response can lose probability when the rejected one loses more: pairs (1, 2)
    # only, reference favours 1 slightly less than 0.
    theta_d = train_tabular_dpo(np.array([2.0, 0.0, 0.0]), [(1, 2)], beta=0.5, lr=0.5, steps=50)
    pi_d, pi_ref = softmax(theta_d), softmax(np.array([2.0, 0.0, 0.0]))
    assert pi_d[1] > pi_ref[1] and pi_d[2] < pi_ref[2]
    assert pi_d[0] < pi_ref[0]  # untouched response 0 loses mass because probabilities sum to 1


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
