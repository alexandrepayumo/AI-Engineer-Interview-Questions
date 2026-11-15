"""Exercise 07 - KL-penalised reward, the optimal KL-regularised policy and per-token KL estimators.

TASK
----
Implement:

    kl_exact(p, q) -> float
        KL(p || q) = sum p * log(p / q) for categorical distributions (terms with p = 0 are 0).

    kl_estimators(logp, logq) -> dict[str, np.ndarray]
        Per-sample estimators of KL(p || q) from samples x ~ p. logp and logq are the log-probs
        of the sampled items under p (the policy) and q (the reference). With log_r = logq - logp:
          "k1": -log_r                      (unbiased, high variance, can be negative)
          "k2": 0.5 * log_r ** 2            (low variance, biased, always >= 0)
          "k3": exp(log_r) - 1 - log_r      (unbiased, always >= 0)

    shaped_rewards(final_reward, logp, ref_logp, beta) -> np.ndarray
        Per-token rewards for RLHF: -beta * (logp_t - ref_logp_t) at every token, plus
        final_reward added at the last token. logp and ref_logp are 1-D arrays of token log-probs.

    returns_to_go(rewards, gamma=1.0) -> np.ndarray
        G_t = r_t + gamma * G_{t+1}, computed right to left.

    optimal_policy(ref_probs, rewards, beta) -> np.ndarray
        pi*(y) proportional to ref(y) * exp(r(y) / beta), normalised.

    kl_regularised_value(pi, ref_probs, rewards, beta) -> float
        E_pi[r] - beta * KL(pi || ref).

HINTS
-----
- Sum of per-token log-ratios over a sequence is the sequence-level log-ratio, so summing the
  k1 values over tokens estimates the sequence KL.
- k3 is the "control variate" form: k1 plus (exp(log_r) - 1), which has mean zero under p.
- At the optimum, E_pi*[r] - beta * KL equals beta * log sum_y ref(y) exp(r(y)/beta).

WHAT YOU SHOULD LEARN
---------------------
- How the RLHF per-token KL penalty turns into a dense reward signal.
- Why the reference-anchored objective has a closed-form optimum (the starting point of DPO).
- Why implementations use k3 (or k2) rather than raw k1 for the KL term.
- beta controls the trade-off: tiny beta chases reward, huge beta stays at the reference.
"""
from __future__ import annotations

import numpy as np


def kl_exact(p: np.ndarray, q: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def kl_estimators(logp: np.ndarray, logq: np.ndarray) -> dict[str, np.ndarray]:
    raise NotImplementedError("TODO")


def shaped_rewards(final_reward: float, logp: np.ndarray, ref_logp: np.ndarray, beta: float) -> np.ndarray:
    raise NotImplementedError("TODO")


def returns_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    raise NotImplementedError("TODO")


def optimal_policy(ref_probs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    # Work in log space with a max shift so small beta does not overflow.
    raise NotImplementedError("TODO")


def kl_regularised_value(pi: np.ndarray, ref_probs: np.ndarray, rewards: np.ndarray, beta: float) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # Exact KL
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    expected = 0.5 * np.log(0.5 / 0.4) + 0.3 * np.log(0.3 / 0.4) + 0.2 * 0.0
    assert abs(kl_exact(p, q) - expected) < 1e-12
    assert abs(kl_exact(p, p)) < 1e-15
    assert kl_exact(p, q) != kl_exact(q, p)               # asymmetric
    assert abs(kl_exact(np.array([1.0, 0.0]), np.array([0.5, 0.5])) - np.log(2)) < 1e-12

    # Estimators from samples drawn from p
    rng = np.random.default_rng(0)
    samples = rng.choice(3, size=200_000, p=p)
    est = kl_estimators(np.log(p)[samples], np.log(q)[samples])
    true = kl_exact(p, q)
    assert abs(est["k1"].mean() - true) < 2e-3            # unbiased
    assert abs(est["k3"].mean() - true) < 2e-3            # unbiased
    assert abs(est["k2"].mean() - true) < 2e-3            # biased, but close when KL is small
    assert est["k1"].min() < 0                            # k1 can go negative on single samples
    assert est["k2"].min() >= 0 and est["k3"].min() >= 0
    assert est["k3"].std() < est["k1"].std()              # lower variance than k1
    assert est["k2"].std() < est["k1"].std()
    # k2 is biased: with a large gap the mean visibly differs from the true KL
    p2, q2 = np.array([0.95, 0.05]), np.array([0.05, 0.95])
    s2 = rng.choice(2, size=200_000, p=p2)
    e2 = kl_estimators(np.log(p2)[s2], np.log(q2)[s2])
    assert abs(e2["k1"].mean() - kl_exact(p2, q2)) < 0.05
    assert abs(e2["k3"].mean() - kl_exact(p2, q2)) < 0.05
    assert abs(e2["k2"].mean() - kl_exact(p2, q2)) > 1.0
    # Sequence KL is the sum of token k1 values
    tok_logp = np.array([-1.0, -0.5, -2.0])
    tok_ref = np.array([-1.2, -0.4, -1.5])
    assert abs(kl_estimators(tok_logp, tok_ref)["k1"].sum() - (tok_logp.sum() - tok_ref.sum())) < 1e-12

    # Shaped rewards
    r = shaped_rewards(2.0, tok_logp, tok_ref, beta=0.1)
    assert np.allclose(r, [-0.1 * 0.2, -0.1 * (-0.1), -0.1 * (-0.5) + 2.0])
    # At the reference policy only the final reward remains
    r0 = shaped_rewards(1.5, tok_ref, tok_ref, beta=0.5)
    assert np.allclose(r0, [0.0, 0.0, 1.5])
    # Returns: gamma = 1 gives suffix sums, gamma < 1 discounts
    assert np.allclose(returns_to_go(np.array([1.0, 2.0, 3.0])), [6.0, 5.0, 3.0])
    assert np.allclose(returns_to_go(np.array([0.0, 0.0, 4.0]), gamma=0.5), [1.0, 2.0, 4.0])
    # Return at t=0 of the shaped reward = final reward - beta * sequence log-ratio
    g0 = returns_to_go(r)[0]
    assert abs(g0 - (2.0 - 0.1 * (tok_logp.sum() - tok_ref.sum()))) < 1e-12

    # Optimal policy
    ref = np.array([0.5, 0.3, 0.2])
    rew = np.array([0.0, 1.0, 2.0])
    pi_big = optimal_policy(ref, rew, beta=1e6)           # huge beta -> stays at reference
    assert np.allclose(pi_big, ref, atol=1e-4)
    pi_small = optimal_policy(ref, rew, beta=0.01)        # tiny beta -> argmax reward
    assert pi_small[2] > 0.999
    pi_mid = optimal_policy(ref, rew, beta=1.0)
    assert abs(pi_mid.sum() - 1.0) < 1e-12
    manual = ref * np.exp(rew)
    assert np.allclose(pi_mid, manual / manual.sum())
    # pi* maximises the KL-regularised value; its value is beta * log Z
    beta = 1.0
    v_star = kl_regularised_value(pi_mid, ref, rew, beta)
    assert abs(v_star - beta * np.log(np.sum(ref * np.exp(rew / beta)))) < 1e-12
    for _ in range(200):
        pert = np.abs(pi_mid + rng.normal(scale=0.05, size=3))
        pert /= pert.sum()
        assert kl_regularised_value(pert, ref, rew, beta) <= v_star + 1e-12
    # Reward gained and KL paid both increase as beta falls
    kls = [kl_exact(optimal_policy(ref, rew, b), ref) for b in (4.0, 1.0, 0.25)]
    exp_r = [float(np.dot(optimal_policy(ref, rew, b), rew)) for b in (4.0, 1.0, 0.25)]
    assert kls[0] < kls[1] < kls[2] and exp_r[0] < exp_r[1] < exp_r[2]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
