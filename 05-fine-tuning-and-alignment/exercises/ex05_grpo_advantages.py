"""Exercise 05 - GRPO: group-relative advantages, clipped surrogate and loss aggregation.

TASK
----
Implement:

    group_advantages(rewards, eps=1e-8, normalise_std=True) -> np.ndarray
        rewards: (G,) rewards of G sampled responses to ONE prompt.
        advantage_i = (r_i - mean) / (std + eps) with the population std (ddof=0).
        With normalise_std=False return r_i - mean only (the Dr. GRPO variant).
        A group whose rewards are all equal must give all-zero advantages.

    batch_advantages(rewards, eps=1e-8, normalise_std=True) -> np.ndarray
        rewards: (P, G), one row per prompt. Apply group_advantages to every row.

    useful_group_mask(rewards) -> np.ndarray
        (P,) bool: True for prompts whose group has at least two distinct reward values
        (groups where every sample scores the same carry no gradient; DAPO filters them out).

    clipped_surrogate(logp_new, logp_old, adv, clip_eps) -> np.ndarray
        Per-token PPO objective to MAXIMISE: min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv),
        with ratio = exp(logp_new - logp_old). Arrays have the same shape.

    token_weights(lengths, mode) -> list[np.ndarray]
        The weight each token gets in the final scalar loss for G responses of the given lengths.
        mode "sample": mean over tokens inside each response, then mean over responses
                       (original GRPO): weight = 1 / (G * len_i).
        mode "token":  every token in the group counts equally (DAPO-style): weight = 1 / sum(lengths).
        Raise ValueError for another mode.

    grpo_policy_loss(logp_new, logp_old, advantages, clip_eps, mode) -> float
        logp_new / logp_old: lists of 1-D arrays (one per response, different lengths).
        advantages: (G,) one scalar per response, shared by all its tokens.
        Returns - sum over tokens of weight * surrogate (a loss to minimise).

HINTS
-----
- For binary rewards with success rate p, std = sqrt(p (1 - p)), so a rare success gets
  advantage sqrt((1 - p) / p).
- The min() makes clipping one-sided: it only removes incentive in the direction the
  update already moved far enough.
- token_weights for mode "sample" sum to 1 over the whole group; so do the weights for "token".

WHAT YOU SHOULD LEARN
---------------------
- GRPO replaces the learned value baseline with the group mean. No critic is trained.
- Std normalisation up-weights prompts that are very easy or very hard (small std), which is
  one of the biases Dr. GRPO removes.
- Sample-level averaging gives a token in a short answer more weight than a token in a long one;
  with negative advantages this makes long wrong answers cheap. Token-level averaging fixes that.
- Zero-variance groups waste samples: no advantage means no learning signal.
"""
from __future__ import annotations

import numpy as np


def group_advantages(rewards: np.ndarray, eps: float = 1e-8, normalise_std: bool = True) -> np.ndarray:
    raise NotImplementedError("TODO")


def batch_advantages(rewards: np.ndarray, eps: float = 1e-8, normalise_std: bool = True) -> np.ndarray:
    raise NotImplementedError("TODO")


def useful_group_mask(rewards: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def clipped_surrogate(logp_new: np.ndarray, logp_old: np.ndarray, adv: np.ndarray, clip_eps: float) -> np.ndarray:
    raise NotImplementedError("TODO")


def token_weights(lengths: list[int], mode: str) -> list[np.ndarray]:
    raise NotImplementedError("TODO")


def grpo_policy_loss(
    logp_new: list[np.ndarray], logp_old: list[np.ndarray], advantages: np.ndarray, clip_eps: float, mode: str
) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # One success in four samples
    adv = group_advantages(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(adv.sum()) < 1e-12
    assert abs(adv[0] - np.sqrt(3)) < 1e-6          # sqrt((1-p)/p) with p = 0.25
    assert np.allclose(adv[1:], -1 / np.sqrt(3), atol=1e-6)
    # Dr. GRPO: no std division
    raw = group_advantages(np.array([1.0, 0.0, 0.0, 0.0]), normalise_std=False)
    assert np.allclose(raw, [0.75, -0.25, -0.25, -0.25])
    # Zero-variance group -> zero advantage, no NaN
    assert np.array_equal(group_advantages(np.ones(5)), np.zeros(5))
    assert np.array_equal(group_advantages(np.zeros(5), normalise_std=False), np.zeros(5))
    # Normalised advantages have unit population std when rewards vary
    r = np.array([0.2, 0.9, 0.4, 0.7, 0.1])
    a = group_advantages(r)
    assert abs(a.mean()) < 1e-12 and abs(a.std() - 1.0) < 1e-6
    # Affine invariance: scaling and shifting rewards does not change normalised advantages
    assert np.allclose(group_advantages(3 * r + 10), a, atol=1e-6)
    # ... but it does change the un-normalised ones
    assert not np.allclose(group_advantages(3 * r, normalise_std=False), group_advantages(r, normalise_std=False))

    # Batch version and filtering
    R = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=float)
    A = batch_advantages(R)
    assert A.shape == R.shape
    assert np.array_equal(A[1], np.zeros(4)) and np.array_equal(A[2], np.zeros(4))
    assert np.allclose(A[3], [1, 1, -1, -1], atol=1e-6)
    assert useful_group_mask(R).tolist() == [True, False, False, True]
    # std normalisation makes a rare success (p=0.25) weigh more than a coin-flip group
    assert A[0, 0] > A[3, 0]

    # Clipped surrogate: four regimes
    old = np.zeros(4)
    new = np.log(np.array([1.5, 0.5, 1.5, 0.5]))
    advs = np.array([1.0, 1.0, -1.0, -1.0])
    out = clipped_surrogate(new, old, advs, 0.2)
    # A>0, ratio 1.5 -> clipped to 1.2 ; A>0, ratio 0.5 -> min(0.5, 0.8) = 0.5
    # A<0, ratio 1.5 -> min(-1.5, -1.2) = -1.5 ; A<0, ratio 0.5 -> min(-0.5, -0.8) = -0.8
    assert np.allclose(out, [1.2, 0.5, -1.5, -0.8])
    # On-policy: ratio 1, surrogate equals the advantage
    assert np.allclose(clipped_surrogate(old, old, advs, 0.2), advs)

    # Token weights
    ws = token_weights([1, 99], "sample")
    assert abs(sum(w.sum() for w in ws) - 1.0) < 1e-12
    assert abs(ws[0][0] - 0.5) < 1e-12 and abs(ws[1][0] - 0.5 / 99) < 1e-12
    assert ws[0][0] / ws[1][0] > 90  # a token in the short answer counts ~99x more
    wt = token_weights([1, 99], "token")
    assert abs(sum(w.sum() for w in wt) - 1.0) < 1e-12
    assert abs(wt[0][0] - wt[1][0]) < 1e-12
    try:
        token_weights([1], "bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Loss: on-policy, per-token surrogate equals the advantage, so the loss is -sum(weights*adv)
    lens = [2, 6]
    lp = [np.zeros(n) for n in lens]
    adv2 = np.array([1.0, -1.0])
    assert abs(grpo_policy_loss(lp, lp, adv2, 0.2, "sample") - 0.0) < 1e-12   # (1 - 1) / 2
    assert abs(grpo_policy_loss(lp, lp, adv2, 0.2, "token") - (-(2 * 1 + 6 * -1) / 8)) < 1e-12
    # Length bias: with a wrong (A<0) long answer and a right (A>0) short one, the sample mean
    # penalises the long answer's tokens less per token than the token mean does.
    per_tok_long_sample = token_weights(lens, "sample")[1][0]
    per_tok_long_token = token_weights(lens, "token")[1][0]
    assert per_tok_long_sample < per_tok_long_token


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
