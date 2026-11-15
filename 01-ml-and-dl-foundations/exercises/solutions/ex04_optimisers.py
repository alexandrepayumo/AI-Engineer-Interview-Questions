"""Exercise 4 - Optimisers: momentum, Adam, AdamW, and a warmup + cosine schedule.

TASK
    All functions are pure: they take arrays and return NEW arrays, no globals.

    sgd_momentum_step(w, g, v, lr, beta=0.9) -> tuple[w_new, v_new]
        v_new = beta * v + g ;  w_new = w - lr * v_new       (beta = 0 is plain SGD)
    adam_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, l2=0.0)
            -> tuple[w_new, m_new, v_new]
        Adam with bias correction. t is the 1-based step count. `l2` is the classic
        L2 penalty: it is ADDED TO THE GRADIENT (g + l2 * w) before the moments.
    adamw_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
            -> tuple[w_new, m_new, v_new]
        AdamW: moments use the raw gradient; decay is applied directly to the weights:
        w_new = w - lr * (m_hat / (sqrt(v_hat) + eps)) - lr * weight_decay * w
    cosine_lr(step, total_steps, peak_lr, warmup_steps, min_ratio=0.1) -> float
        step is 0-based. For step < warmup: peak_lr * (step + 1) / warmup_steps.
        After warmup: cosine decay from peak_lr to peak_lr * min_ratio, reaching the
        floor at step == total_steps (clip progress to [0, 1]).
    run_quadratic(method, steps, lr) -> float
        Minimise f(w) = 0.5 * (100 * w0^2 + 1 * w1^2) from w = [1, 1] and return the
        final f. method is "sgd", "momentum" (beta 0.9) or "adam".

HINTS
    - Bias correction: m_hat = m / (1 - beta1**t), v_hat = v / (1 - beta2**t).
    - At t = 1 with m = v = 0, m_hat = g and v_hat = g**2, so the update is
      lr * g / (|g| + eps), which is about lr * sign(g).
    - Gradient of the quadratic: [100 * w0, 1 * w1].

WHAT YOU SHOULD LEARN
    - Momentum as an EMA of gradients; Adam as a per-parameter step-size normaliser
      (scale invariance, sign-like first step).
    - Why bias correction exists.
    - Exactly why Adam + L2 differs from AdamW: the L2 term passes through the
      1 / sqrt(v_hat) rescaling, the decoupled decay does not.
    - What warmup + cosine looks like as a function of the step number.
"""
from __future__ import annotations

import math

import numpy as np


def sgd_momentum_step(w: np.ndarray, g: np.ndarray, v: np.ndarray, lr: float, beta: float = 0.9):
    v_new = beta * v + g
    return w - lr * v_new, v_new


def adam_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, l2=0.0):
    g = g + l2 * w  # L2 enters the gradient, so it is later divided by sqrt(v_hat)
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), m_new, v_new


def adamw_step(w, g, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    m_new = beta1 * m + (1 - beta1) * g  # raw gradient only
    v_new = beta2 * v + (1 - beta2) * g * g
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * w  # decoupled
    return w_new, m_new, v_new


def cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    floor = peak_lr * min_ratio
    return floor + (peak_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_quadratic(method: str, steps: int, lr: float) -> float:
    curv = np.array([100.0, 1.0])
    w = np.array([1.0, 1.0])
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, steps + 1):
        g = curv * w
        if method == "sgd":
            w, v = sgd_momentum_step(w, g, v, lr, beta=0.0)
        elif method == "momentum":
            w, v = sgd_momentum_step(w, g, v, lr, beta=0.9)
        elif method == "adam":
            w, m, v = adam_step(w, g, m, v, t, lr)
        else:
            raise ValueError(method)
    return float(0.5 * np.sum(curv * w * w))


def _run_tests() -> None:
    # momentum: two steps by hand. g = 1, beta = 0.9, lr = 0.1
    w, v = np.array([1.0]), np.array([0.0])
    w, v = sgd_momentum_step(w, np.array([1.0]), v, 0.1)
    assert np.allclose(v, [1.0]) and np.allclose(w, [0.9])
    w, v = sgd_momentum_step(w, np.array([1.0]), v, 0.1)
    assert np.allclose(v, [1.9]) and np.allclose(w, [0.9 - 0.19])
    # beta = 0 is plain SGD
    w2, _ = sgd_momentum_step(np.array([2.0]), np.array([0.5]), np.array([7.0]), 0.1, beta=0.0)
    assert np.allclose(w2, [2.0 - 0.05])

    # Adam first step: g = 0.5 -> m = 0.05, v = 0.00025, m_hat = 0.5, v_hat = 0.25, update = lr
    z = np.zeros(1)
    w1, m1, v1 = adam_step(np.array([1.0]), np.array([0.5]), z, z, 1, lr=0.01)
    assert np.allclose(m1, [0.05]) and np.allclose(v1, [0.00025])
    assert np.allclose(w1, [1.0 - 0.01], atol=1e-7)

    # scale invariance: multiplying the gradient by 1000 leaves the first step unchanged
    w1b, _, _ = adam_step(np.array([1.0]), np.array([500.0]), z, z, 1, lr=0.01)
    assert np.allclose(w1, w1b, atol=1e-7)

    # sign-like behaviour per coordinate with very different gradient scales
    wv, _, _ = adam_step(np.zeros(2), np.array([1e-3, -1e3]), np.zeros(2), np.zeros(2), 1, lr=0.1)
    assert np.allclose(wv, [-0.1, 0.1], atol=1e-4)

    # AdamW with zero gradient: pure multiplicative decay w * (1 - lr * wd)
    wd, _, _ = adamw_step(np.array([1.0]), z, z, z, 1, lr=0.1, weight_decay=0.5)
    assert np.allclose(wd, [1.0 * (1 - 0.1 * 0.5)])

    # THE key difference. Large task gradient (100), w = 1, step 1.
    big = np.array([100.0])
    plain, _, _ = adam_step(np.array([1.0]), big, z, z, 1, lr=0.1)
    with_l2, _, _ = adam_step(np.array([1.0]), big, z, z, 1, lr=0.1, l2=0.5)
    decoupled, _, _ = adamw_step(np.array([1.0]), big, z, z, 1, lr=0.1, weight_decay=0.5)
    assert abs(with_l2[0] - plain[0]) < 1e-3  # L2 is almost erased by the 1/sqrt(v) scaling
    assert abs((plain[0] - decoupled[0]) - 0.1 * 0.5 * 1.0) < 1e-6  # decay is lr * wd * w

    # hand-checked second step of Adam, g = 1 both times, lr = 0.1
    w_, m_, v_ = adam_step(np.array([0.0]), np.array([1.0]), z, z, 1, lr=0.1)
    w_, m_, v_ = adam_step(w_, np.array([1.0]), m_, v_, 2, lr=0.1)
    assert np.allclose(m_, [0.19]) and np.allclose(v_, [0.001999], atol=1e-9)
    assert np.allclose(w_, [-0.2], atol=1e-6)  # m_hat = v_hat^(1/2) = 1 on both steps

    # schedule
    kw = dict(total_steps=1000, peak_lr=1.0, warmup_steps=100)
    assert abs(cosine_lr(0, **kw) - 0.01) < 1e-12
    assert abs(cosine_lr(49, **kw) - 0.5) < 1e-12
    assert abs(cosine_lr(99, **kw) - 1.0) < 1e-12
    assert abs(cosine_lr(100, **kw) - 1.0) < 1e-12
    assert abs(cosine_lr(550, **kw) - 0.55) < 1e-12  # halfway through decay: (1 + 0.1) / 2
    assert abs(cosine_lr(1000, **kw) - 0.1) < 1e-12
    assert abs(cosine_lr(5000, **kw) - 0.1) < 1e-12  # clipped past the end
    lrs = [cosine_lr(s, **kw) for s in range(100, 1001)]
    assert all(lrs[i + 1] <= lrs[i] + 1e-15 for i in range(len(lrs) - 1))

    # toy problem: curvature 100 vs 1. Stable SGD needs lr < 2/100.
    sgd = run_quadratic("sgd", 100, 0.015)
    mom = run_quadratic("momentum", 100, 0.015)
    adam = run_quadratic("adam", 100, 0.1)
    assert mom < 0.1 * sgd, (sgd, mom)
    assert adam < 1e-1, adam
    assert sgd < 0.5  # it is stable, just slow along the flat direction
    assert run_quadratic("sgd", 20, 0.03) > 1e3  # lr beyond 2/100 diverges


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
