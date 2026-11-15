"""Exercise 05 - Diffusion forward process, DDIM sampling and classifier-free guidance on a 1-D toy.

TASK
----
Data is a 1-D Gaussian N(m, s0^2), for which the optimal noise predictor is known in
closed form, so no network is needed. Implement (numpy only):

    linear_betas(T, beta_start=1e-4, beta_end=0.02) -> np.ndarray
        T values evenly spaced from beta_start to beta_end.

    alpha_bars(betas) -> np.ndarray
        Cumulative product of (1 - beta). Entry t is abar_t.

    cosine_alpha_bars(T, s=0.008) -> np.ndarray
        f(u) = cos(((u / T + s) / (1 + s)) * pi / 2) ** 2. Return f(t) / f(0) for
        t = 1..T (length T), clipped below at 1e-5.

    snr(abar) -> np.ndarray
        Signal-to-noise ratio abar / (1 - abar).

    q_sample(x0, t, eps, abar) -> np.ndarray
        Forward process in one jump: sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * eps.

    gaussian_eps_star(xt, abar_t, m, s0) -> np.ndarray
        Optimal noise prediction E[eps | x_t] when x0 ~ N(m, s0^2). Marginally
        x_t ~ N(sqrt(abar_t) * m, abar_t * s0^2 + 1 - abar_t) and
        E[eps | x_t] = sqrt(1 - abar_t) * (x_t - sqrt(abar_t) * m) / var_t.

    ddim_sample(x_T, abars, eps_fn, steps) -> np.ndarray
        Deterministic DDIM (eta = 0). Timesteps ts = np.linspace(len(abars) - 1, 0, steps)
        rounded to int, descending. At each t with next timestep t' (abar' = 1.0 after the
        last step): eps = eps_fn(x, t); x0_hat = (x - sqrt(1 - abar_t) * eps) / sqrt(abar_t);
        x = sqrt(abar') * x0_hat + sqrt(1 - abar') * eps. Return the final x.

    cfg_combine(eps_uncond, eps_cond, scale) -> np.ndarray
        eps_uncond + scale * (eps_cond - eps_uncond).

HINTS
-----
- Use abar_t = prod(1 - beta_i). With the linear schedule abar_T is about 4e-5, so x_T is
  almost exactly N(0, 1) and sampling can start from standard normal noise.
- For eps_fn pass a lambda or closure of gaussian_eps_star with fixed m and s0.
- scale = 1 recovers the conditional prediction, scale = 0 the unconditional one.

WHAT YOU SHOULD LEARN
---------------------
- The forward process needs no loop: you can jump to any t directly.
- What a noise schedule is, and how linear and cosine schedules differ in SNR.
- That a denoiser trained for noise prediction defines a sampler, and that DDIM trades
  steps for accuracy deterministically.
- What guidance does to samples: sharper (lower variance), pushed away from the
  unconditional mean.
"""
from __future__ import annotations

import numpy as np


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    return np.linspace(beta_start, beta_end, T)


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 - betas)


def cosine_alpha_bars(T: int, s: float = 0.008) -> np.ndarray:
    def f(u):
        return np.cos(((u / T + s) / (1 + s)) * np.pi / 2) ** 2
    t = np.arange(1, T + 1)
    return np.maximum(f(t) / f(0), 1e-5)


def snr(abar: np.ndarray) -> np.ndarray:
    return abar / (1.0 - abar)


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, abar: np.ndarray) -> np.ndarray:
    return np.sqrt(abar[t]) * x0 + np.sqrt(1.0 - abar[t]) * eps


def gaussian_eps_star(xt: np.ndarray, abar_t: float, m: float, s0: float) -> np.ndarray:
    var_t = abar_t * s0 ** 2 + 1.0 - abar_t            # variance of the noised marginal
    return np.sqrt(1.0 - abar_t) * (xt - np.sqrt(abar_t) * m) / var_t


def ddim_sample(x_T: np.ndarray, abars: np.ndarray, eps_fn, steps: int) -> np.ndarray:
    ts = np.rint(np.linspace(len(abars) - 1, 0, steps)).astype(int)
    x = x_T
    for i, t in enumerate(ts):
        ab = abars[t]
        ab_next = abars[ts[i + 1]] if i + 1 < len(ts) else 1.0
        eps = eps_fn(x, int(t))
        x0_hat = (x - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)   # predicted clean sample
        x = np.sqrt(ab_next) * x0_hat + np.sqrt(1.0 - ab_next) * eps  # re-noise to the next level
    return x


def cfg_combine(eps_uncond: np.ndarray, eps_cond: np.ndarray, scale: float) -> np.ndarray:
    return eps_uncond + scale * (eps_cond - eps_uncond)


def _run_tests() -> None:
    T = 1000
    betas = linear_betas(T)
    ab = alpha_bars(betas)
    assert betas.shape == (T,) and abs(betas[0] - 1e-4) < 1e-12 and abs(betas[-1] - 0.02) < 1e-12
    assert np.all(np.diff(ab) < 0) and 0.99 < ab[0] < 1.0
    assert ab[-1] < 1e-4                                # almost pure noise at the end
    assert abs(ab[0] - (1 - 1e-4)) < 1e-12

    # SNR falls monotonically, and the cosine schedule keeps more signal in the middle
    assert np.all(np.diff(snr(ab)) < 0)
    cab = cosine_alpha_bars(T)
    assert cab.shape == (T,) and np.all(np.diff(cab) <= 0) and cab[-1] <= 1e-4
    assert cab[T // 2] > ab[T // 2]                     # cosine at t = 500 is much cleaner
    assert abs(cab[T // 2] - 0.4914) < 0.01             # cos^2((0.5 + s) / (1 + s) * pi / 2) / f(0)

    # forward process statistics for x0 ~ N(3, 0.5^2)
    rng = np.random.default_rng(0)
    m, s0 = 3.0, 0.5
    x0 = m + s0 * rng.normal(size=200_000)
    eps = rng.normal(size=x0.shape)
    for t in (0, 100, 500, 999):
        xt = q_sample(x0, t, eps, ab)
        assert abs(xt.mean() - np.sqrt(ab[t]) * m) < 0.01
        assert abs(xt.var() - (ab[t] * s0 ** 2 + 1 - ab[t])) < 0.01
    # t = 0 is almost the clean sample, t = 999 is almost pure noise
    assert np.abs(q_sample(x0, 0, eps, ab) - x0).max() < 0.1
    assert abs(q_sample(x0, 999, eps, ab).mean()) < 0.02

    # optimal eps predictor: check against the definition E[eps | xt] by regression on samples
    t = 300
    xt = q_sample(x0, t, eps, ab)
    pred = gaussian_eps_star(xt, ab[t], m, s0)
    assert np.mean((pred - eps) ** 2) < np.mean(eps ** 2)       # beats predicting zero
    assert abs(np.mean(pred * (xt - xt.mean()))
               - np.mean(eps * (xt - xt.mean()))) < 0.01          # same covariance with xt
    # pure noise input (abar -> 0) means eps = x_t
    assert np.allclose(gaussian_eps_star(np.array([0.7]), 1e-12, m, s0), 0.7, atol=1e-6)

    # DDIM sampling recovers the data distribution from standard normal noise
    x_T = rng.normal(size=20_000)
    eps_fn = lambda x, t: gaussian_eps_star(x, ab[t], m, s0)
    samples = ddim_sample(x_T, ab, eps_fn, 50)
    assert abs(samples.mean() - m) < 0.03 and abs(samples.std() - s0) < 0.05
    # very few steps keep the mean but lose spread: discretisation error shrinks variance
    coarse = ddim_sample(x_T, ab, eps_fn, 8)
    assert abs(coarse.mean() - m) < 0.05 and coarse.std() < samples.std() < s0 + 0.01
    # deterministic: same noise in, same sample out
    assert np.array_equal(ddim_sample(x_T[:5], ab, eps_fn, 20), ddim_sample(x_T[:5], ab, eps_fn, 20))

    # classifier-free guidance
    u, c = np.array([0.0, 1.0]), np.array([1.0, 3.0])
    assert np.array_equal(cfg_combine(u, c, 0.0), u)
    assert np.array_equal(cfg_combine(u, c, 1.0), c)
    assert np.array_equal(cfg_combine(u, c, 2.0), np.array([2.0, 5.0]))

    # unconditional model: N(0, 1). Conditional model: N(2, 0.3^2).
    def guided(scale, steps=50):
        def fn(x, t):
            return cfg_combine(gaussian_eps_star(x, ab[t], 0.0, 1.0),
                               gaussian_eps_star(x, ab[t], 2.0, 0.3), scale)
        return ddim_sample(x_T, ab, fn, steps)

    g0, g1, g3 = guided(0.0), guided(1.0), guided(3.0)
    assert abs(g0.mean()) < 0.05 and abs(g0.std() - 1.0) < 0.05            # scale 0: unconditional
    assert abs(g1.mean() - 2.0) < 0.03 and abs(g1.std() - 0.3) < 0.03      # scale 1: conditional
    assert g3.mean() > g1.mean() + 0.05                                    # pushed away from the unconditional mean
    assert g3.std() < g1.std()                                             # and sharper: less diversity


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
