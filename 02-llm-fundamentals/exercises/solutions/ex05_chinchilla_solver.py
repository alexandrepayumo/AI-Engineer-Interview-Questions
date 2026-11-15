"""Exercise 5 - Chinchilla compute-optimal allocation solver.

TASK
----
Use the parametric loss  L(N, D) = E + A / N**alpha + B / D**beta  with the constants
(approximately those fitted by Hoffmann et al., 2022):

    E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28

and the compute model C = 6 * N * D. Implement:

    loss(N, D)                         -> float
    optimal_allocation(C)              -> (N_opt, D_opt)   closed form
    optimal_allocation_numeric(C, n_grid=4000) -> (N, D)   log-spaced grid search over N
    tokens_per_param(C)                -> float            D_opt / N_opt
    tokens_to_match_loss(N_small, target_loss) -> float    D needed so loss(N_small, D) == target;
                                                           float("inf") if impossible

HINTS
-----
- Substitute D = C / (6N), set dL/dN = 0:
      N_opt = (alpha*A / (beta*B)) ** (1/(alpha+beta)) * (C/6) ** (beta/(alpha+beta))
- The numeric version should search N between C/6/1e14 and C/6/1e6 (so D stays sane).
- tokens_to_match_loss: solve B / D**beta = target - E - A / N**alpha for D.

WHAT YOU SHOULD LEARN
---------------------
- Where "scale parameters and tokens together" comes from, as an actual derivation.
- That the optimal tokens-per-parameter ratio drifts slowly with compute (not a constant 20).
- Why a smaller model needs far more tokens to match a larger one, and that below an
  irreducible floor no amount of data helps (the "overtraining" trade-off).
"""
from __future__ import annotations

import math

import numpy as np

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N: float, D: float) -> float:
    """Parametric loss in nats per token."""
    return E + A / N ** ALPHA + B / D ** BETA


def optimal_allocation(C: float) -> tuple[float, float]:
    """Closed-form compute-optimal (N, D) for a FLOP budget C = 6 N D."""
    G = (ALPHA * A / (BETA * B)) ** (1.0 / (ALPHA + BETA))
    N = G * (C / 6.0) ** (BETA / (ALPHA + BETA))
    return N, C / (6.0 * N)


def optimal_allocation_numeric(C: float, n_grid: int = 4000) -> tuple[float, float]:
    """Grid search over N on a log scale, D = C / (6 N)."""
    Ns = np.logspace(math.log10(C / 6 / 1e14), math.log10(C / 6 / 1e6), n_grid)
    Ds = C / (6.0 * Ns)
    losses = E + A / Ns ** ALPHA + B / Ds ** BETA
    i = int(np.argmin(losses))
    return float(Ns[i]), float(Ds[i])


def tokens_per_param(C: float) -> float:
    """Optimal D / N at budget C."""
    N, D = optimal_allocation(C)
    return D / N


def tokens_to_match_loss(N_small: float, target_loss: float) -> float:
    """Tokens a model of N_small parameters needs to reach target_loss."""
    gap = target_loss - E - A / N_small ** ALPHA   # loss budget left for the data term
    if gap <= 0:
        return float("inf")                        # the model-size floor is already above target
    return (B / gap) ** (1.0 / BETA)


def _run_tests() -> None:
    # Closed form respects the budget and agrees with brute force.
    for C in (1e20, 1e22, 5.76e23, 1e25):
        N, D = optimal_allocation(C)
        assert abs(6 * N * D - C) / C < 1e-12
        Nn, Dn = optimal_allocation_numeric(C)
        assert abs(math.log(Nn / N)) < 0.01, (C, N, Nn)
        # No nearby allocation on the same budget does better.
        base = loss(N, D)
        for f in (0.5, 0.8, 1.25, 2.0):
            assert loss(N * f, C / (6 * N * f)) > base

    # Chinchilla-scale budget (~6 * 70e9 * 1.4e12 = 5.9e23): tens of billions of parameters,
    # and (with these published constants) a tokens-per-parameter ratio far above 20.
    N, D = optimal_allocation(5.76e23)
    assert 2e10 < N < 1.2e11
    assert 30 < D / N < 150
    # The ratio drifts upward slowly with compute: exponent (alpha - beta) / (alpha + beta) ~ 0.097.
    r1, r2 = tokens_per_param(1e21), tokens_per_param(1e25)
    assert r2 > r1
    expected = 10 ** (4 * (ALPHA - BETA) / (ALPHA + BETA))   # 4 decades of compute
    assert abs(r2 / r1 - expected) < 1e-9

    # N grows as C^0.45 and D as C^0.55: 100x compute -> ~8x N and ~12.5x D.
    Na, Da = optimal_allocation(1e22)
    Nb, Db = optimal_allocation(1e24)
    assert 7 < Nb / Na < 9 and 11 < Db / Da < 14

    # Overtraining: a model 4x smaller than optimal needs several times more tokens
    # to reach the same loss, and therefore costs more training compute.
    C = 1e23
    N, D = optimal_allocation(C)
    target = loss(N, D)
    D_small = tokens_to_match_loss(N / 4, target)
    assert abs(loss(N / 4, D_small) - target) < 1e-9
    assert D_small > 4 * D
    assert 6 * (N / 4) * D_small > C
    # Far too small a model can never reach the target loss.
    assert tokens_to_match_loss(N / 1e4, target) == float("inf")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
