"""Exercise 07 - Little's law and queueing: why latency explodes near saturation.

TASK
----
Implement the formulas and a simulator for a FIFO queue with c identical servers,
Poisson arrivals (rate lam) and exponential service times (rate mu per server).

    littles_law(arrival_rate, mean_time_in_system) -> float
        L = lambda * W, the mean number of requests in the system.
    mm1_metrics(lam, mu) -> dict
        keys rho, L, W, Wq for a single server. rho = lam/mu, W = 1/(mu - lam),
        Wq = rho / (mu - lam), L = lam * W. Raise ValueError if lam >= mu.
    mm1_sojourn_quantile(lam, mu, p) -> float
        In M/M/1 the time in system is exponential with rate (mu - lam), so the p-quantile
        is -ln(1 - p) / (mu - lam).
    erlang_c(c, a) -> float
        Probability that an arriving request must wait in M/M/c, offered load a = lam / mu,
        requires a < c. Compute the sum with a running term so large c does not overflow.
    mmc_mean_wait(lam, mu, c) -> float
        Mean queueing delay Wq = erlang_c(c, lam/mu) / (c * mu - lam).
    simulate_queue(lam, mu, c, n, seed) -> np.ndarray
        Waiting time (not including service) of each of n requests in a FIFO M/M/c queue.
        Draw inter-arrival times and service times with np.random.default_rng(seed).exponential
        (draw all inter-arrivals first, then all services). Keep a heap of server-free times.

HINTS
-----
- Erlang C: with rho = a/c, C = T / (S + T) where T = (a^c / c!) / (1 - rho) and
  S = sum_{k=0}^{c-1} a^k / k!. Build a^k / k! iteratively: term_k = term_{k-1} * a / k.
- Simulation: arrival_i = cumulative sum of inter-arrivals. Pop the earliest-free server,
  start = max(arrival, free_time), wait = start - arrival, push start + service.

WHAT YOU SHOULD LEARN
---------------------
- Little's law holds for any stable system: concurrency = arrival rate x time in system.
  This is how you turn "40 QPS, 8 s per request" into 320 in-flight requests.
- Waiting time grows like rho / (1 - rho): doubling load from 45% to 90% multiplies the
  queue delay by 11, not 2. This is why you do not run an inference fleet at 95%.
- One pooled queue of c servers beats c separate queues at the same utilisation, which is why
  routing to a shared pool (or cache-aware routing with load caps) matters.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


def littles_law(arrival_rate: float, mean_time_in_system: float) -> float:
    """Mean number in system."""
    raise NotImplementedError("TODO")


def mm1_metrics(lam: float, mu: float) -> dict:
    """Closed-form M/M/1 metrics."""
    raise NotImplementedError("TODO")


def mm1_sojourn_quantile(lam: float, mu: float, p: float) -> float:
    """p-quantile of the time in system for M/M/1."""
    raise NotImplementedError("TODO")


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in M/M/c with offered load a."""
    raise NotImplementedError("TODO")


def mmc_mean_wait(lam: float, mu: float, c: int) -> float:
    """Mean queueing delay in M/M/c."""
    raise NotImplementedError("TODO")


def simulate_queue(lam: float, mu: float, c: int, n: int, seed: int) -> np.ndarray:
    """Simulate a FIFO M/M/c queue and return each request's waiting time."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # Little's law: 40 QPS with 8 s in the system means 320 requests in flight
    assert littles_law(40, 8.0) == 320.0

    # M/M/1 closed forms: mu = 10/s, lam = 8/s -> rho 0.8, W = 0.5 s, Wq = 0.4 s, L = 4
    m = mm1_metrics(8.0, 10.0)
    assert abs(m["rho"] - 0.8) < 1e-12 and abs(m["W"] - 0.5) < 1e-12
    assert abs(m["Wq"] - 0.4) < 1e-12 and abs(m["L"] - 4.0) < 1e-12
    assert abs(m["L"] - littles_law(8.0, m["W"])) < 1e-12
    try:
        mm1_metrics(10.0, 10.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # queue delay grows like rho / (1 - rho): 45% -> 90% load is an 11x increase in Wq
    w45 = mm1_metrics(4.5, 10.0)["Wq"]
    w90 = mm1_metrics(9.0, 10.0)["Wq"]
    assert abs(w90 / w45 - 11.0) < 1e-9

    # P99 time in system at rho = 0.8 is ln(100) / 2 = 2.30 s, versus a 0.5 s mean
    q99 = mm1_sojourn_quantile(8.0, 10.0, 0.99)
    assert abs(q99 - math.log(100) / 2.0) < 1e-12
    assert q99 > 4 * m["W"]

    # Erlang C: c = 1 reduces to rho; known value for c = 2, a = 1 is 1/3
    assert abs(erlang_c(1, 0.8) - 0.8) < 1e-12
    assert abs(erlang_c(2, 1.0) - 1.0 / 3.0) < 1e-12
    assert abs(mmc_mean_wait(8.0, 10.0, 1) - m["Wq"]) < 1e-12
    assert 0.0 < erlang_c(200, 180.0) < 1.0                    # no overflow for large c
    try:
        erlang_c(2, 2.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # pooling: 8 servers sharing one queue at 80% load wait far less than 8 separate M/M/1 queues
    pooled = mmc_mean_wait(64.0, 10.0, 8)                      # a = 6.4, rho = 0.8
    separate = mm1_metrics(8.0, 10.0)["Wq"]
    assert pooled < 0.2 * separate

    # simulation agrees with theory (fixed seed, generous tolerance for sampling noise)
    w = simulate_queue(7.0, 10.0, 1, 100000, seed=1)
    assert abs(w.mean() / mm1_metrics(7.0, 10.0)["Wq"] - 1.0) < 0.1
    w8 = simulate_queue(56.0, 10.0, 8, 100000, seed=2)         # a = 5.6, rho = 0.7
    assert abs(w8.mean() / mmc_mean_wait(56.0, 10.0, 8) - 1.0) < 0.1
    assert (w >= 0).all() and (w8 >= 0).all()

    # Little's law on the simulation itself: mean in system = lam * mean sojourn time.
    # Verify through the identity W = Wq + 1/mu on the simulated wait.
    sojourn = w.mean() + 1.0 / 10.0
    assert abs(littles_law(7.0, sojourn) - mm1_metrics(7.0, 10.0)["L"]) / mm1_metrics(7.0, 10.0)["L"] < 0.1

    # tail behaviour: with a stable M/M/1 queue the P99 wait is several times the mean wait
    assert np.percentile(w, 99) > 3 * w.mean()
    # determinism
    assert np.array_equal(simulate_queue(7.0, 10.0, 1, 1000, seed=5), simulate_queue(7.0, 10.0, 1, 1000, seed=5))


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
