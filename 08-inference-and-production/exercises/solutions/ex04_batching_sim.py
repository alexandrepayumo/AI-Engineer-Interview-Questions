"""Exercise 04 - Static versus continuous batching: a discrete-event simulation.

TASK
----
Simulate a decode-only server. A request is a tuple (arrival_time, output_len) in seconds
and tokens. One decode step with n sequences in the batch takes

    step_time(n, t0, t1) = t0 + t1 * n        # t0 = weight read, t1 = per-sequence cost

and every sequence in the batch emits one token per step. Implement:

    step_time(n, t0, t1) -> float

    simulate_static(requests, max_batch, t0, t1) -> list[tuple[float, float]]
        Static batching: when the server is idle it takes up to max_batch of the requests that
        have already arrived (FIFO) and runs them together until the LONGEST one finishes.
        Finished sequences keep their slot, so every step costs step_time(batch size, ...).
        All requests in the batch are returned when the batch ends. If nothing has arrived,
        jump the clock to the next arrival.

    simulate_continuous(requests, max_batch, t0, t1) -> list[tuple[float, float]]
        Continuous batching: before every step admit waiting, already-arrived requests while
        there is a free slot. A step costs step_time(active count, ...). A request leaves the
        batch as soon as it has emitted its last token. If the batch is empty, jump the clock
        to the next arrival.

    Both return, in the SAME ORDER as `requests`, a tuple (ttft, latency) per request:
        ttft = time of its first token - arrival, latency = time of its last token
        (static: batch end) - arrival.

    summarize(requests, results) -> dict
        keys: throughput (total output tokens / time of the last completion, measured from 0),
        mean_latency, p95_latency (np.percentile), mean_ttft.

HINTS
-----
- requests need not be sorted. Sort indices by arrival and write results by original index.
- Use a small epsilon-free comparison: a request has arrived if arrival <= clock.
- In the continuous loop keep a list of [index, remaining_tokens].

WHAT YOU SHOULD LEARN
---------------------
- Static batching wastes slots on finished sequences and makes short requests wait for the
  longest one in their batch (the convoy effect).
- Continuous batching admits work at token granularity, so slots are never idle while a queue
  exists. The gain grows with output-length variance.
- Latency of short requests improves most, which is why P50 and P95 move more than throughput
  on skewed workloads.
"""
from __future__ import annotations

import numpy as np


def step_time(n: int, t0: float, t1: float) -> float:
    """Seconds for one decode step with n sequences."""
    return t0 + t1 * n


def simulate_static(requests: list[tuple[float, int]], max_batch: int, t0: float, t1: float) -> list[tuple[float, float]]:
    """Static batching. Returns (ttft, latency) per request."""
    order = sorted(range(len(requests)), key=lambda i: requests[i][0])
    results: list[tuple[float, float] | None] = [None] * len(requests)
    clock = 0.0
    pos = 0                                                    # next request in arrival order
    while pos < len(order):
        if requests[order[pos]][0] > clock:
            clock = requests[order[pos]][0]                    # idle: jump to next arrival
        batch = []
        while pos < len(order) and len(batch) < max_batch and requests[order[pos]][0] <= clock:
            batch.append(order[pos])
            pos += 1
        n = len(batch)
        steps = max(requests[i][1] for i in batch)             # runs until the longest finishes
        dt = step_time(n, t0, t1)                              # finished sequences still occupy slots
        first = clock + dt
        end = clock + steps * dt
        for i in batch:
            results[i] = (first - requests[i][0], end - requests[i][0])
        clock = end
    return [r for r in results if r is not None]


def simulate_continuous(requests: list[tuple[float, int]], max_batch: int, t0: float, t1: float) -> list[tuple[float, float]]:
    """Continuous batching. Returns (ttft, latency) per request."""
    order = sorted(range(len(requests)), key=lambda i: requests[i][0])
    first_tok: dict[int, float] = {}
    last_tok: dict[int, float] = {}
    active: list[list[int]] = []                               # [index, remaining tokens]
    clock = 0.0
    pos = 0
    while pos < len(order) or active:
        # admit at token granularity
        while pos < len(order) and len(active) < max_batch and requests[order[pos]][0] <= clock:
            i = order[pos]
            active.append([i, requests[i][1]])
            pos += 1
        if not active:
            clock = requests[order[pos]][0]                    # idle: jump to next arrival
            continue
        clock += step_time(len(active), t0, t1)
        still = []
        for item in active:
            i = item[0]
            first_tok.setdefault(i, clock)
            item[1] -= 1
            if item[1] == 0:
                last_tok[i] = clock                            # leaves the batch immediately
            else:
                still.append(item)
        active = still
    return [(first_tok[i] - requests[i][0], last_tok[i] - requests[i][0]) for i in range(len(requests))]


def summarize(requests: list[tuple[float, int]], results: list[tuple[float, float]]) -> dict:
    """Aggregate metrics for one simulation run."""
    total_tokens = sum(r[1] for r in requests)
    finish = max(requests[i][0] + results[i][1] for i in range(len(requests)))
    lat = np.array([r[1] for r in results])
    ttft = np.array([r[0] for r in results])
    return {
        "throughput": total_tokens / finish,
        "mean_latency": float(lat.mean()),
        "p95_latency": float(np.percentile(lat, 95)),
        "mean_ttft": float(ttft.mean()),
    }


def _run_tests() -> None:
    t0, t1 = 0.020, 0.0005                                     # 20 ms weight read, 0.5 ms per sequence

    # single request: both policies agree. 10 tokens at 20.5 ms each
    one = [(0.0, 10)]
    for sim in (simulate_static, simulate_continuous):
        (ttft, lat), = sim(one, 4, t0, t1)
        assert abs(ttft - 0.0205) < 1e-12 and abs(lat - 0.205) < 1e-12

    # two simultaneous requests with lengths 2 and 10
    two = [(0.0, 2), (0.0, 10)]
    s = simulate_static(two, 2, t0, t1)
    step2 = step_time(2, t0, t1)
    assert abs(s[0][1] - 10 * step2) < 1e-12 and abs(s[1][1] - 10 * step2) < 1e-12   # convoy effect
    c = simulate_continuous(two, 2, t0, t1)
    assert abs(c[0][1] - 2 * step2) < 1e-12                                          # short one leaves early
    assert abs(c[1][1] - (2 * step2 + 8 * step_time(1, t0, t1))) < 1e-12

    # a late arrival joins a running batch in continuous mode but waits in static mode
    late = [(0.0, 20), (0.05, 3)]
    sl = simulate_static(late, 4, t0, t1)
    cl = simulate_continuous(late, 4, t0, t1)
    assert cl[1][0] < 0.04                                                           # first token within ~2 steps
    assert sl[1][0] > 0.3                                                            # waited for the first batch to end
    assert cl[1][1] < sl[1][1]

    # request order is preserved even when input is unsorted; idle gaps are skipped
    gap = [(5.0, 4), (0.0, 4)]
    for sim in (simulate_static, simulate_continuous):
        r = sim(gap, 2, t0, t1)
        assert abs(r[0][1] - 4 * step_time(1, t0, t1)) < 1e-12 and abs(r[1][1] - 4 * step_time(1, t0, t1)) < 1e-12

    # skewed offline workload: 80% short (20 tokens), 20% long (400 tokens), 64 requests
    rng = np.random.default_rng(3)
    lens = rng.choice([20, 400], size=64, p=[0.8, 0.2])
    reqs = [(0.0, int(n)) for n in lens]
    ss = summarize(reqs, simulate_static(reqs, 8, t0, t1))
    sc = summarize(reqs, simulate_continuous(reqs, 8, t0, t1))
    assert sc["throughput"] > 1.5 * ss["throughput"]
    assert sc["mean_latency"] < ss["mean_latency"]
    # neither policy can beat the physical ceiling of a full batch every step
    ceiling = 8 / step_time(8, t0, t1)
    assert sc["throughput"] <= ceiling + 1e-9 and ss["throughput"] < sc["throughput"]

    # with uniform lengths and a full batch the two policies are equivalent
    uni = [(0.0, 50)] * 16
    a = summarize(uni, simulate_static(uni, 8, t0, t1))
    b = summarize(uni, simulate_continuous(uni, 8, t0, t1))
    assert abs(a["throughput"] - b["throughput"]) / a["throughput"] < 1e-9

    # TTFT is never below one step and latency never below ttft
    for sim in (simulate_static, simulate_continuous):
        for ttft, lat in sim(reqs, 8, t0, t1):
            assert ttft >= step_time(1, t0, t1) - 1e-12 and lat >= ttft


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
