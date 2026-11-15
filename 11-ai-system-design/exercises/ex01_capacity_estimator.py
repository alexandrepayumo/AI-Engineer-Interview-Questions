"""Exercise 01 - Back-of-envelope capacity and cost estimator for an LLM product.

TASK
----
Implement the following (all pure functions, no I/O):

    @dataclass(frozen=True) Workload        (given below: dau, requests_per_user_per_day,
                                              input_tokens, output_tokens, peak_to_avg)

    avg_qps(w) -> float
        dau * requests_per_user_per_day / 86400
    peak_qps(w) -> float
        avg_qps(w) * w.peak_to_avg
    daily_tokens(w) -> tuple[float, float]
        (input tokens per day, output tokens per day) summed over all requests
    request_cost(input_tokens, output_tokens, price_in_per_m, price_out_per_m,
                 cached_fraction=0.0, cached_price_per_m=None) -> float
        Dollar cost of one request. Prices are dollars per million tokens.
        cached_fraction of the input tokens are billed at cached_price_per_m, which
        defaults to 10% of price_in_per_m. The rest of the input is billed at full price.
        Raise ValueError if cached_fraction is outside [0, 1].
    monthly_cost(w, per_request_cost, days=30) -> float
        requests per day * per_request_cost * days
    gpus_needed(peak, input_tokens, output_tokens, prefill_tps, decode_tps,
                utilisation=0.6, min_replicas=2) -> int
        GPU-seconds of work arriving per second = peak * input_tokens / prefill_tps
        + peak * output_tokens / decode_tps. Divide by the target utilisation, round up,
        and never return fewer than min_replicas (you need N+1 for failures).
        prefill_tps and decode_tps are aggregate tokens per second per GPU under batching.

HINTS
-----
- 1 day = 86,400 seconds. Peak is a multiple of the average, not the max of a series.
- Use math.ceil. A tiny float error (10.000000001) must not add a whole GPU, so round the
  load to 9 decimals before ceil.
- Output tokens usually cost 3-5x input tokens, and decode is far slower per token than
  prefill, so a short answer can still dominate cost and GPU time.

WHAT YOU SHOULD LEARN
---------------------
- The chain requests -> QPS -> tokens -> GPUs -> dollars that interviewers expect in
  five minutes with stated assumptions.
- Why prompt caching is a large lever when inputs are long and mostly shared.
- Why you size for peak and a target utilisation, not for the average.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Workload:
    dau: int
    requests_per_user_per_day: float
    input_tokens: int
    output_tokens: int
    peak_to_avg: float = 3.0


def avg_qps(w: Workload) -> float:
    raise NotImplementedError("TODO")


def peak_qps(w: Workload) -> float:
    raise NotImplementedError("TODO")


def daily_tokens(w: Workload) -> tuple[float, float]:
    raise NotImplementedError("TODO")


def request_cost(
    input_tokens: int,
    output_tokens: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_fraction: float = 0.0,
    cached_price_per_m: float | None = None,
) -> float:
    raise NotImplementedError("TODO")


def monthly_cost(w: Workload, per_request_cost: float, days: int = 30) -> float:
    raise NotImplementedError("TODO")


def gpus_needed(
    peak: float,
    input_tokens: int,
    output_tokens: int,
    prefill_tps: float,
    decode_tps: float,
    utilisation: float = 0.6,
    min_replicas: int = 2,
) -> int:
    # fraction of one GPU consumed per second of wall-clock time
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    w = Workload(dau=200_000, requests_per_user_per_day=5, input_tokens=3_000, output_tokens=400)
    assert abs(avg_qps(w) - 1_000_000 / 86_400) < 1e-9
    assert abs(avg_qps(w) - 11.574) < 1e-3
    assert abs(peak_qps(w) - 34.722) < 1e-3
    tin, tout = daily_tokens(w)
    assert tin == 3_000_000_000 and tout == 400_000_000

    # frontier-style pricing: $3 in / $15 out per million tokens
    c = request_cost(3_000, 400, 3.0, 15.0)
    assert abs(c - 0.015) < 1e-12
    assert abs(monthly_cost(w, c) - 450_000) < 1e-6

    # small-model pricing: $0.15 in / $0.60 out
    cs = request_cost(3_000, 400, 0.15, 0.60)
    assert abs(cs - 0.00069) < 1e-12
    assert abs(monthly_cost(w, cs) - 20_700) < 1e-6
    assert 21 < c / cs < 22

    # caching: 2,400 of 3,000 input tokens are a shared prefix billed at 10%
    cc = request_cost(3_000, 400, 3.0, 15.0, cached_fraction=0.8)
    # fresh 600*3 + cached 2400*0.3 + out 400*15 = 1800 + 720 + 6000 = 8520 -> $0.00852
    assert abs(cc - 0.00852) < 1e-12
    assert cc < c
    # explicit cached price overrides the default
    cc2 = request_cost(3_000, 400, 3.0, 15.0, cached_fraction=1.0, cached_price_per_m=0.75)
    assert abs(cc2 - (3_000 * 0.75 + 400 * 15) / 1e6) < 1e-12
    # zero cache fraction equals the plain price
    assert request_cost(3_000, 400, 3.0, 15.0, cached_fraction=0.0) == c
    for bad in (-0.1, 1.1):
        try:
            request_cost(10, 10, 1.0, 1.0, cached_fraction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # GPUs: prefill 20k tok/s, decode 2.5k tok/s aggregate per GPU
    peak = peak_qps(w)
    load = peak * 3_000 / 20_000 + peak * 400 / 2_500
    assert abs(load - 10.764) < 1e-3
    assert gpus_needed(peak, 3_000, 400, 20_000, 2_500) == 18  # 10.764 / 0.6 = 17.94
    assert gpus_needed(peak, 3_000, 400, 20_000, 2_500, utilisation=1.0) == 11
    # floor of two replicas for failure tolerance
    assert gpus_needed(0.01, 100, 50, 20_000, 2_500) == 2
    assert gpus_needed(0.01, 100, 50, 20_000, 2_500, min_replicas=3) == 3
    # exact multiples do not round up because of float noise
    assert gpus_needed(1.0, 1000, 0, 1000.0, 1000.0, utilisation=0.5, min_replicas=1) == 2

    # doubling users doubles QPS, tokens and cost but not per-request cost
    w2 = Workload(dau=400_000, requests_per_user_per_day=5, input_tokens=3_000, output_tokens=400)
    assert abs(peak_qps(w2) - 2 * peak) < 1e-9
    assert abs(monthly_cost(w2, c) - 900_000) < 1e-6


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
