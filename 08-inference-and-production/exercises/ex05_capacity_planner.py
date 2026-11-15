"""Exercise 05 - A capacity planner: how many GPUs for a QPS target, and what does a token cost?

TASK
----
Implement a first-order planner for a dense model served with continuous batching.
All sizes are in bytes, all rates in the units named.

    kv_capacity_tokens(gpu_mem, n_gpus, weight_bytes, kv_per_token, mem_util=0.9) -> int
        floor((mem_util * gpu_mem * n_gpus - weight_bytes) / kv_per_token).
        Raise ValueError if the weights alone do not fit (result would be <= 0).
    decode_step_s(weight_bytes, kv_bytes, n_gpus, mem_bw, bw_eff) -> float
        (weight_bytes + kv_bytes) / (n_gpus * mem_bw * bw_eff).
    prefill_s(params, tokens, n_gpus, peak_flops, mfu) -> float
        2 * params * tokens / (n_gpus * peak_flops * mfu).
    max_batch_for_itl(weight_bytes, avg_ctx, kv_per_token, n_gpus, mem_bw, bw_eff, itl_slo) -> int
        Largest integer batch whose decode_step_s is <= itl_slo, where each sequence holds
        avg_ctx tokens of KV. May be <= 0 when even batch 1 misses the SLO.
    replica_capacity(cfg) -> dict
        `cfg` is a dict with keys: params, weight_bytes, kv_per_token, n_gpus, gpu_mem, mem_bw,
        peak_flops, bw_eff, mfu, mem_util, input_tokens, output_tokens, itl_slo, max_batch,
        cached_tokens (prompt tokens served from a prefix cache, default 0 if missing).
        Steps:
          kv_tokens = kv_capacity_tokens(...)
          batch_mem = kv_tokens // (input_tokens + output_tokens)   # reserve peak length
          avg_ctx   = input_tokens + output_tokens / 2
          batch_slo = max_batch_for_itl(...)
          batch     = min(batch_mem, batch_slo, max_batch); raise ValueError if batch < 1
          step      = decode_step_s with kv_bytes = batch * avg_ctx * kv_per_token
          pre       = prefill_s for (input_tokens - cached_tokens) tokens
          seconds_per_request = pre + output_tokens * step / batch
          rps       = 1 / seconds_per_request
        Return keys: batch, batch_mem, batch_slo, step_s, prefill_s, seconds_per_request, rps.
    replicas_needed(qps, rps_per_replica, target_util=0.7) -> int
        ceil(qps / (rps_per_replica * target_util)).
    cost_per_million_tokens(fleet_gpu_hourly, tokens_per_hour) -> float
        fleet_gpu_hourly (dollars per hour for the whole fleet) / tokens_per_hour * 1e6.

HINTS
-----
- Prefill time is spent by the whole replica, so it is NOT divided by the decode batch.
  Decode time per request is its share of each step: output_tokens steps, each shared by `batch`.
- Guard against floating point when applying ceil: use math.ceil(x - 1e-9).

WHAT YOU SHOULD LEARN
---------------------
- KV memory, not compute, usually caps the decode batch.
- At 1500 input and 300 output tokens, prefill dominates replica time even though decode takes
  longer for each individual request. Prefix caching therefore moves capacity a lot.
- Provisioning is replicas = qps / (rps_per_replica * target_utilisation), and $/token follows
  directly from the fleet price.
"""
from __future__ import annotations

import math


def kv_capacity_tokens(gpu_mem: float, n_gpus: int, weight_bytes: float, kv_per_token: float, mem_util: float = 0.9) -> int:
    """Number of KV tokens that fit after the weights."""
    raise NotImplementedError("TODO")


def decode_step_s(weight_bytes: float, kv_bytes: float, n_gpus: int, mem_bw: float, bw_eff: float) -> float:
    """Memory-bound decode step: bytes moved over effective aggregate bandwidth."""
    raise NotImplementedError("TODO")


def prefill_s(params: float, tokens: float, n_gpus: int, peak_flops: float, mfu: float) -> float:
    """Compute-bound prefill: 2 FLOPs per parameter per token at a realistic MFU."""
    raise NotImplementedError("TODO")


def max_batch_for_itl(weight_bytes: float, avg_ctx: float, kv_per_token: float, n_gpus: int,
                      mem_bw: float, bw_eff: float, itl_slo: float) -> int:
    """Largest batch whose decode step fits the inter-token-latency SLO."""
    raise NotImplementedError("TODO")


def replica_capacity(cfg: dict) -> dict:
    """Sustainable request rate of one replica."""
    raise NotImplementedError("TODO")


def replicas_needed(qps: float, rps_per_replica: float, target_util: float = 0.7) -> int:
    """Replicas to serve qps while staying at or below target utilisation."""
    raise NotImplementedError("TODO")


def cost_per_million_tokens(fleet_gpu_hourly: float, tokens_per_hour: float) -> float:
    """Dollars per million tokens for a fleet that is billed per hour."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    cfg = dict(
        params=70e9, weight_bytes=70e9, kv_per_token=327680, n_gpus=2,
        gpu_mem=80e9, mem_bw=3.35e12, peak_flops=989e12, bw_eff=0.7, mfu=0.4,
        mem_util=0.9, input_tokens=1500, output_tokens=300, itl_slo=0.050, max_batch=256,
    )

    # KV capacity: 0.9 * 160 GB - 70 GB = 74 GB -> 74e9 / 327680 = 225,830 tokens
    assert kv_capacity_tokens(80e9, 2, 70e9, 327680) == 225830
    try:
        kv_capacity_tokens(80e9, 1, 140e9, 327680)               # FP16 70B on one GPU
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # primitives
    assert abs(decode_step_s(70e9, 0, 2, 3.35e12, 0.7) - 70e9 / (2 * 3.35e12 * 0.7)) < 1e-15
    assert abs(prefill_s(70e9, 1500, 2, 989e12, 0.4) - 0.26542) < 1e-4
    # the SLO bound matches the closed form (about 304 at a 50 ms ITL SLO)
    assert max_batch_for_itl(70e9, 1650, 327680, 2, 3.35e12, 0.7, 0.050) == 304
    assert max_batch_for_itl(70e9, 1650, 327680, 2, 3.35e12, 0.7, 0.010) < 1   # 10 ms is impossible

    # one replica: memory allows 225830 // 1800 = 125 sequences, step about 29.3 ms
    cap = replica_capacity(cfg)
    assert cap["batch_mem"] == 125 and cap["batch"] == 125 and cap["batch_slo"] == 304
    assert abs(cap["step_s"] - 0.029336) < 1e-5
    assert abs(cap["prefill_s"] - 0.26542) < 1e-4
    assert abs(cap["seconds_per_request"] - 0.33583) < 1e-4
    assert abs(cap["rps"] - 2.9777) < 1e-3
    # prefill, not decode, is the bigger share of replica time here
    assert cap["prefill_s"] > 3 * (cfg["output_tokens"] * cap["step_s"] / cap["batch"])

    # fleet for 40 QPS at 70% target utilisation: ceil(40 / (2.9777 * 0.7)) = 20 replicas = 40 GPUs
    n_rep = replicas_needed(40, cap["rps"], 0.7)
    assert n_rep == 20
    assert replicas_needed(2.0, 1.0, 1.0) == 2                    # exact division does not round up

    # cost: 40 GPUs at $2.50/h, 40 QPS * 1800 tokens
    fleet = n_rep * cfg["n_gpus"] * 2.5
    assert fleet == 100.0
    per_m = cost_per_million_tokens(fleet, 40 * 3600 * 1800)
    assert abs(per_m - 0.3858) < 1e-3
    per_m_out = cost_per_million_tokens(fleet, 40 * 3600 * 300)
    assert abs(per_m_out - 2.3148) < 1e-3                         # if all cost is charged to output

    # a tight ITL SLO shrinks the batch and the capacity
    tight = replica_capacity({**cfg, "itl_slo": 0.025})
    assert tight["batch"] < cap["batch"] and tight["rps"] < cap["rps"]
    assert tight["step_s"] <= 0.025 + 1e-12
    try:
        replica_capacity({**cfg, "itl_slo": 0.010})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # prefix caching 1000 of the 1500 input tokens: prefill falls 3x and capacity more than doubles
    cached = replica_capacity({**cfg, "cached_tokens": 1000})
    assert abs(cached["prefill_s"] - cap["prefill_s"] / 3) < 1e-9
    assert cached["rps"] > 2 * cap["rps"]
    assert replicas_needed(40, cached["rps"], 0.7) == 10          # 20 replicas drop to 10

    # FP8 KV cache halves bytes per token and (nearly) doubles the memory-limited batch
    fp8kv = replica_capacity({**cfg, "kv_per_token": 327680 / 2, "itl_slo": 0.5})
    assert fp8kv["batch_mem"] >= 2 * cap["batch_mem"]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
