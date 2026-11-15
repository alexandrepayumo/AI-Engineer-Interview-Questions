"""Exercise 01 - A roofline calculator for prefill and decode.

TASK
----
Implement a back-of-envelope performance model for a dense transformer on one GPU.

    @dataclass GPU(peak_flops, mem_bw, mem_bytes)        # FLOP/s, bytes/s, bytes
    @dataclass Model(params, n_layers, n_kv_heads, head_dim, weight_bytes)
        # weight_bytes = bytes per parameter (2.0 for FP16/BF16, 1.0 for FP8, 0.5 for INT4)

    kv_bytes_per_token(model, kv_elem_bytes=2.0) -> float
        2 (K and V) * n_layers * n_kv_heads * head_dim * kv_elem_bytes.
    ridge_point(gpu) -> float
        peak_flops / mem_bw, in FLOPs per byte.
    decode_step(model, gpu, batch, ctx, kv_elem_bytes=2.0) -> dict
        One decode step for `batch` sequences that each hold `ctx` cached tokens.
        flops = 2 * params * batch                       (ignore attention FLOPs)
        bytes = params * weight_bytes + batch * ctx * kv_bytes_per_token
        Return keys: flops, bytes, intensity (flops/bytes), t_compute, t_memory,
        time (the max of the two), bound ("compute" if t_compute > t_memory else "memory").
    prefill_time(model, gpu, tokens) -> dict
        flops = 2 * params * tokens, bytes = params * weight_bytes (weights read once,
        ignore KV writes). Return keys: flops, bytes, t_compute, t_memory, time, bound.
    decode_tokens_per_s(model, gpu, batch, ctx, kv_elem_bytes=2.0) -> float
        batch / decode_step time.
    crossover_batch(model, gpu) -> float
        Batch size at which weight-only decode (ctx = 0) has t_compute == t_memory.

HINTS
-----
- t_compute = flops / gpu.peak_flops, t_memory = bytes / gpu.mem_bw.
- Setting 2 * params * B / peak = params * weight_bytes / bw gives a closed form for the
  crossover. Check that it equals ridge_point(gpu) * weight_bytes / 2.

WHAT YOU SHOULD LEARN
---------------------
- Batch-1 decode has intensity ~1 FLOP/byte at FP16, far below the ridge of ~300.
- Decode only becomes compute-bound at batch ~ ridge (FP16 weights). INT4 weights move the
  crossover down 4x (to ~74) because each FLOP is paid for with fewer bytes.
- With a long KV cache the per-sequence KV traffic can exceed the per-sequence compute, so
  attention-heavy decode never leaves the memory-bound regime however large the batch.
- Prefill of a few hundred tokens or more is compute-bound; a 1-token "prefill" is not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    peak_flops: float
    mem_bw: float
    mem_bytes: float


@dataclass(frozen=True)
class Model:
    params: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    weight_bytes: float


def kv_bytes_per_token(model: Model, kv_elem_bytes: float = 2.0) -> float:
    """Bytes of KV cache stored per token."""
    raise NotImplementedError("TODO")


def ridge_point(gpu: GPU) -> float:
    """FLOPs per byte at which the GPU switches from memory-bound to compute-bound."""
    raise NotImplementedError("TODO")


def _bound(t_compute: float, t_memory: float) -> str:
    return "compute" if t_compute > t_memory else "memory"


def decode_step(model: Model, gpu: GPU, batch: int, ctx: int, kv_elem_bytes: float = 2.0) -> dict:
    """Cost of one decode step for `batch` sequences with `ctx` cached tokens each."""
    raise NotImplementedError("TODO")


def prefill_time(model: Model, gpu: GPU, tokens: int) -> dict:
    """Cost of processing `tokens` prompt tokens in one pass."""
    raise NotImplementedError("TODO")


def decode_tokens_per_s(model: Model, gpu: GPU, batch: int, ctx: int, kv_elem_bytes: float = 2.0) -> float:
    """Aggregate decode throughput across the batch."""
    raise NotImplementedError("TODO")


def crossover_batch(model: Model, gpu: GPU) -> float:
    """Batch size where weight-only decode flips from memory-bound to compute-bound."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    h100 = GPU(peak_flops=989e12, mem_bw=3.35e12, mem_bytes=80e9)
    llama8b = Model(params=8e9, n_layers=32, n_kv_heads=8, head_dim=128, weight_bytes=2.0)

    # KV bytes per token: 2 * 32 * 8 * 128 * 2 = 131072 (128 KiB)
    assert kv_bytes_per_token(llama8b) == 131072
    assert kv_bytes_per_token(llama8b, kv_elem_bytes=1.0) == 65536
    llama70b = Model(70e9, 80, 8, 128, 2.0)
    assert kv_bytes_per_token(llama70b) == 327680

    # ridge point of the H100 numbers is about 295 FLOPs/byte
    assert abs(ridge_point(h100) - 295.22) < 0.01

    # batch-1 decode, empty cache: intensity 1, time = 16 GB / 3.35 TB/s = 4.776 ms
    d1 = decode_step(llama8b, h100, batch=1, ctx=0)
    assert abs(d1["intensity"] - 1.0) < 1e-9
    assert abs(d1["time"] - 16e9 / 3.35e12) < 1e-12
    assert d1["bound"] == "memory"
    assert abs(decode_tokens_per_s(llama8b, h100, 1, 0) - 209.4) < 0.1

    # batching is nearly free while memory-bound: batch 64 costs the same weight read
    d64 = decode_step(llama8b, h100, batch=64, ctx=0)
    assert d64["time"] == d1["time"]
    assert abs(decode_tokens_per_s(llama8b, h100, 64, 0) / decode_tokens_per_s(llama8b, h100, 1, 0) - 64) < 1e-6

    # crossover is ridge * weight_bytes / 2 = 295.2 for FP16 weights
    assert abs(crossover_batch(llama8b, h100) - ridge_point(h100)) < 1e-9
    assert decode_step(llama8b, h100, 295, 0)["bound"] == "memory"
    assert decode_step(llama8b, h100, 296, 0)["bound"] == "compute"

    # INT4 weights (0.5 bytes) quadruple batch-1 intensity and lower the crossover batch 4x
    int4 = Model(8e9, 32, 8, 128, 0.5)
    assert abs(crossover_batch(int4, h100) - 0.25 * ridge_point(h100)) < 1e-9
    assert abs(decode_step(int4, h100, 1, 0)["intensity"] - 4.0) < 1e-9

    # with 2000 cached tokens per sequence, KV traffic per sequence (262 MB, 78 us) exceeds
    # the per-sequence compute (16 GFLOP, 16 us): decode stays memory-bound at any batch
    for b in (1, 32, 295, 5000):
        assert decode_step(llama8b, h100, b, 2000)["bound"] == "memory"
    d32 = decode_step(llama8b, h100, 32, 2000)
    assert abs(d32["bytes"] - (16e9 + 32 * 2000 * 131072)) < 1
    assert abs(d32["time"] - 7.28e-3) < 5e-5
    # halving KV element size (FP8 KV) cuts the KV term in half
    d32q = decode_step(llama8b, h100, 32, 2000, kv_elem_bytes=1.0)
    assert d32q["time"] < d32["time"]
    assert abs(d32q["bytes"] - (16e9 + 32 * 2000 * 65536)) < 1

    # prefill: 2000 tokens = 32 TFLOP -> 32.4 ms compute-bound; 1 token is memory-bound
    p = prefill_time(llama8b, h100, 2000)
    assert p["bound"] == "compute" and abs(p["time"] - 3.2e13 / 989e12) < 1e-9
    assert prefill_time(llama8b, h100, 1)["bound"] == "memory"
    assert prefill_time(llama8b, h100, 295)["bound"] == "memory"
    assert prefill_time(llama8b, h100, 296)["bound"] == "compute"

    # prefill of 2000 tokens costs as much as ~6.8 batch-1 decode steps
    ratio = p["time"] / d1["time"]
    assert 6.5 < ratio < 7.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
