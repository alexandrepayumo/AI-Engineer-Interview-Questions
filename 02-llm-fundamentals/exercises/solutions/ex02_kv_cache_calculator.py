"""Exercise 2 - KV cache memory calculator for MHA / GQA / MQA / sliding window / MLA.

TASK
----
Implement:

    kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem) -> int
        2 (K and V) * layers * kv_heads * head_dim * bytes
    kv_cache_bytes(batch, seq_len, n_layers, n_kv_heads, head_dim, bytes_per_elem) -> int
    kv_bytes_hybrid(seq_len, n_global, n_local, window, n_kv_heads, head_dim, bytes_per_elem) -> int
        one sequence; global layers cache seq_len tokens, local layers cache
        min(seq_len, window) tokens
    mla_bytes_per_token(n_layers, latent_dim, rope_dim, bytes_per_elem) -> int
        MLA caches one latent vector (latent_dim + rope_dim) per layer per token
    max_concurrent_sequences(gpu_bytes, weight_bytes, seq_len, per_token_bytes,
                             reserve_frac=0.1) -> int
        sequences that fit after weights and a reserved fraction of GPU memory
        (activations, fragmentation) are subtracted; never negative
    reduction_factor(n_heads, n_kv_heads) -> float
        cache shrink factor of GQA/MQA relative to MHA

HINTS
-----
- All results are integers in bytes. Use integer arithmetic where possible.
- reserve_frac applies to gpu_bytes, not to what is left after weights.

WHAT YOU SHOULD LEARN
---------------------
- Why a single 128k sequence of a 70B-class model needs ~40 GiB of cache.
- How GQA, MQA, sliding windows and MLA each attack a different factor.
- That serving capacity is limited by cache memory, not by FLOPs.
"""
from __future__ import annotations


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_elem: float) -> int:
    """Cache bytes for one token across all layers."""
    return int(2 * n_layers * n_kv_heads * head_dim * bytes_per_elem)


def kv_cache_bytes(batch: int, seq_len: int, n_layers: int, n_kv_heads: int,
                   head_dim: int, bytes_per_elem: float) -> int:
    """Cache bytes for batch sequences of seq_len tokens each."""
    return batch * seq_len * kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem)


def kv_bytes_hybrid(seq_len: int, n_global: int, n_local: int, window: int,
                    n_kv_heads: int, head_dim: int, bytes_per_elem: float) -> int:
    """One sequence in a stack of full-attention and sliding-window layers."""
    per_layer_token = kv_bytes_per_token(1, n_kv_heads, head_dim, bytes_per_elem)
    return per_layer_token * (n_global * seq_len + n_local * min(seq_len, window))


def mla_bytes_per_token(n_layers: int, latent_dim: int, rope_dim: int, bytes_per_elem: float) -> int:
    """MLA-style cache: one compressed latent plus a shared rope key per layer."""
    return int(n_layers * (latent_dim + rope_dim) * bytes_per_elem)


def max_concurrent_sequences(gpu_bytes: int, weight_bytes: int, seq_len: int,
                             per_token_bytes: int, reserve_frac: float = 0.1) -> int:
    """How many full-length sequences fit in the memory left for cache."""
    free = gpu_bytes * (1.0 - reserve_frac) - weight_bytes
    if free <= 0:
        return 0
    return int(free // (seq_len * per_token_bytes))


def reduction_factor(n_heads: int, n_kv_heads: int) -> float:
    """Cache shrink of GQA/MQA versus MHA."""
    return n_heads / n_kv_heads


def _run_tests() -> None:
    GiB = 1024 ** 3
    # Llama-3-70B shape: 80 layers, 8 KV heads, head_dim 128, fp16.
    per_tok = kv_bytes_per_token(80, 8, 128, 2)
    assert per_tok == 327_680
    assert kv_cache_bytes(1, 131_072, 80, 8, 128, 2) == 40 * GiB
    # Full MHA (64 KV heads) is 8x larger; MQA (1 head) is 8x smaller.
    assert kv_bytes_per_token(80, 64, 128, 2) == 8 * per_tok
    assert kv_bytes_per_token(80, 1, 128, 2) * 8 == per_tok
    assert reduction_factor(64, 8) == 8.0
    assert reduction_factor(64, 1) == 64.0
    # Quantising the cache to 1 byte per element halves it; 0.5 byte quarters it.
    assert kv_bytes_per_token(80, 8, 128, 1) * 2 == per_tok
    assert kv_bytes_per_token(80, 8, 128, 0.5) * 4 == per_tok
    # Batch scales linearly.
    assert kv_cache_bytes(4, 1000, 80, 8, 128, 2) == 4 * 1000 * per_tok

    # Hybrid stack: 6 global + 30 local layers (window 1024), 8 KV heads, hd 128, fp16.
    one_layer_tok = kv_bytes_per_token(1, 8, 128, 2)
    h = kv_bytes_hybrid(100_000, 6, 30, 1024, 8, 128, 2)
    assert h == one_layer_tok * (6 * 100_000 + 30 * 1024)
    full = kv_cache_bytes(1, 100_000, 36, 8, 128, 2)
    assert h < full / 5            # about 5.5x smaller than all-global
    # Short sequences: local layers cannot cache more than the sequence itself.
    assert kv_bytes_hybrid(500, 6, 30, 1024, 8, 128, 2) == kv_cache_bytes(1, 500, 36, 8, 128, 2)

    # MLA (DeepSeek-V3-like): 61 layers, latent 512 + rope 64, fp16.
    mla = mla_bytes_per_token(61, 512, 64, 2)
    assert mla == 70_272
    # Versus a GQA model with the same 61 layers, 8 KV heads of 128: ~3.6x smaller.
    gqa = kv_bytes_per_token(61, 8, 128, 2)
    assert 3.4 < gqa / mla < 3.7

    # Capacity: one 80 GiB GPU, 140 GB of fp16 weights does not fit at all.
    assert max_concurrent_sequences(80 * GiB, 140 * 10**9, 8192, per_tok) == 0
    # Two GPUs (160 GiB) with 140e9 bytes of weights, 10% reserved, 8k context.
    n = max_concurrent_sequences(160 * GiB, 140 * 10**9, 8192, per_tok)
    free = 160 * GiB * 0.9 - 140 * 10**9
    assert n == int(free // (8192 * per_tok))
    assert 3 < n < 10
    # Doubling the context halves capacity (up to rounding).
    n2 = max_concurrent_sequences(160 * GiB, 140 * 10**9, 16384, per_tok)
    assert abs(n - 2 * n2) <= 1


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
