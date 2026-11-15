"""Exercise 1 - Parameter and FLOP counter for a decoder-only config.

TASK
----
Given a `Config`, implement (no biases anywhere, RMSNorm with a weight vector only):

    attn_params(cfg)      -> int   # Wq, Wk, Wv, Wo of one layer
    mlp_params(cfg)       -> int   # gated (SwiGLU: 3 matrices) or plain (2 matrices)
    layer_params(cfg)     -> int   # attention + MLP + 2 RMSNorm weight vectors
    embedding_params(cfg) -> int   # token embedding matrix (vocab * d_model)
    total_params(cfg)     -> int   # embeddings + layers + final norm + untied LM head
    non_embedding_params(cfg) -> int  # total minus embedding and untied LM head
    forward_flops_per_token(cfg, ctx_len) -> int
    train_flops_6nd(n_params, n_tokens) -> int
    train_flops_detailed(cfg, n_tokens, ctx_len) -> int

FLOP convention: one multiply-add = 2 FLOPs. Forward FLOPs per token =
2 * (matmul parameters: all layer attention and MLP weights plus the LM head)
+ attention score and value mixing: 4 * n_layers * ctx_len * n_heads * head_dim
(QK^T costs 2*ctx*hd per head, AV costs 2*ctx*hd per head, no causal halving).
The embedding lookup costs no FLOPs. Training = forward + backward, and
backward ~ 2 * forward, so detailed training FLOPs = 3 * forward * n_tokens.

HINTS
-----
- GQA: K and V projections have n_kv_heads * head_dim output columns, Q and O
  have n_heads * head_dim.
- A tied LM head adds no parameters but still costs matmul FLOPs.

WHAT YOU SHOULD LEARN
---------------------
- Where the parameters of a modern LLM actually live (MLP ~ 2/3 of each layer).
- Why 6ND is a good estimate and where it is off (embeddings, attention term).
- That the well-known "8B" model is 8,030,261,248 parameters by pure arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    head_dim_override: int | None = None
    gated_mlp: bool = True
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.head_dim_override or self.d_model // self.n_heads


def attn_params(cfg: Config) -> int:
    """Parameters of one attention block (Wq, Wk, Wv, Wo)."""
    raise NotImplementedError("TODO")


def mlp_params(cfg: Config) -> int:
    """Parameters of one MLP block."""
    raise NotImplementedError("TODO")


def layer_params(cfg: Config) -> int:
    """One decoder layer: attention + MLP + two RMSNorm weight vectors."""
    raise NotImplementedError("TODO")


def embedding_params(cfg: Config) -> int:
    """Token embedding matrix."""
    raise NotImplementedError("TODO")


def total_params(cfg: Config) -> int:
    """Everything: embeddings, layers, final norm, and LM head if untied."""
    raise NotImplementedError("TODO")


def non_embedding_params(cfg: Config) -> int:
    """Total minus the embedding matrix and the untied LM head."""
    raise NotImplementedError("TODO")


def forward_flops_per_token(cfg: Config, ctx_len: int) -> int:
    """Forward FLOPs for one token attending over ctx_len positions."""
    raise NotImplementedError("TODO")


def train_flops_6nd(n_params: int, n_tokens: int) -> int:
    """The rule of thumb: 6 FLOPs per parameter per training token."""
    raise NotImplementedError("TODO")


def train_flops_detailed(cfg: Config, n_tokens: int, ctx_len: int) -> int:
    """Forward + backward (backward = 2x forward) over n_tokens tokens."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    llama8b = Config(vocab=128256, d_model=4096, n_layers=32, n_heads=32,
                     n_kv_heads=8, d_ff=14336)
    assert attn_params(llama8b) == 41_943_040
    assert mlp_params(llama8b) == 176_160_768
    assert layer_params(llama8b) == 218_112_000
    assert total_params(llama8b) == 8_030_261_248
    assert non_embedding_params(llama8b) == 8_030_261_248 - 2 * 128256 * 4096

    # Tying the head removes exactly vocab * d_model parameters.
    tied = Config(**{**llama8b.__dict__, "tied_embeddings": True})
    assert total_params(llama8b) - total_params(tied) == 128256 * 4096
    # ...but the head matmul still costs FLOPs.
    assert forward_flops_per_token(llama8b, 0) == forward_flops_per_token(tied, 0)

    # Full MHA has more attention parameters than GQA by exactly the K/V difference.
    mha = Config(**{**llama8b.__dict__, "n_kv_heads": 32})
    assert attn_params(mha) - attn_params(llama8b) == 2 * 4096 * (32 - 8) * 128
    assert attn_params(mha) == 4 * 4096 * 4096

    # Plain (non-gated) MLP with d_ff = 4d has 8 d^2 parameters.
    gpt = Config(vocab=50000, d_model=1024, n_layers=24, n_heads=16, n_kv_heads=16,
                 d_ff=4096, gated_mlp=False)
    assert mlp_params(gpt) == 8 * 1024 * 1024
    assert attn_params(gpt) == 4 * 1024 * 1024

    # Forward FLOPs at context 0 are 2 * matmul params.
    f0 = forward_flops_per_token(llama8b, 0)
    assert f0 == 2 * (32 * (41_943_040 + 176_160_768) + 128256 * 4096)
    # Attention term grows linearly with context.
    f8k = forward_flops_per_token(llama8b, 8192)
    assert f8k - f0 == 4 * 32 * 8192 * 4096
    assert forward_flops_per_token(llama8b, 16384) - f8k == f8k - f0

    # 6ND is within ~10% of the detailed count at short context.
    n_tok = 15 * 10**12
    approx = train_flops_6nd(total_params(llama8b), n_tok)
    detailed = train_flops_detailed(llama8b, n_tok, 0)
    assert 0.9 < detailed / approx < 1.0
    assert approx == 6 * 8_030_261_248 * n_tok
    # At long context the attention term makes the true cost clearly larger.
    assert train_flops_detailed(llama8b, n_tok, 8192) / approx > 1.1


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
