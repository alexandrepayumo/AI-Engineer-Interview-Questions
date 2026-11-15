"""Exercise 02 - Training memory estimator: full fine-tuning vs LoRA vs QLoRA.

TASK
----
Use decimal gigabytes: GB = 1e9 bytes. Implement:

    llama_shapes(hidden, ffn, layers, kv_dim) -> list[tuple[int, int]]
        (d_out, d_in) of every linear layer LoRA would target, per transformer layer:
        q (hidden x hidden), k and v (kv_dim x hidden), o (hidden x hidden),
        gate and up (ffn x hidden), down (hidden x ffn). Repeat for `layers` layers
        (a list of 7 * layers shapes).

    lora_params(shapes, r) -> int
        Trainable parameters when every matrix gets rank-r factors: sum of r * (d_in + d_out).

    nf4_bytes_per_param(block=64, double_quant=True, block2=256) -> float
        Storage per weight for a 4-bit blockwise format.
        Without double quantisation: 4 bits + 32 bits / block (one fp32 absmax per block).
        With double quantisation: 4 bits + 8 bits / block (absmax stored as 8-bit)
        + 32 bits / (block * block2) (one fp32 constant per block2 absmax values).
        Return bytes (bits / 8).

    training_memory(n_params, mode, n_trainable=0, optimizer_bytes=8.0) -> dict[str, float]
        Bytes, excluding activations. Keys: "weights", "grads", "master", "optimizer", "total".
        Mixed precision: bf16 weights 2 B, bf16 grads 2 B, fp32 master copy 4 B, optimizer states
        `optimizer_bytes` per trained parameter (8 for Adam: fp32 m and v).
        mode "full":   every parameter is trained.
        mode "lora":   frozen base in bf16 (2 B/param, no grads, no master, no optimizer);
                       the n_trainable adapter params get weights 2 + grads 2 + master 4 + optimizer.
        mode "qlora":  like "lora" but the frozen base costs nf4_bytes_per_param() per param.
        Raise ValueError for an unknown mode.

    activation_bytes(batch, seq, hidden, layers, checkpointing) -> float
        Rough bf16 activation memory with fused attention (no seq^2 term):
        34 * seq * batch * hidden bytes per layer without checkpointing.
        With checkpointing: only each layer's input is kept (2 * seq * batch * hidden per layer)
        plus one layer's full activations being recomputed (34 * seq * batch * hidden).

HINTS
-----
- 7e9 params full fine-tune: 16 B/param -> 112 GB. That is the number to reproduce.
- The LoRA saving is not "fewer adapter params in the forward pass": it is that grads, master
  weights and Adam states exist only for the adapter.
- QLoRA saving is on the frozen base: 2 B -> ~0.516 B per parameter.

WHAT YOU SHOULD LEARN
---------------------
- Where each of the 16 bytes per parameter goes, and which ones LoRA and QLoRA remove.
- Real adapter parameter counts for 7B and 70B shapes (tens to hundreds of millions).
- Why activations, not weights, become the limit once the base is 4-bit, and what
  gradient checkpointing buys.
"""
from __future__ import annotations

GB = 1e9


def llama_shapes(hidden: int, ffn: int, layers: int, kv_dim: int) -> list[tuple[int, int]]:
    per_layer = [
        (hidden, hidden),   # q
        (kv_dim, hidden),   # k
        (kv_dim, hidden),   # v
        (hidden, hidden),   # o
        (ffn, hidden),      # gate
        (ffn, hidden),      # up
        (hidden, ffn),      # down
    ]
    return per_layer * layers


def lora_params(shapes: list[tuple[int, int]], r: int) -> int:
    # B is (d_out x r), A is (r x d_in): r * (d_out + d_in) per matrix.
    return sum(r * (d_out + d_in) for d_out, d_in in shapes)


def nf4_bytes_per_param(block: int = 64, double_quant: bool = True, block2: int = 256) -> float:
    if double_quant:
        bits = 4 + 8 / block + 32 / (block * block2)
    else:
        bits = 4 + 32 / block
    return bits / 8


def training_memory(n_params: float, mode: str, n_trainable: float = 0, optimizer_bytes: float = 8.0) -> dict[str, float]:
    if mode == "full":
        weights, grads = 2 * n_params, 2 * n_params
        master, opt = 4 * n_params, optimizer_bytes * n_params
    elif mode in ("lora", "qlora"):
        base = 2 * n_params if mode == "lora" else nf4_bytes_per_param() * n_params
        weights = base + 2 * n_trainable          # frozen base plus bf16 adapter
        grads = 2 * n_trainable                   # adapter only
        master = 4 * n_trainable
        opt = optimizer_bytes * n_trainable
    else:
        raise ValueError(f"unknown mode {mode}")
    return {"weights": weights, "grads": grads, "master": master, "optimizer": opt,
            "total": weights + grads + master + opt}


def activation_bytes(batch: int, seq: int, hidden: int, layers: int, checkpointing: bool) -> float:
    unit = seq * batch * hidden
    if not checkpointing:
        return 34.0 * unit * layers
    return 2.0 * unit * layers + 34.0 * unit


def _run_tests() -> None:
    # Full fine-tuning: 16 bytes per parameter
    m = training_memory(7e9, "full")
    assert m["total"] == 16 * 7e9
    assert m["weights"] == 14e9 and m["grads"] == 14e9 and m["master"] == 28e9 and m["optimizer"] == 56e9
    assert abs(m["total"] / GB - 112.0) < 1e-9
    assert abs(training_memory(70e9, "full")["total"] / GB - 1120.0) < 1e-9
    # 8-bit optimiser states (2 bytes) shrink the optimiser term only
    assert training_memory(7e9, "full", optimizer_bytes=2.0)["total"] == 10 * 7e9

    # Adapter parameter counts
    s7 = llama_shapes(4096, 11008, 32, 4096)
    assert len(s7) == 7 * 32
    assert lora_params(s7, 16) == 39_976_960
    assert lora_params(s7, 32) == 2 * 39_976_960  # linear in r
    s70 = llama_shapes(8192, 28672, 80, 1024)
    assert lora_params(s70, 16) == 207_093_760
    assert lora_params([(100, 50)], 4) == 4 * 150

    # NF4 storage
    assert abs(nf4_bytes_per_param(64, False) * 8 - 4.5) < 1e-12
    assert abs(nf4_bytes_per_param(64, True) * 8 - (4 + 0.125 + 32 / (64 * 256))) < 1e-12
    assert abs(nf4_bytes_per_param(64, True) * 8 - 4.127) < 1e-3
    assert nf4_bytes_per_param(64, True) < nf4_bytes_per_param(64, False)
    assert nf4_bytes_per_param(32, False) > nf4_bytes_per_param(64, False)  # smaller block, more constants

    # LoRA: frozen bf16 base + 16 B per adapter parameter
    t7 = lora_params(s7, 16)
    lo = training_memory(7e9, "lora", t7)
    assert abs(lo["total"] - (14e9 + 16 * t7)) < 1
    assert lo["grads"] == 2 * t7 and lo["optimizer"] == 8 * t7
    assert 14.0 < lo["total"] / GB < 15.0
    # QLoRA 7B fits in a small GPU before activations
    ql = training_memory(7e9, "qlora", t7)
    assert 4.0 < ql["total"] / GB < 4.6
    assert ql["total"] < lo["total"] < m["total"]
    # QLoRA 70B: ~36 GB base + ~3.3 GB adapter state -> fits on 48 GB with room for activations
    t70 = lora_params(s70, 16)
    ql70 = training_memory(70e9, "qlora", t70)
    assert 38.0 < ql70["total"] / GB < 41.0
    assert training_memory(70e9, "lora", t70)["total"] / GB > 140.0
    try:
        training_memory(1e9, "bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Activations: 7B shape, batch 1, seq 4096
    a_plain = activation_bytes(1, 4096, 4096, 32, False)
    a_ckpt = activation_bytes(1, 4096, 4096, 32, True)
    assert abs(a_plain / GB - 34 * 4096 * 4096 * 32 / GB) < 1e-9
    assert abs(a_plain / GB - 18.253611008) < 1e-6
    assert a_ckpt < a_plain / 10
    assert abs(a_ckpt / GB - (2 * 32 + 34) * 4096 * 4096 / GB) < 1e-9
    # Activations scale linearly with batch and sequence length
    assert abs(activation_bytes(4, 4096, 4096, 32, False) / a_plain - 4.0) < 1e-12
    # With a 4-bit base, plain activations at seq 4096 exceed the weight memory itself
    assert a_plain > ql["total"]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
