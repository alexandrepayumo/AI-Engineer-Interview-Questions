"""Exercise 03 - Quantisation: symmetric, asymmetric and group-wise, with error and memory math.

TASK
----
Implement (numpy arrays in, numpy arrays out):

    quantize_sym_int8(x) -> (q, scale)
        scale = max|x| / 127 (use 1.0 if x is all zeros), q = clip(round(x / scale), -127, 127)
        as int8. Per-tensor, zero maps exactly to zero.
    dequantize_sym(q, scale) -> float64 array
    quantize_asym_uint8(x) -> (q, scale, zero_point)
        scale = (max - min) / 255 (1.0 if max == min), zero_point = clip(round(-min / scale), 0, 255),
        q = clip(round(x / scale) + zero_point, 0, 255) as uint8.
    dequantize_asym(q, scale, zero_point) -> float64 array
    quantize_groupwise_int4(w, group_size) -> (q, scales)
        w is 1-D with len divisible by group_size. For each group, scale = max|g| / 7
        (1.0 if the group is all zeros), q = clip(round(w / scale), -8, 7) stored in int8.
        scales has one entry per group.
    dequantize_groupwise(q, scales, group_size) -> float64 array
    mse(a, b) -> float
    bits_per_weight(weight_bits, group_size, scale_bits=16) -> float
        weight_bits + scale_bits / group_size.
    weight_memory_gb(params, bits) -> float
        params * bits / 8 / 1e9.
    int8_matmul(x, w) -> float64 array
        W8A8: quantise x (m,k) and w (k,n) per tensor with quantize_sym_int8, multiply the
        integer matrices with int32 accumulation, then rescale by both scales.

HINTS
-----
- np.round rounds half to even, which is fine here.
- Cast to int32 before the integer matmul or int8 products will overflow.
- For the group-wise version reshape to (n_groups, group_size) and use axis=1.

WHAT YOU SHOULD LEARN
---------------------
- Symmetric quantisation wastes half the levels on one-sided data such as ReLU outputs, and
  asymmetric quantisation fixes that with a zero point.
- One outlier inflates a per-tensor scale and crushes the resolution of every other weight.
  Small groups contain the damage, at a cost of extra scale bits per weight.
- "4-bit" really means ~4.25 to 4.5 bits per weight once scales are counted.
- Activation outliers hurt W8A8 much more than weight-only quantisation, which is why
  methods such as AWQ and SmoothQuant exist.
"""
from __future__ import annotations

import numpy as np


def quantize_sym_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8."""
    raise NotImplementedError("TODO")


def dequantize_sym(q: np.ndarray, scale: float) -> np.ndarray:
    """Back to float."""
    raise NotImplementedError("TODO")


def quantize_asym_uint8(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Per-tensor asymmetric UINT8 with a zero point."""
    raise NotImplementedError("TODO")


def dequantize_asym(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Back to float."""
    raise NotImplementedError("TODO")


def quantize_groupwise_int4(w: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise symmetric INT4 (values stored in int8, range -8..7)."""
    raise NotImplementedError("TODO")


def dequantize_groupwise(q: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Back to float."""
    raise NotImplementedError("TODO")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    raise NotImplementedError("TODO")


def bits_per_weight(weight_bits: int, group_size: int, scale_bits: int = 16) -> float:
    """Effective storage per weight including one scale per group."""
    raise NotImplementedError("TODO")


def weight_memory_gb(params: float, bits: float) -> float:
    """Weight storage in decimal gigabytes."""
    raise NotImplementedError("TODO")


def int8_matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """W8A8 matmul: integer products, float rescale."""
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # symmetric INT8: round trip error is at most half a step; zero stays zero
    x = rng.normal(size=1000)
    q, s = quantize_sym_int8(x)
    assert q.dtype == np.int8 and abs(s - np.max(np.abs(x)) / 127) < 1e-12
    assert np.max(np.abs(dequantize_sym(q, s) - x)) <= s / 2 + 1e-12
    q0, s0 = quantize_sym_int8(np.zeros(4))
    assert s0 == 1.0 and np.all(q0 == 0)
    assert quantize_sym_int8(np.array([0.0, 1.0, -1.0]))[0].tolist() == [0, 127, -127]

    # asymmetric UINT8 on one-sided data (ReLU-like): about 4x lower MSE than symmetric
    # (256 levels over [0, 6] versus 128 usable levels)
    relu = np.abs(rng.normal(size=20000)) * 2.0
    relu = np.clip(relu, 0, 6.0)
    qs, ss = quantize_sym_int8(relu)
    qa, sa, za = quantize_asym_uint8(relu)
    assert qa.dtype == np.uint8 and za == 0                      # min is 0 so zero point is 0
    e_sym = mse(relu, dequantize_sym(qs, ss))
    e_asym = mse(relu, dequantize_asym(qa, sa, za))
    assert e_asym < 0.4 * e_sym
    # asymmetric on data with a negative offset keeps zero exactly representable
    shifted = rng.uniform(-1.0, 5.0, size=500)
    qa, sa, za = quantize_asym_uint8(shifted)
    assert 0 < za < 255
    assert dequantize_asym(np.array([za], dtype=np.uint8), sa, za)[0] == 0.0
    assert np.max(np.abs(dequantize_asym(qa, sa, za) - shifted)) <= sa / 2 + 1e-12
    # degenerate constant input does not divide by zero
    qc, sc, zc = quantize_asym_uint8(np.full(5, 3.0))
    assert np.all(np.isfinite(dequantize_asym(qc, sc, zc)))

    # group-wise INT4 on weights with a few large outliers
    w = rng.normal(size=4096) * 0.02
    w[[7, 1000, 3000]] = [1.5, -1.2, 0.9]
    q4, sc4 = quantize_groupwise_int4(w, group_size=4096)        # one group = per-tensor
    err_tensor = mse(w, dequantize_groupwise(q4, sc4, 4096))
    q4, sc4 = quantize_groupwise_int4(w, group_size=64)
    err_group = mse(w, dequantize_groupwise(q4, sc4, 64))
    assert q4.min() >= -8 and q4.max() <= 7 and sc4.shape == (64,)
    assert err_group < err_tensor / 5                            # outliers are contained to their group
    # smaller groups never hurt on this data
    q4b, sc4b = quantize_groupwise_int4(w, group_size=16)
    assert mse(w, dequantize_groupwise(q4b, sc4b, 16)) <= err_group
    # a zero group is handled
    z, zs = quantize_groupwise_int4(np.zeros(8), 4)
    assert np.all(z == 0) and np.all(zs == 1.0)

    # INT4 is coarser than INT8 on the same data
    w2 = rng.normal(size=4096) * 0.02
    e4 = mse(w2, dequantize_groupwise(*quantize_groupwise_int4(w2, 128), 128))
    e8 = mse(w2, dequantize_sym(*quantize_sym_int8(w2)))
    assert e8 < e4 / 50

    # memory math: effective bits and model sizes
    assert bits_per_weight(4, 128) == 4.125
    assert bits_per_weight(4, 32) == 4.5
    assert bits_per_weight(4, 64) == 4.25
    assert abs(weight_memory_gb(70e9, 16) - 140.0) < 1e-9
    assert abs(weight_memory_gb(70e9, bits_per_weight(4, 128)) - 36.09375) < 1e-9
    assert abs(weight_memory_gb(8e9, 8) - 8.0) < 1e-9

    # W8A8 matmul stays close to the float result on well-behaved data
    x = rng.normal(size=(16, 256))
    wm = rng.normal(size=(256, 64)) * 0.05
    exact = x @ wm
    approx = int8_matmul(x, wm)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.02
    # one activation outlier channel inflates the per-tensor scale and hurts accuracy a lot
    x_out = x.copy()
    x_out[:, 5] *= 60.0
    # measure the damage on the non-outlier channels: drop channel 5 from the product
    mask = np.ones(256, dtype=bool)
    mask[5] = False
    clean_exact = x_out[:, mask] @ wm[mask]
    qx, sx = quantize_sym_int8(x_out)
    qw, sw = quantize_sym_int8(wm)
    clean_q = (qx[:, mask].astype(np.int32) @ qw[mask].astype(np.int32)) * (sx * sw)
    rel_clean = np.linalg.norm(clean_q - clean_exact) / np.linalg.norm(clean_exact)
    assert rel_clean > 10 * rel                                  # normal channels lose resolution


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
