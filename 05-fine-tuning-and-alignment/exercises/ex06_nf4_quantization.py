"""Exercise 06 - 4-bit blockwise quantisation: absmax uniform vs NF4-style codebook, with error measurement.

TASK
----
Implement:

    uniform4_codebook() -> np.ndarray
        16 evenly spaced levels from -1 to 1 (np.linspace(-1, 1, 16)).

    nf4_codebook() -> np.ndarray
        The 16 NormalFloat-4 levels (sorted ascending, in [-1, 1]) built from normal quantiles,
        as in the QLoRA paper (offset = 0.9677083):
          positive = NormalDist().inv_cdf(p) for p in np.linspace(offset, 0.5, 9)[:-1]   (8 values)
          negative = -NormalDist().inv_cdf(p) for p in np.linspace(offset, 0.5, 8)[:-1]  (7 values)
          levels = positive + [0.0] + negative, divided by the largest absolute value, sorted.
        (Note the asymmetry: 8 positive levels, 7 negative levels, and an exact zero.)

    quantize_blockwise(x, codebook, block) -> tuple[np.ndarray, np.ndarray]
        x: 1-D float array whose length is a multiple of `block`.
        For every block: absmax = max |x|; normalised = x / absmax (zero blocks: absmax 1);
        index of the nearest codebook level for every element.
        Returns (indices as uint8, absmax per block as float64).

    dequantize_blockwise(indices, absmax, codebook, block) -> np.ndarray
        codebook[indices] * absmax of the block.

    quant_mse(x, x_hat) -> float
        Mean squared error.

    bits_per_param(block, double_quant, block2=256) -> float
        4 + 32 / block, or with double quantisation 4 + 8 / block + 32 / (block * block2).

HINTS
-----
- Reshape to (n_blocks, block) and work on rows. Nearest level: argmin of |normalised[..., None]
  - codebook| over the last axis.
- Weights of a trained network are close to zero-mean normal, so levels placed at normal
  quantiles waste fewer codes than evenly spaced levels.
- One outlier inflates the block's absmax, which squeezes every other value into a few codes.

WHAT YOU SHOULD LEARN
---------------------
- Blockwise absmax scaling and why smaller blocks cost more constants but give less error.
- NF4 beats uniform 4-bit on normally distributed data at equal storage.
- Outliers are the enemy of absmax quantisation, and block size is the control.
- Storage arithmetic: 4.5 bits per weight at block 64, ~4.13 with double quantisation.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def uniform4_codebook() -> np.ndarray:
    raise NotImplementedError("TODO")


def nf4_codebook() -> np.ndarray:
    raise NotImplementedError("TODO")


def quantize_blockwise(x: np.ndarray, codebook: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO")


def dequantize_blockwise(indices: np.ndarray, absmax: np.ndarray, codebook: np.ndarray, block: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def quant_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def bits_per_param(block: int, double_quant: bool, block2: int = 256) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # Codebooks
    u = uniform4_codebook()
    n = nf4_codebook()
    assert len(u) == 16 and len(n) == 16
    assert u[0] == -1.0 and u[-1] == 1.0
    assert np.all(np.diff(n) > 0)
    assert n[0] == -1.0 and n[-1] == 1.0 and 0.0 in n
    assert int((n > 0).sum()) == 8 and int((n < 0).sum()) == 7
    # Published NF4 values (to 4 decimals): -1, -0.6962, ..., 0, 0.0796, ..., 0.7230, 1
    assert abs(n[1] + 0.6962) < 1e-3 and abs(n[8] - 0.0796) < 1e-3 and abs(n[14] - 0.7230) < 1e-3
    # NF4 levels are denser near zero than the uniform levels
    assert np.sort(np.abs(n))[1:4].max() < np.sort(np.abs(u))[0:3].max()

    # Round trip basics
    rng = np.random.default_rng(0)
    x = rng.normal(size=64 * 8)
    idx, am = quantize_blockwise(x, n, 64)
    assert idx.dtype == np.uint8 and idx.shape == x.shape and am.shape == (8,)
    assert idx.max() <= 15
    assert np.allclose(am, np.abs(x.reshape(8, 64)).max(axis=1))
    xh = dequantize_blockwise(idx, am, n, 64)
    assert xh.shape == x.shape
    # The block maximum is always representable exactly (it maps to +-1)
    for b in range(8):
        blk, blk_hat = x.reshape(8, 64)[b], xh.reshape(8, 64)[b]
        j = np.abs(blk).argmax()
        assert abs(blk[j] - blk_hat[j]) < 1e-12
    # Error is bounded by half the largest gap in the codebook times absmax
    max_gap = np.diff(n).max()
    assert np.all(np.abs(x - xh).reshape(8, 64) <= (max_gap / 2) * am[:, None] + 1e-12)
    # Zero blocks survive
    z_idx, z_am = quantize_blockwise(np.zeros(64), n, 64)
    assert np.array_equal(dequantize_blockwise(z_idx, z_am, n, 64), np.zeros(64))
    # Idempotent: quantising the dequantised values changes nothing
    idx2, am2 = quantize_blockwise(xh, n, 64)
    assert np.allclose(dequantize_blockwise(idx2, am2, n, 64), xh)

    # NF4 beats uniform 4-bit on Gaussian data at the same block size
    g = rng.normal(size=64 * 2000)
    mse_nf4 = quant_mse(g, dequantize_blockwise(*quantize_blockwise(g, n, 64), n, 64))
    mse_uni = quant_mse(g, dequantize_blockwise(*quantize_blockwise(g, u, 64), u, 64))
    assert mse_nf4 < mse_uni
    assert mse_nf4 < 0.009 < 0.0095 < mse_uni  # ~0.0085 vs ~0.0102 on N(0,1)

    # Smaller blocks: lower error but more constants
    mse_by_block = {}
    for blk in (16, 64, 256, 1024):
        mse_by_block[blk] = quant_mse(g, dequantize_blockwise(*quantize_blockwise(g, n, blk), n, blk))
    assert mse_by_block[16] < mse_by_block[64] < mse_by_block[256] < mse_by_block[1024]
    assert bits_per_param(16, False) > bits_per_param(64, False) > bits_per_param(256, False)

    # Outliers: one huge value ruins its block but not the others
    h = g[:64 * 20].copy()
    clean = quant_mse(h, dequantize_blockwise(*quantize_blockwise(h, n, 64), n, 64))
    h_out = h.copy()
    h_out[5] = 50.0
    hh = dequantize_blockwise(*quantize_blockwise(h_out, n, 64), n, 64)
    err_blocks = ((h_out - hh) ** 2).reshape(20, 64).mean(axis=1)
    assert err_blocks[0] > 100 * clean         # the block holding the outlier
    assert err_blocks[1:].max() < 3 * clean    # other blocks are unaffected
    # Smaller blocks contain the damage to fewer weights
    h16 = quant_mse(h_out, dequantize_blockwise(*quantize_blockwise(h_out, n, 16), n, 16))
    assert h16 < quant_mse(h_out, hh)

    # Storage arithmetic
    assert abs(bits_per_param(64, False) - 4.5) < 1e-12
    assert abs(bits_per_param(64, True) - 4.127) < 1e-3
    assert bits_per_param(64, True) < bits_per_param(64, False)
    # 7B parameters at ~4.13 bits is ~3.6 GB
    assert abs(7e9 * bits_per_param(64, True) / 8 / 1e9 - 3.61) < 0.01


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
