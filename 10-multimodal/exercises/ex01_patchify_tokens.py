"""Exercise 01 - Patchify an image and count visual tokens for three budgeting schemes.

TASK
----
Implement these functions (numpy and math only):

    num_patches(height, width, patch) -> int
        Patches in a ViT grid when the image is padded up to a multiple of `patch`:
        ceil(height / patch) * ceil(width / patch). ValueError if any argument <= 0.

    patchify(img, patch) -> np.ndarray
        img has shape (H, W, C) and H, W must be divisible by `patch` (else ValueError).
        Return shape (N, patch*patch*C), patches in row-major order (left to right, then
        top to bottom), each patch flattened in (row, col, channel) order.

    unpatchify(patches, height, width, channels, patch) -> np.ndarray
        Exact inverse of patchify.

    attention_pairs(n_tokens) -> int
        Number of query-key pairs in one full self-attention layer: n_tokens ** 2.

    tile_tokens_fixed(width, height) -> int
        A fixed-tile scheme (a sketch of the "512 px tiles" style, not any vendor's
        contract). Steps: (1) if the longer side exceeds 2048, scale both sides down so
        the longer side is 2048; (2) if the shorter side exceeds 768, scale both sides
        down so the shorter side is 768; (3) round scaled sides with int(); (4) tiles =
        ceil(w / 512) * ceil(h / 512); (5) tokens = 85 + 170 * tiles.

    grid_tokens(width, height, patch=28, max_tokens=1568) -> tuple[int, tuple[int, int]]
        Per-patch scheme: tokens = ceil(w / patch) * ceil(h / patch). While tokens exceed
        max_tokens, shrink both sides by 5% (w = max(1, int(w * 0.95)), same for h) and
        recount. Return (tokens, (final_width, final_height)).

    choose_grid(width, height, tile=336, max_tiles=6) -> tuple[int, int]
        Dynamic tiling. Consider every (cols, rows) with 1 <= cols * rows <= max_tiles, in
        ascending order of (cols * rows, cols). Pick the grid whose cols / rows is closest
        to width / height. A candidate replaces the current best if its distance is
        smaller by more than 1e-9, or if the distance is equal (within 1e-9) and the image
        area is greater than 0.5 * tile * tile * cols * rows (so big images take the
        bigger of two equally well-shaped grids, small images do not upsample).

    dynamic_views(width, height, tile=336, max_tiles=6) -> int
        Views sent to the encoder: cols * rows tiles, plus one global thumbnail when
        there is more than one tile.

HINTS
-----
- patchify is one reshape, one transpose and one reshape:
  (H/p, p, W/p, p, C) -> swap axes 1 and 2 -> (H/p * W/p, p*p*C).
- The fixed-tile scheme with a 1024x1024 image gives 4 tiles and 765 tokens.

WHAT YOU SHOULD LEARN
---------------------
- Token count scales with image area, attention cost with its square.
- Why different providers' token counts for the same image differ, and how to estimate
  them before sending anything.
- How dynamic tiling spends tokens on aspect ratio instead of padding to a square.
"""
from __future__ import annotations

import math

import numpy as np


def num_patches(height: int, width: int, patch: int) -> int:
    raise NotImplementedError("TODO")


def patchify(img: np.ndarray, patch: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def unpatchify(patches: np.ndarray, height: int, width: int, channels: int, patch: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def attention_pairs(n_tokens: int) -> int:
    raise NotImplementedError("TODO")


def tile_tokens_fixed(width: int, height: int) -> int:
    raise NotImplementedError("TODO")


def grid_tokens(width: int, height: int, patch: int = 28, max_tokens: int = 1568) -> tuple[int, tuple[int, int]]:
    raise NotImplementedError("TODO")


def choose_grid(width: int, height: int, tile: int = 336, max_tiles: int = 6) -> tuple[int, int]:
    raise NotImplementedError("TODO")


def dynamic_views(width: int, height: int, tile: int = 336, max_tiles: int = 6) -> int:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # patch counts
    assert num_patches(224, 224, 16) == 196
    assert num_patches(336, 336, 14) == 576
    assert num_patches(100, 50, 16) == 7 * 4  # padded up
    try:
        num_patches(0, 10, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    # patchify layout and round trip
    img = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    pt = patchify(img, 2)
    assert pt.shape == (6, 12)
    assert np.array_equal(pt[0], img[:2, :2].reshape(-1))      # top-left patch
    assert np.array_equal(pt[1], img[:2, 2:4].reshape(-1))     # next to the right
    assert np.array_equal(pt[3], img[2:4, :2].reshape(-1))     # first patch of second row
    assert np.array_equal(unpatchify(pt, 4, 6, 3, 2), img)
    try:
        patchify(img, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-divisible size")

    # quadratic attention: doubling the side quadruples tokens and multiplies pairs by 16
    n1, n2 = num_patches(336, 336, 14), num_patches(672, 672, 14)
    assert n2 == 4 * n1 and attention_pairs(n2) == 16 * attention_pairs(n1)

    # fixed-tile scheme (hand-checked values)
    assert tile_tokens_fixed(1024, 1024) == 85 + 170 * 4 == 765
    assert tile_tokens_fixed(2048, 4096) == 85 + 170 * 6 == 1105
    assert tile_tokens_fixed(512, 512) == 255
    assert tile_tokens_fixed(300, 200) == 255          # small images still cost one tile
    assert tile_tokens_fixed(4096, 4096) == 765        # huge images are capped by downscaling

    # per-patch scheme
    assert grid_tokens(560, 280) == (200, (560, 280))
    assert grid_tokens(1000, 1000)[0] == 36 * 36
    t, (w, h) = grid_tokens(1568, 1568)
    assert t <= 1568 and w == h and w < 1568
    t2, (w2, h2) = grid_tokens(3000, 1000)
    assert t2 <= 1568 and abs(w2 / h2 - 3.0) < 0.05    # aspect ratio preserved
    assert grid_tokens(2000, 2000, max_tokens=100)[0] <= 100

    # dynamic tiling
    assert choose_grid(1344, 672) == (2, 1)
    assert choose_grid(672, 1344) == (1, 2)
    assert choose_grid(672, 672) == (2, 2)             # big square image takes the 2x2 grid
    assert choose_grid(200, 200) == (1, 1)             # tiny image is not upsampled to 4 tiles
    assert choose_grid(2016, 672) == (3, 1)
    assert choose_grid(10000, 500, max_tiles=6) == (6, 1)
    assert dynamic_views(1344, 672) == 3               # 2 tiles + thumbnail
    assert dynamic_views(672, 672) == 5
    assert dynamic_views(200, 200) == 1                # a single tile needs no thumbnail
    # tokens for a 576-token-per-view encoder
    assert dynamic_views(672, 672) * 576 == 2880


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
