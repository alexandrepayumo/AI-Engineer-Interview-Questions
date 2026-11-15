"""Exercise 03 - Vectorised top-p (nucleus) sampling for a batch.

TASK
----
The classic top-p filter handles one row at a time. Serving code handles a batch of
rows with a different p per row. Implement:

    softmax_rows(logits) -> (B, V)
        Stable softmax along the last axis. Rows may contain -inf entries.
    top_p_filter_batch(logits, p) -> (B, V)
        logits is (B, V). p is a float or a (B,) array with 0 < p <= 1.
        Keep the smallest set of highest-probability tokens whose total probability
        is at least p, and set every other logit to -inf. The kept logits are unchanged.
        Always keep at least the single most likely token. Ties are broken towards the
        lower token id. p >= 1 keeps every token that had a finite logit.
        No Python loop over rows.
    sample_batch(logits, p, temperature, rng) -> (B,) int array
        Divide logits by temperature, apply top_p_filter_batch, then draw one token per
        row by inverse-CDF sampling using ONE call to rng.random((B, 1)).
        temperature must be positive.
    nucleus_size(logits, p) -> (B,) int array
        Number of tokens kept per row (a useful debugging view).

HINTS
-----
- Sort descending (use a stable sort on the negated probabilities so ties keep id order).
- A token is kept when the cumulative probability BEFORE it is still below p. That rule
  keeps the token that crosses p, and keeps the top token even when p is tiny.
- Undo the sort with np.put_along_axis or by sorting the sort-order (argsort of argsort).
- Inverse-CDF draw: idx = number of cumulative entries <= u, clipped to V - 1. Using
  "<=" means zero-probability tokens can never be selected, even when u is exactly 0.
- Floating-point cumulative sums can fall just short of 1.0, so treat p >= 1 as a
  special case rather than trusting cumsum.

WHAT YOU SHOULD LEARN
---------------------
- The exact top-p boundary rule and why "cumulative before" is the right comparison.
- Turning a per-row algorithm into sort, cumsum, compare and scatter.
- Filtering with -inf and renormalising, and why temperature is applied first.
- How to test a sampler: against a loop reference, with invariants, and by comparing
  empirical frequencies with the analytic distribution under a fixed seed.
"""
from __future__ import annotations

import numpy as np


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def _keep_mask(logits: np.ndarray, p) -> np.ndarray:
    """Boolean (B, V): which tokens survive the nucleus cut."""
    raise NotImplementedError("TODO")


def top_p_filter_batch(logits: np.ndarray, p) -> np.ndarray:
    raise NotImplementedError("TODO")


def nucleus_size(logits: np.ndarray, p) -> np.ndarray:
    raise NotImplementedError("TODO")


def sample_batch(logits: np.ndarray, p, temperature: float, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError("TODO")


# ----------------------------------------------------------------- loop reference

def _loop_nucleus(row: np.ndarray, p: float) -> set[int]:
    probs = np.exp(row - row.max())
    probs /= probs.sum()
    order = sorted(range(len(row)), key=lambda i: (-probs[i], i))
    kept, total = set(), 0.0
    for i in order:
        if total >= p and kept:
            break
        kept.add(i)
        total += probs[i]
    if p >= 1.0:
        kept = {i for i in range(len(row)) if np.isfinite(row[i])}
    return kept


def _run_tests() -> None:
    rng = np.random.default_rng(0)

    # hand example: probs [0.5, 0.3, 0.1, 0.06, 0.04], p = 0.85 keeps the first three
    probs = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    logits = np.log(probs)[None, :]
    out = top_p_filter_batch(logits, 0.85)
    assert np.array_equal(np.isfinite(out[0]), [True, True, True, False, False])
    assert np.array_equal(out[0, :3], logits[0, :3])           # kept logits untouched
    r = softmax_rows(out)[0]
    assert np.allclose(r[:3], [0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9]) and r[3] == r[4] == 0.0
    # boundary: the token that crosses p is kept, the next one is not
    assert nucleus_size(logits, 0.79)[0] == 2
    assert nucleus_size(logits, 0.81)[0] == 3

    # tiny p still keeps the argmax; p = 1 keeps everything
    assert np.array_equal(nucleus_size(logits, 1e-9), [1])
    assert np.array_equal(nucleus_size(logits, 1.0), [5])
    assert np.array_equal(np.isfinite(top_p_filter_batch(logits, 1.0)[0]), [True] * 5)

    # ties go to the lower id
    tie = np.zeros((1, 4))
    assert np.array_equal(np.isfinite(top_p_filter_batch(tie, 0.5)[0]), [True, True, False, False])

    # -inf logits are never kept, even with p = 1
    masked = np.array([[0.0, -np.inf, 1.0, -np.inf]])
    assert np.array_equal(np.isfinite(top_p_filter_batch(masked, 1.0)[0]), [True, False, True, False])

    # per-row p against the loop reference
    big = rng.normal(size=(40, 30)) * 3
    ps = rng.uniform(0.05, 1.0, size=40)
    ps[:3] = [1.0, 0.999999, 1e-6]
    got = np.isfinite(top_p_filter_batch(big, ps))
    for i in range(40):
        assert set(np.flatnonzero(got[i])) == _loop_nucleus(big[i], ps[i]), i

    # invariants: kept mass >= p, and dropping the smallest kept token would go below p
    pr = softmax_rows(big)
    for i in range(40):
        kept = np.flatnonzero(got[i])
        mass = pr[i, kept].sum()
        assert mass >= min(ps[i], 1.0) - 1e-12
        if len(kept) > 1 and ps[i] < 1.0:
            assert mass - pr[i, kept].min() < ps[i] + 1e-12

    # sampling never leaves the nucleus, and only the argmax survives at tiny p
    lg = rng.normal(size=(500, 12)) * 2
    draws = sample_batch(lg, 0.5, 1.0, np.random.default_rng(1))
    assert draws.shape == (500,)
    nuc = np.isfinite(top_p_filter_batch(lg, 0.5))
    assert nuc[np.arange(500), draws].all()
    greedy = sample_batch(lg, 1e-9, 1.0, np.random.default_rng(2))
    assert np.array_equal(greedy, lg.argmax(axis=-1))

    # empirical frequencies match the renormalised nucleus
    n = 40000
    base = np.log(np.array([0.5, 0.3, 0.1, 0.06, 0.04]))
    tiled = np.tile(base, (n, 1))
    d = sample_batch(tiled, 0.85, 1.0, np.random.default_rng(3))
    freq = np.bincount(d, minlength=5) / n
    assert np.allclose(freq, [0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0.0, 0.0], atol=0.01)
    assert freq[3] == 0.0 and freq[4] == 0.0

    # temperature acts BEFORE the cut: a low temperature shrinks the nucleus
    cold = nucleus_size(logits / 0.5, 0.85)[0]
    hot = nucleus_size(logits / 2.0, 0.85)[0]
    assert cold < 3 < hot
    d_cold = sample_batch(tiled, 0.85, 0.5, np.random.default_rng(4))
    assert set(np.unique(d_cold)) <= {0, 1}
    try:
        sample_batch(tiled[:2], 0.9, 0.0, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("temperature 0 must raise ValueError")

    # same seed, same draws
    a = sample_batch(lg, 0.9, 1.0, np.random.default_rng(7))
    b = sample_batch(lg, 0.9, 1.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
