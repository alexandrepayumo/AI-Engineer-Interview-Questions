"""Exercise 6 - Cross-validation, group leakage, and a bias-variance experiment.

TASK
    kfold_indices(n, k, seed=0) -> list[tuple[np.ndarray, np.ndarray]]
        Shuffle range(n) with np.random.default_rng(seed), split into k folds whose
        sizes differ by at most 1, and return [(train_idx, val_idx), ...]. Every index
        appears in exactly one validation fold.
    group_kfold_indices(groups, k) -> list[tuple[np.ndarray, np.ndarray]]
        Like kfold, but all rows with the same group id land in the same fold (no
        group is split across train and validation). Assign whole groups greedily,
        largest group first, to the currently smallest fold.
    poly_fit(x, y, degree, ridge=0.0) -> np.ndarray
        Least-squares polynomial coefficients (lowest power first) from the normal
        equations (V.T V + ridge * I) c = V.T y, with V[i, j] = x[i] ** j.
        Use np.linalg.solve. (Penalising the intercept too is fine here.)
    poly_predict(coeffs, x) -> np.ndarray
    cross_val_mse(x, y, degree, k=5, seed=0, ridge=0.0) -> float
        Mean over folds of the validation MSE.
    bias_variance(degree, n_train=30, n_trials=300, noise=0.3, seed=0)
            -> tuple[float, float, float]
        True function f(x) = sin(pi * x). Each trial draws n_train points x ~ U(-1, 1),
        y = f(x) + N(0, noise^2), fits a polynomial, and predicts on a fixed grid
        np.linspace(-1, 1, 101). Return (bias_sq, variance, expected_test_mse) where
        bias_sq = mean over the grid of (mean prediction - f)^2,
        variance = mean over the grid of the variance of predictions across trials
        (ddof = 0), and expected_test_mse = bias_sq + variance + noise^2.

HINTS
    - np.array_split gives fold sizes that differ by at most one.
    - Seed the generator ONCE per bias_variance call and reuse it for all trials.
    - bias_sq + variance equals the average squared error of the predictions against
      the noise-free f, exactly (not just approximately) when ddof = 0.

WHAT YOU SHOULD LEARN
    - What k-fold CV actually does and how group leakage breaks it.
    - The bias-variance decomposition as something you can measure, not just recite:
      low degree means high bias, high degree means high variance.
    - How regularisation (ridge) trades a little bias for a lot less variance.
"""
from __future__ import annotations

import numpy as np


def kfold_indices(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train, val))
    return out


def group_kfold_indices(groups, k: int):
    groups = np.asarray(groups)
    ids, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")  # largest group first
    fold_sizes = [0] * k
    fold_groups: list[list] = [[] for _ in range(k)]
    for gi in order:
        target = int(np.argmin(fold_sizes))  # currently smallest fold
        fold_groups[target].append(ids[gi])
        fold_sizes[target] += int(counts[gi])
    out = []
    for i in range(k):
        val_mask = np.isin(groups, fold_groups[i])
        out.append((np.flatnonzero(~val_mask), np.flatnonzero(val_mask)))
    return out


def poly_fit(x, y, degree: int, ridge: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    V = np.vander(x, degree + 1, increasing=True)  # columns x^0 ... x^degree
    A = V.T @ V + ridge * np.eye(degree + 1)
    return np.linalg.solve(A, V.T @ np.asarray(y, dtype=float))


def poly_predict(coeffs: np.ndarray, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.vander(x, len(coeffs), increasing=True) @ coeffs


def cross_val_mse(x, y, degree: int, k: int = 5, seed: int = 0, ridge: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    errs = []
    for tr, va in kfold_indices(len(x), k, seed):
        c = poly_fit(x[tr], y[tr], degree, ridge)
        errs.append(np.mean((poly_predict(c, x[va]) - y[va]) ** 2))
    return float(np.mean(errs))


def bias_variance(degree: int, n_train: int = 30, n_trials: int = 300, noise: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.0, 1.0, 101)
    truth = np.sin(np.pi * grid)
    preds = np.empty((n_trials, len(grid)))
    for t in range(n_trials):
        x = rng.uniform(-1, 1, n_train)
        y = np.sin(np.pi * x) + rng.normal(0.0, noise, n_train)
        preds[t] = poly_predict(poly_fit(x, y, degree), grid)
    bias_sq = float(np.mean((preds.mean(axis=0) - truth) ** 2))
    variance = float(np.mean(preds.var(axis=0)))  # ddof = 0
    return bias_sq, variance, bias_sq + variance + noise ** 2


def _run_tests() -> None:
    # k-fold: partition property and balanced sizes
    folds = kfold_indices(23, 5, seed=0)
    assert len(folds) == 5
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val.tolist()) == list(range(23))
    sizes = [len(va) for _, va in folds]
    assert max(sizes) - min(sizes) <= 1
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va)) and len(tr) + len(va) == 23
    # deterministic given the seed, different for another seed
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, kfold_indices(23, 5, seed=0)))
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(folds, kfold_indices(23, 5, seed=1)))

    # group k-fold: no group straddles train and validation, every row validated once
    groups = np.repeat(np.arange(12), [5, 1, 2, 8, 3, 3, 4, 2, 6, 1, 2, 3])
    gf = group_kfold_indices(groups, 4)
    assert sorted(np.concatenate([va for _, va in gf]).tolist()) == list(range(len(groups)))
    for tr, va in gf:
        assert set(groups[tr]).isdisjoint(set(groups[va]))
    gsizes = [len(va) for _, va in gf]
    assert max(gsizes) - min(gsizes) <= 2, gsizes

    # polynomial fit recovers exact coefficients from noise-free data
    xs = np.linspace(-1, 1, 20)
    ys = 1.0 - 2.0 * xs + 0.5 * xs ** 3
    c = poly_fit(xs, ys, 3)
    assert np.allclose(c, [1.0, -2.0, 0.0, 0.5], atol=1e-8)
    assert np.allclose(poly_predict(c, xs), ys, atol=1e-8)
    # ridge shrinks coefficients
    assert np.linalg.norm(poly_fit(xs, ys, 3, ridge=1.0)) < np.linalg.norm(c)

    # cross-validation picks a sensible degree on noisy sin data
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, 60)
    y = np.sin(np.pi * x) + rng.normal(0, 0.2, 60)
    cv = {d: cross_val_mse(x, y, d, k=5, seed=0) for d in (1, 3, 5)}
    assert cv[3] < cv[1] and cv[5] < cv[1], cv
    # training error always falls with degree, validation error does not have to
    tr_mse = [np.mean((poly_predict(poly_fit(x, y, d), x) - y) ** 2) for d in (1, 3, 5, 9, 15)]
    assert all(tr_mse[i + 1] <= tr_mse[i] + 1e-9 for i in range(len(tr_mse) - 1))

    # bias-variance experiment
    b1, v1, m1 = bias_variance(1)
    b5, v5, m5 = bias_variance(5)
    b9, v9, m9 = bias_variance(9)
    assert abs(m1 - (b1 + v1 + 0.09)) < 1e-12
    assert b1 > 100 * b5, (b1, b5)  # bias collapses once the model can express sin
    assert v1 < v5 < v9, (v1, v5, v9)  # variance grows with capacity
    assert v9 > 10 * v5  # a degree-9 fit on 30 points chases the noise
    assert m5 < m1 and m5 < m9  # the middle model has the lowest expected error
    assert b1 > v1  # degree 1 is bias dominated
    # the same decomposition is deterministic given the seed
    assert bias_variance(3, seed=0) == bias_variance(3, seed=0)
    # a degree-9 fit on few points has high variance; averaging many fits would not
    assert bias_variance(9, n_train=15, n_trials=150)[1] > bias_variance(9, n_train=60, n_trials=150)[1]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
