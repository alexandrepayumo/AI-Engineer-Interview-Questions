"""Exercise 5 - Classification metrics: P/R/F1, ROC-AUC, average precision, ECE, temperature.

TASK
    confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]
        Returns (tp, fp, fn, tn) for binary labels in {0, 1}.
    precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]
        Return 0.0 for any ratio whose denominator is zero.
    average_ranks(x: np.ndarray) -> np.ndarray
        1-based ranks of x, tied values get the mean of the ranks they span.
    roc_auc(y_true, scores) -> float
        Probability that a random positive outranks a random negative, ties count
        1/2. Implement it from RANKS (Mann-Whitney U), not by sweeping thresholds.
    average_precision(y_true, scores) -> float
        AP = sum over distinct score thresholds (descending) of
        (recall_k - recall_{k-1}) * precision_k. Tied scores form one threshold.
    expected_calibration_error(y_true, probs, n_bins=10) -> float
        Binary ECE with equal-width bins on [0, 1] (the last bin includes 1.0):
        sum over bins of (n_bin / n) * |mean(y_true in bin) - mean(probs in bin)|.
    fit_temperature(logits, y_true, grid=None) -> float
        Binary logits z. Return the T on `grid` (default np.linspace(0.25, 6, 231))
        minimising the NLL of sigmoid(z / T).

HINTS
    - Average ranks for ties: sort the scores, give tied scores the mean of their
      1-based positions. Then AUC = (sum of positive ranks - P(P+1)/2) / (P * N).
    - For AP, sort by score descending, then walk groups of equal scores and keep
      running tp / fp counts.
    - NLL from logits: mean(softplus(z') - y * z') with z' = z / T and
      softplus(x) = max(x, 0) + log1p(exp(-|x|)).

WHAT YOU SHOULD LEARN
    - AUC is a ranking probability and ignores calibration entirely.
    - Why ROC-AUC can look excellent while precision at any useful threshold is awful
      under heavy imbalance, and why average precision exposes it.
    - ECE is a binned estimate of the gap between confidence and accuracy.
    - Temperature scaling changes calibration without changing the ranking (so AUC
      and accuracy stay identical).
"""
from __future__ import annotations

import numpy as np


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # mean of 1-based positions i+1..j+1
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes")
    ranks = average_ranks(s)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0  # Mann-Whitney U statistic
    return float(u / (n_pos * n_neg))


def average_precision(y_true, scores) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    total_pos = int(y.sum())
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1  # all tied scores enter together
        tp += int(y[i : j + 1].sum())
        fp += (j - i + 1) - int(y[i : j + 1].sum())
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return float(ap)


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)  # p = 1.0 goes in the last bin
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def fit_temperature(logits, y_true, grid=None) -> float:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if grid is None:
        grid = np.linspace(0.25, 6.0, 231)
    best_t, best_nll = None, np.inf
    for t in grid:
        zt = z / t
        nll = np.mean(np.maximum(zt, 0) - y * zt + np.log1p(np.exp(-np.abs(zt))))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def _run_tests() -> None:
    # confusion matrix and P/R/F1 by hand: tp=3 fp=1 fn=2 tn=4
    yt = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    yp = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert confusion_counts(yt, yp) == (3, 1, 2, 4)
    p, r, f1 = precision_recall_f1(yt, yp)
    assert abs(p - 0.75) < 1e-12 and abs(r - 0.6) < 1e-12
    assert abs(f1 - 2 * 0.75 * 0.6 / 1.35) < 1e-12
    assert precision_recall_f1([0, 0], [0, 0]) == (0.0, 0.0, 0.0)

    # accuracy lies: 1% positives, predict-all-negative
    yt_imb = np.zeros(1000, dtype=int)
    yt_imb[:10] = 1
    assert (np.zeros(1000) == yt_imb).mean() == 0.99
    assert precision_recall_f1(yt_imb, np.zeros(1000))[2] == 0.0

    # the textbook AUC / AP example
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert abs(roc_auc(y, s) - 0.75) < 1e-12
    assert abs(average_precision(y, s) - (0.5 * 1.0 + 0.5 * (2 / 3))) < 1e-12

    # ties: constant scores give AUC 0.5; a perfect ranking gives 1; reversed gives 0
    assert abs(roc_auc([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]) - 0.5) < 1e-12
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert abs(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
    # partial ties: positive tied with one negative -> that pair counts 1/2
    assert abs(roc_auc([0, 1, 0], [0.5, 0.5, 0.1]) - 0.75) < 1e-12

    # AUC is invariant to monotonic transforms of the scores, and matches brute force
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 200)
    ss = rng.normal(size=200) + yy
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in ss[yy == 1] for b in ss[yy == 0]])
    assert abs(roc_auc(yy, ss) - brute) < 1e-12
    assert abs(roc_auc(yy, np.exp(ss)) - brute) < 1e-12

    # heavy imbalance: AUC looks great, average precision does not
    neg = np.linspace(0.0, 1.0, 9990)
    pos = np.linspace(0.95, 0.99, 10)
    labels = np.concatenate([np.zeros(9990), np.ones(10)])
    scores = np.concatenate([neg, pos])
    assert roc_auc(labels, scores) > 0.95
    assert average_precision(labels, scores) < 0.1

    # ECE: a calibrated group has 0; an overconfident one has |0.99 - 0.7|
    probs = np.full(10, 0.85)
    assert expected_calibration_error([1] * 8 + [0] * 2, np.full(10, 0.8)) < 1e-12
    assert abs(expected_calibration_error([1] * 7 + [0] * 3, np.full(10, 0.99)) - 0.29) < 1e-12
    assert expected_calibration_error([1, 0], [1.0, 0.0]) == 0.0
    assert probs.shape == (10,)

    # temperature scaling: overconfident logits (true T = 3) are recovered, ranking is unchanged
    rng = np.random.default_rng(1)
    true_logit = rng.normal(0.0, 2.0, size=20000)
    labels = (rng.uniform(size=20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    reported = 3.0 * true_logit  # the model is 3x too confident
    t = fit_temperature(reported, labels)
    assert 2.6 < t < 3.4, t
    p_before = 1 / (1 + np.exp(-reported))
    p_after = 1 / (1 + np.exp(-reported / t))
    assert expected_calibration_error(labels, p_after) < expected_calibration_error(labels, p_before) / 3
    assert abs(roc_auc(labels, p_after) - roc_auc(labels, reported)) < 1e-12


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
