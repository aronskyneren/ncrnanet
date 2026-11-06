from __future__ import annotations

import numpy as np
from scipy.stats import norm, ttest_rel


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[b] = float(sample.mean())
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def paired_t_test(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    statistic, pvalue = ttest_rel(left, right)
    return float(statistic), float(pvalue)


def cohens_d_paired(left: np.ndarray, right: np.ndarray) -> float:
    diff = left - right
    spread = diff.std(ddof=1)
    if spread == 0:
        return 0.0
    return float(diff.mean() / spread)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(predictions: np.ndarray, positives: int) -> tuple[np.ndarray, np.ndarray]:
    m = positives
    n = predictions.shape[1] - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(predictions[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    labels: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> tuple[float, float, float, float]:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(-labels, kind="stable")
    predictions = np.vstack((score_a, score_b))[:, order]
    aucs, cov = _fast_delong(predictions, positives)
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = 0.0 if variance <= 0 else float((aucs[0] - aucs[1]) / np.sqrt(variance))
    pvalue = float(2 * (1 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), z, pvalue
