from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def cross_validation_splits(n: int, folds: int, seed: int) -> list[Split]:
    return [holdout_split(n, seed + fold) for fold in range(folds)]


def holdout_split(
    n: int, seed: int, val_fraction: float = 0.1, test_fraction: float = 0.1
) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val_size = int(round(n * val_fraction))
    test_size = int(round(n * test_fraction))
    test = order[:test_size]
    val = order[test_size : test_size + val_size]
    train = order[test_size + val_size :]
    return Split(train=train, val=val, test=test)
