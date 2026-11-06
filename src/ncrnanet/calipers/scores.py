from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


@dataclass
class InteractionScores:
    auc: float
    aupr: float
    accuracy: float
    f1: float


def evaluate_interactions(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> InteractionScores:
    predicted = (probabilities >= threshold).astype(np.int64)
    truth = labels.astype(np.int64)
    if len(np.unique(truth)) < 2:
        auc = float("nan")
        aupr = float("nan")
    else:
        auc = float(roc_auc_score(truth, probabilities))
        aupr = float(average_precision_score(truth, probabilities))
    return InteractionScores(
        auc=auc,
        aupr=aupr,
        accuracy=float(accuracy_score(truth, predicted)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
    )
