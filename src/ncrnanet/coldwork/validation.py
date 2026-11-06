from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score

_Z = 1.959963984540054


@dataclass
class Case:
    name: str
    database: str
    entry: str
    difficulty: str
    prediction: bool
    concordant: int
    confidence: float


@dataclass
class ConcordanceReport:
    overall: float
    ci_low: float
    ci_high: float
    kappa: float
    by_difficulty: dict[str, float]
    n: int


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + _Z * _Z / total
    centre = phat + _Z * _Z / (2 * total)
    margin = _Z * math.sqrt(phat * (1 - phat) / total + _Z * _Z / (4 * total * total))
    return (centre - margin) / denom, (centre + margin) / denom


def retrospective_concordance(cases: list[Case]) -> ConcordanceReport:
    if not cases:
        return ConcordanceReport(0.0, 0.0, 0.0, 0.0, {}, 0)
    successes = sum(case.concordant for case in cases)
    total = len(cases)
    low, high = wilson_interval(successes, total)

    predicted = [int(case.prediction) for case in cases]
    truth = [
        int(case.prediction) if case.concordant == 1 else 1 - int(case.prediction) for case in cases
    ]
    if len(set(predicted)) < 2 or len(set(truth)) < 2:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(predicted, truth))

    buckets: dict[str, list[int]] = defaultdict(list)
    for case in cases:
        buckets[case.difficulty].append(case.concordant)
    by_difficulty = {key: sum(values) / len(values) for key, values in buckets.items()}

    return ConcordanceReport(
        overall=successes / total,
        ci_low=low,
        ci_high=high,
        kappa=kappa,
        by_difficulty=by_difficulty,
        n=total,
    )
