from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import fisher_exact


@dataclass
class NetworkScores:
    precision: float
    recall: float
    enrichment_p: float


def precision_recall(
    predicted: set[tuple[str, str]], validated: set[tuple[str, str]]
) -> tuple[float, float]:
    if not predicted:
        return 0.0, 0.0
    hits = len(predicted & validated)
    precision = hits / len(predicted)
    recall = hits / len(validated) if validated else 0.0
    return precision, recall


def enrichment_pvalue(
    predicted: set[tuple[str, str]],
    validated: set[tuple[str, str]],
    universe: int,
) -> float:
    hits = len(predicted & validated)
    predicted_only = len(predicted) - hits
    validated_only = len(validated) - hits
    remainder = max(universe - hits - predicted_only - validated_only, 0)
    table = [[hits, predicted_only], [validated_only, remainder]]
    _, pvalue = fisher_exact(table, alternative="greater")
    return float(pvalue)


def evaluate_network(
    predicted: set[tuple[str, str]],
    validated: set[tuple[str, str]],
    universe: int,
) -> NetworkScores:
    precision, recall = precision_recall(predicted, validated)
    pvalue = enrichment_pvalue(predicted, validated, universe)
    return NetworkScores(precision=precision, recall=recall, enrichment_p=pvalue)
