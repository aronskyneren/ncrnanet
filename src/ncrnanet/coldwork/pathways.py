from __future__ import annotations

import numpy as np

SCI_PATHWAYS: tuple[str, ...] = (
    "NF-kB",
    "PI3K/Akt",
    "MAPK/TNF",
    "NLRP3",
    "Wnt/beta-catenin",
    "Apoptosis/autophagy",
)


class PathwayPrior:
    def __init__(self, membership: np.ndarray, enrichment: np.ndarray) -> None:
        if enrichment.shape[0] != len(SCI_PATHWAYS):
            raise ValueError("enrichment must score every SCI pathway")
        self.membership = membership.astype(np.int64)
        self.enrichment = enrichment.astype(np.float64)

    @classmethod
    def from_expression(cls, membership: np.ndarray, expression: np.ndarray) -> PathwayPrior:
        ranks = expression.argsort().argsort().astype(np.float64)
        normed = ranks / max(len(expression) - 1, 1)
        scores = np.zeros(len(SCI_PATHWAYS), dtype=np.float64)
        for k in range(len(SCI_PATHWAYS)):
            members = membership == k
            scores[k] = float(normed[members].mean()) if bool(members.any()) else 0.0
        return cls(membership=membership, enrichment=scores)

    def pair_prior(self, i: int, j: int) -> float:
        best = 0.0
        for k in range(len(SCI_PATHWAYS)):
            shared = int(self.membership[i] == k) + int(self.membership[j] == k)
            value = (shared / 2.0) * self.enrichment[k]
            best = max(best, value)
        return best

    def prior_vector(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        values = [self.pair_prior(int(s), int(d)) for s, d in zip(src, dst, strict=False)]
        return np.array(values, dtype=np.float64)

    def node_enrichment(self, node: int) -> float:
        label = int(self.membership[node])
        if 0 <= label < len(SCI_PATHWAYS):
            return float(self.enrichment[label])
        return 0.0
