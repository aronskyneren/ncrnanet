from __future__ import annotations

from dataclasses import dataclass

from ..gaffer.config import FusionConfig
from .network import CeRNANetwork


@dataclass
class PrioritisedTarget:
    node: str
    score: float
    betweenness: float
    pathway: float
    concordance: float


def prioritise(
    network: CeRNANetwork,
    pathway_score: dict[str, float],
    concordance: dict[str, float],
    config: FusionConfig,
) -> list[PrioritisedTarget]:
    ranked: list[PrioritisedTarget] = []
    for node in sorted(network.nodes):
        betweenness = network.betweenness.get(node, 0.0)
        pathway = pathway_score.get(node, 0.0)
        concord = concordance.get(node, 0.0)
        score = (
            config.gamma_betw * betweenness
            + config.gamma_path * pathway
            + config.gamma_conc * concord
        )
        ranked.append(
            PrioritisedTarget(
                node=node,
                score=score,
                betweenness=betweenness,
                pathway=pathway,
                concordance=concord,
            )
        )
    ranked.sort(key=lambda t: t.score, reverse=True)
    return ranked
