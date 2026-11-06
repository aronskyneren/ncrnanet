from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..gaffer.config import FusionConfig


@dataclass
class EdgeRecord:
    source: str
    target: str
    relation: str
    weight: float
    tier: str


@dataclass
class CeRNANetwork:
    edges: list[EdgeRecord]
    betweenness: dict[str, float]
    degree: dict[str, float]
    eigenvector: dict[str, float]
    tier_counts: dict[str, int] = field(default_factory=dict)

    @property
    def nodes(self) -> set[str]:
        out: set[str] = set()
        for edge in self.edges:
            out.add(edge.source)
            out.add(edge.target)
        return out


def confidence_tier(weight: float, high: float, tau: float) -> str:
    if weight >= high:
        return "high"
    if weight >= tau:
        return "medium"
    return "low"


def construct_network(
    sources: list[str],
    targets: list[str],
    relations: list[str],
    weights: np.ndarray,
    config: FusionConfig,
) -> CeRNANetwork:
    graph = nx.Graph()
    staged: list[EdgeRecord] = []
    for source, target, relation, weight in zip(sources, targets, relations, weights, strict=False):
        value = float(weight)
        if value < config.tau:
            continue
        graph.add_edge(source, target, weight=value)
        staged.append(
            EdgeRecord(
                source=source,
                target=target,
                relation=relation,
                weight=value,
                tier=confidence_tier(value, config.high_tier, config.tau),
            )
        )

    kept_nodes: set[str] = set()
    for component in nx.connected_components(graph):
        if len(component) >= config.min_component:
            kept_nodes |= set(component)
    pruned = graph.subgraph(kept_nodes).copy()

    edges = [e for e in staged if e.source in kept_nodes and e.target in kept_nodes]

    betweenness = {
        str(k): float(v) for k, v in nx.betweenness_centrality(pruned, weight="weight").items()
    }
    degree = {str(k): float(v) for k, v in nx.degree_centrality(pruned).items()}
    if pruned.number_of_edges() > 0:
        eigenvector = {
            str(k): float(v)
            for k, v in nx.eigenvector_centrality_numpy(pruned, weight="weight").items()
        }
    else:
        eigenvector = {node: 0.0 for node in kept_nodes}

    tier_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for edge in edges:
        tier_counts[edge.tier] += 1

    return CeRNANetwork(
        edges=edges,
        betweenness=betweenness,
        degree=degree,
        eigenvector=eigenvector,
        tier_counts=tier_counts,
    )
