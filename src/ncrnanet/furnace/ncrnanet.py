from __future__ import annotations

import torch
from torch import nn

from ..batch.hetero_graph import HeteroGraph
from ..gaffer.config import ModelConfig
from .embedding import HybridEmbedding
from .hetero_gat import HeteroGAT
from .predictor import BilinearPredictor


class NcRNANet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = HybridEmbedding(
            fm_dim=config.fm_dim,
            dom_dim=config.dom_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_foundation=config.use_foundation,
        )
        self.use_gat = config.layers > 0
        self.gat = (
            HeteroGAT(config.hidden_dim, config.heads, config.layers, config.dropout)
            if self.use_gat
            else None
        )
        self.predictor = BilinearPredictor(config.hidden_dim)

    def encode(self, graph: HeteroGraph) -> torch.Tensor:
        nodes: torch.Tensor = self.embedding(graph.inputs)
        if self.gat is not None:
            nodes = self.gat(nodes, graph.edges)
        return nodes

    def forward(
        self,
        graph: HeteroGraph,
        src: torch.Tensor,
        dst: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.encode(graph)
        logits: torch.Tensor = self.predictor(nodes, src, dst, relation)
        return logits

    def score_pairs(
        self,
        nodes: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.predictor(nodes, src, dst, relation))


def build_model(config: ModelConfig) -> NcRNANet:
    return NcRNANet(config)
