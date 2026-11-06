from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ..gaffer.relations import ALL_RELATIONS, Relation


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    heads = scores.shape[1]
    expanded = index.unsqueeze(-1).expand_as(scores)
    ceiling = scores.new_full((num_nodes, heads), float("-inf"))
    ceiling = ceiling.scatter_reduce(0, expanded, scores, reduce="amax", include_self=True)
    centered = scores - ceiling[index]
    weights = centered.exp()
    denom = scores.new_zeros((num_nodes, heads))
    denom = denom.index_add(0, index, weights)
    return weights / (denom[index] + 1e-16)


class RelationAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden dim must be divisible by head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.transform = nn.ModuleDict(
            {relation.value: nn.Linear(dim, dim, bias=False) for relation in ALL_RELATIONS}
        )
        self.attn_src = nn.ParameterDict(
            {
                relation.value: nn.Parameter(torch.empty(heads, self.head_dim))
                for relation in ALL_RELATIONS
            }
        )
        self.attn_dst = nn.ParameterDict(
            {
                relation.value: nn.Parameter(torch.empty(heads, self.head_dim))
                for relation in ALL_RELATIONS
            }
        )
        self.leaky = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for relation in ALL_RELATIONS:
            linear = cast(nn.Linear, self.transform[relation.value])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.xavier_uniform_(self.attn_src[relation.value])
            nn.init.xavier_uniform_(self.attn_dst[relation.value])

    def _relation_message(
        self, relation: Relation, features: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = features.shape[0]
        neighbour = torch.cat([edges[0], edges[1]], dim=0)
        receiver = torch.cat([edges[1], edges[0]], dim=0)
        linear = cast(nn.Linear, self.transform[relation.value])
        projected: torch.Tensor = linear(features).view(num_nodes, self.heads, self.head_dim)
        key_src = (projected * self.attn_src[relation.value]).sum(-1)
        key_dst = (projected * self.attn_dst[relation.value]).sum(-1)
        logits = self.leaky(key_src[neighbour] + key_dst[receiver])
        weights = segment_softmax(logits, receiver, num_nodes)
        messages = projected[neighbour] * weights.unsqueeze(-1)
        aggregate = projected.new_zeros((num_nodes, self.heads, self.head_dim))
        aggregate = aggregate.index_add(0, receiver, messages)
        return aggregate.reshape(num_nodes, self.dim)

    def forward(self, features: torch.Tensor, edges: dict[Relation, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros_like(features)
        for relation in ALL_RELATIONS:
            index = edges.get(relation)
            if index is None or index.numel() == 0:
                continue
            total = total + self._relation_message(relation, features, index)
        return total


class HeteroGAT(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([RelationAttention(dim, heads) for _ in range(layers)])
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edges: dict[Relation, torch.Tensor]) -> torch.Tensor:
        state = features
        for layer in self.layers:
            message = layer(state, edges)
            state = self.dropout(self.activation(message)) + state
        return state
