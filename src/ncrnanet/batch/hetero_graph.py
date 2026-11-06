from __future__ import annotations

from dataclasses import dataclass

import torch

from ..gaffer.relations import ALL_RELATIONS, Relation


@dataclass
class HeteroGraph:
    inputs: torch.Tensor
    kinds: torch.Tensor
    edges: dict[Relation, torch.Tensor]

    @property
    def n_nodes(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.inputs.shape[1])

    def to(self, device: torch.device) -> HeteroGraph:
        return HeteroGraph(
            inputs=self.inputs.to(device),
            kinds=self.kinds.to(device),
            edges={rel: idx.to(device) for rel, idx in self.edges.items()},
        )

    def validate(self) -> None:
        for relation in ALL_RELATIONS:
            index = self.edges.get(relation)
            if index is None:
                continue
            if index.numel() == 0:
                continue
            if int(index.max()) >= self.n_nodes or int(index.min()) < 0:
                raise ValueError(f"edge index out of bounds for relation {relation.value}")


@dataclass
class InteractionSet:
    src: torch.Tensor
    dst: torch.Tensor
    relation: torch.Tensor
    label: torch.Tensor

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def subset(self, index: torch.Tensor) -> InteractionSet:
        return InteractionSet(
            src=self.src[index],
            dst=self.dst[index],
            relation=self.relation[index],
            label=self.label[index],
        )

    def to(self, device: torch.device) -> InteractionSet:
        return InteractionSet(
            src=self.src.to(device),
            dst=self.dst.to(device),
            relation=self.relation.to(device),
            label=self.label.to(device),
        )


def relation_code(relation: Relation) -> int:
    return ALL_RELATIONS.index(relation)
