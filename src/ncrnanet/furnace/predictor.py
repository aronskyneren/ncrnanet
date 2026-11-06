from __future__ import annotations

import torch
from torch import nn

from ..gaffer.relations import ALL_RELATIONS


class BilinearPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.ParameterDict(
            {relation.value: nn.Parameter(torch.empty(dim, dim)) for relation in ALL_RELATIONS}
        )
        self.bias = nn.ParameterDict(
            {relation.value: nn.Parameter(torch.zeros(1)) for relation in ALL_RELATIONS}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for relation in ALL_RELATIONS:
            nn.init.eye_(self.weight[relation.value])

    def relation_logits(
        self, src_embed: torch.Tensor, dst_embed: torch.Tensor, relation_code: int
    ) -> torch.Tensor:
        relation = ALL_RELATIONS[relation_code]
        matrix = self.weight[relation.value]
        projected = dst_embed @ matrix.t()
        logits: torch.Tensor = (src_embed * projected).sum(-1) + self.bias[relation.value]
        return logits

    def forward(
        self,
        embeddings: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        logits = embeddings.new_zeros(src.shape[0])
        for code in range(len(ALL_RELATIONS)):
            mask = relation == code
            if not bool(mask.any()):
                continue
            rows = mask.nonzero(as_tuple=False).squeeze(-1)
            src_embed = embeddings[src[rows]]
            dst_embed = embeddings[dst[rows]]
            logits[rows] = self.relation_logits(src_embed, dst_embed, code)
        return logits
