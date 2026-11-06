from __future__ import annotations

import torch
from torch import nn


class HybridEmbedding(nn.Module):
    def __init__(
        self,
        fm_dim: int,
        dom_dim: int,
        hidden_dim: int,
        dropout: float,
        use_foundation: bool = True,
        use_domain: bool = True,
    ) -> None:
        super().__init__()
        self.fm_dim = fm_dim
        self.dom_dim = dom_dim
        self.use_foundation = use_foundation
        self.use_domain = use_domain
        self.project = nn.Sequential(
            nn.Linear(fm_dim + dom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fm = inputs[:, : self.fm_dim]
        dom = inputs[:, self.fm_dim :]
        if not self.use_foundation:
            fm = torch.zeros_like(fm)
        if not self.use_domain:
            dom = torch.zeros_like(dom)
        joined = torch.cat([fm, dom], dim=1)
        projected: torch.Tensor = self.project(joined)
        normalised: torch.Tensor = self.norm(projected)
        return normalised
