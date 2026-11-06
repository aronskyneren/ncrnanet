from __future__ import annotations

import torch
from torch import nn


class ParameterAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }

    @property
    def active(self) -> bool:
        return self.decay > 0.0

    def update(self, model: nn.Module) -> None:
        if not self.active:
            return
        for name, param in model.named_parameters():
            stored = self.shadow[name]
            stored.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        if not self.active:
            return
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
