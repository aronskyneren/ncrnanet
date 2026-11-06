from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(optimizer: Optimizer, warmup_epochs: int, max_epochs: int) -> LambdaLR:
    span = max(max_epochs - warmup_epochs, 1)

    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / span
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda=factor)
