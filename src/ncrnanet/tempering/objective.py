from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    prediction: torch.Tensor
    pathway: torch.Tensor


def prediction_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def pathway_consistency(
    logits: torch.Tensor, prior: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    positive = labels > 0.5
    if not bool(positive.any()):
        return logits.new_zeros(())
    gap = probs[positive] - prior[positive]
    return (gap * gap).mean()


def composite_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_pathway: float,
    prior: torch.Tensor | None = None,
) -> LossBreakdown:
    prediction = prediction_loss(logits, labels)
    if prior is None or lambda_pathway == 0.0:
        pathway = logits.new_zeros(())
    else:
        pathway = pathway_consistency(logits, prior, labels)
    total = prediction + lambda_pathway * pathway
    return LossBreakdown(total=total, prediction=prediction, pathway=pathway)
