from __future__ import annotations

from .evaluator import CrossValidationResult


def format_cross_validation(name: str, dataset: str, result: CrossValidationResult) -> str:
    lines = [
        f"experiment: {name}",
        f"dataset:    {dataset}",
        f"folds:      {len(result.folds)}",
        f"AUC   {result.auc_mean:.4f} +/- {result.auc_std:.4f}",
        f"AUPR  {result.aupr_mean:.4f} +/- {result.aupr_std:.4f}",
        f"Acc   {result.acc_mean:.4f} +/- {result.acc_std:.4f}",
        f"F1    {result.f1_mean:.4f} +/- {result.f1_std:.4f}",
    ]
    return "\n".join(lines)
