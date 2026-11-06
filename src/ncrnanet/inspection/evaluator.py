from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..forming.folds import cross_validation_splits
from ..forming.trainer import FoldResult, Trainer
from ..furnace.ncrnanet import build_model
from ..gaffer.config import ExperimentConfig
from ..gaffer.logbook import get_logger
from ..gaffer.seed import set_seed
from .preparation import Dataset, build_dataset

_LOG = get_logger("inspection.evaluator")


@dataclass
class CrossValidationResult:
    auc_mean: float
    auc_std: float
    aupr_mean: float
    aupr_std: float
    acc_mean: float
    acc_std: float
    f1_mean: float
    f1_std: float
    folds: list[FoldResult] = field(default_factory=list)


def _summary(values: list[float]) -> tuple[float, float]:
    array = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), 0.0
    return float(array.mean()), float(array.std(ddof=0))


def run_cross_validation(
    config: ExperimentConfig,
    data_root: Path | None = None,
    device: torch.device | None = None,
    dataset: Dataset | None = None,
) -> CrossValidationResult:
    set_seed(config.train.seed)
    prepared = dataset or build_dataset(config, data_root, config.train.seed)
    splits = cross_validation_splits(
        len(prepared.interactions), config.train.folds, config.train.seed
    )
    trainer = Trainer(config, device)

    folds: list[FoldResult] = []
    for fold, split in enumerate(splits):
        set_seed(config.train.seed + fold)
        model = build_model(config.model)
        result = trainer.train(
            model,
            prepared.graph,
            prepared.interactions,
            split.train,
            split.val,
            split.test,
            prepared.prior_vector,
        )
        _LOG.info("fold %d test AUC %.4f", fold, result.test_scores.auc)
        folds.append(result)

    auc_mean, auc_std = _summary([f.test_scores.auc for f in folds])
    aupr_mean, aupr_std = _summary([f.test_scores.aupr for f in folds])
    acc_mean, acc_std = _summary([f.test_scores.accuracy for f in folds])
    f1_mean, f1_std = _summary([f.test_scores.f1 for f in folds])
    return CrossValidationResult(
        auc_mean=auc_mean,
        auc_std=auc_std,
        aupr_mean=aupr_mean,
        aupr_std=aupr_std,
        acc_mean=acc_mean,
        acc_std=acc_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
        folds=folds,
    )
