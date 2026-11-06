from __future__ import annotations

from typing import cast

import numpy as np

from ..gaffer.config import FusionConfig


def fuse_scores(
    prediction: np.ndarray,
    database: np.ndarray,
    pathway: np.ndarray,
    config: FusionConfig,
) -> np.ndarray:
    fused = (
        config.beta_pred * prediction + config.beta_db * database + config.beta_pathway * pathway
    )
    return cast(np.ndarray, fused)


def expression_mask(
    expression: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    percentile: float,
) -> np.ndarray:
    if expression.size == 0:
        return np.ones(src.shape[0], dtype=bool)
    cutoff = float(np.percentile(expression, percentile))
    keep_src = expression[src] >= cutoff
    keep_dst = expression[dst] >= cutoff
    return cast(np.ndarray, keep_src & keep_dst)
