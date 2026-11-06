from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..gaffer.relations import Relation, RnaKind


@dataclass
class Cohort:
    ids: list[str]
    sequences: list[str]
    kinds: list[RnaKind]
    expression: np.ndarray
    pathways: np.ndarray
    positives: dict[Relation, list[tuple[int, int]]]
    negatives: dict[Relation, list[tuple[int, int]]]
    offsets: dict[RnaKind, tuple[int, int]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.sequences)
