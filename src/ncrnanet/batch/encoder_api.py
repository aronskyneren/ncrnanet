from __future__ import annotations

from typing import Protocol

import numpy as np


class SequenceEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_many(self, sequences: list[str]) -> np.ndarray: ...
