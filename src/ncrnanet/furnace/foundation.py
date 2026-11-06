from __future__ import annotations

from pathlib import Path

import numpy as np


class FoundationEncoder:
    def __init__(self, sequences: list[str], embeddings: np.ndarray) -> None:
        self._sequences = sequences
        self._embeddings = embeddings

    @property
    def dim(self) -> int:
        return int(self._embeddings.shape[1])

    def embed_many(self, sequences: list[str]) -> np.ndarray:
        if sequences != self._sequences:
            raise ValueError('Sequence order differs from the supplied embeddings.')
        return self._embeddings.copy()


def load_foundation_encoder(
    path: Path, ids: list[str], sequences: list[str], dim: int
) -> FoundationEncoder:
    if not path.is_file():
        raise FileNotFoundError(f'Foundation embeddings not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        stored_ids = data['ids'].tolist()
        values = np.asarray(data['embeddings'], dtype=np.float64)
    if stored_ids != ids:
        raise ValueError('Embedding IDs do not match cohort IDs and order.')
    if values.shape != (len(ids), dim) or not np.isfinite(values).all():
        raise ValueError(f'Expected a finite embedding matrix of shape ({len(ids)}, {dim}).')
    return FoundationEncoder(sequences, values)
