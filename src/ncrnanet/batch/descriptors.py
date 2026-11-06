from __future__ import annotations

import numpy as np

from .alphabet import normalise
from .folding import folding_energy_proxy

_SCALAR_SLOTS = 8
_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3}
_MIX = 2654435761


def _kmer_index(kmer: str) -> int:
    value = 0
    for symbol in kmer:
        value = value * 4 + _BASE_INDEX[symbol]
    return value


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for s in seq if s in ("G", "C"))
    return gc / len(seq)


def _longest_run(seq: str) -> int:
    best = 0
    current = 0
    previous = ""
    for symbol in seq:
        current = current + 1 if symbol == previous else 1
        previous = symbol
        best = max(best, current)
    return best


class DomainDescriptor:
    def __init__(
        self,
        dim: int = 128,
        kmer_sizes: tuple[int, ...] = (3, 4, 5),
        seed_length: int = 8,
    ) -> None:
        if dim <= _SCALAR_SLOTS:
            raise ValueError("descriptor dim must exceed the scalar slot count")
        self.dim = dim
        self.kmer_sizes = kmer_sizes
        self.seed_length = seed_length
        self.buckets = dim - _SCALAR_SLOTS

    def _hashed_kmers(self, seq: str) -> np.ndarray:
        counts = np.zeros(self.buckets, dtype=np.float64)
        total = 0
        for size in self.kmer_sizes:
            if len(seq) < size:
                continue
            for start in range(len(seq) - size + 1):
                index = _kmer_index(seq[start : start + size])
                bucket = (index * _MIX + size) % self.buckets
                counts[bucket] += 1.0
                total += 1
        if total > 0:
            counts /= float(total)
        return counts

    def _scalars(self, seq: str) -> np.ndarray:
        length = len(seq)
        seed = seq[1 : 1 + self.seed_length] if length >= 1 else ""
        purine = sum(1 for s in seq if s in ("A", "G"))
        pair_capacity = folding_energy_proxy(seq)
        return np.array(
            [
                _gc_content(seq),
                pair_capacity,
                min(length / 256.0, 1.0),
                _gc_content(seed),
                (sum(1 for s in seed if s in ("A", "U")) / len(seed)) if seed else 0.0,
                (purine / length) if length else 0.0,
                _longest_run(seq) / float(length) if length else 0.0,
                float(len(seq) >= self.seed_length),
            ],
            dtype=np.float64,
        )

    def encode(self, sequence: str) -> np.ndarray:
        seq = normalise(sequence)
        return np.concatenate([self._scalars(seq), self._hashed_kmers(seq)])

    def encode_many(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self.encode(s) for s in sequences])
