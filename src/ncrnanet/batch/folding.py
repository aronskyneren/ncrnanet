from __future__ import annotations

from .alphabet import can_pair, normalise

_MIN_LOOP = 3


def nussinov_pairs(sequence: str) -> int:
    seq = normalise(sequence)
    n = len(seq)
    if n < _MIN_LOOP + 2:
        return 0
    table = [[0] * n for _ in range(n)]
    for span in range(_MIN_LOOP + 1, n):
        for i in range(0, n - span):
            j = i + span
            best = table[i + 1][j]
            if table[i][j - 1] > best:
                best = table[i][j - 1]
            if can_pair(seq[i], seq[j]):
                candidate = table[i + 1][j - 1] + 1
                if candidate > best:
                    best = candidate
            for k in range(i + 1, j):
                split = table[i][k] + table[k + 1][j]
                if split > best:
                    best = split
            table[i][j] = best
    return table[0][n - 1]


def folding_energy_proxy(sequence: str) -> float:
    seq = normalise(sequence)
    n = len(seq)
    if n == 0:
        return 0.0
    pairs = nussinov_pairs(seq)
    return -2.0 * pairs / n
