from __future__ import annotations

BASES: tuple[str, ...] = ("A", "C", "G", "U")
_CANON = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U", "N": "A"}
_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
_WOBBLE = {("G", "U"), ("U", "G")}


def normalise(sequence: str) -> str:
    upper = sequence.strip().upper()
    return "".join(_CANON.get(symbol, "A") for symbol in upper)


def can_pair(left: str, right: str) -> bool:
    if _COMPLEMENT.get(left) == right:
        return True
    return (left, right) in _WOBBLE
