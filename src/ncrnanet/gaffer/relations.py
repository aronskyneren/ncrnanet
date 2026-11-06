from __future__ import annotations

from enum import Enum


class RnaKind(str, Enum):
    LNCRNA = "lncrna"
    CIRCRNA = "circrna"
    MIRNA = "mirna"
    MRNA = "mrna"


class Relation(str, Enum):
    LNCRNA_MIRNA = "lncrna_mirna"
    CIRCRNA_MIRNA = "circrna_mirna"
    MIRNA_MRNA = "mirna_mrna"

    @property
    def source_kind(self) -> RnaKind:
        return _SOURCE[self]

    @property
    def target_kind(self) -> RnaKind:
        return _TARGET[self]


_SOURCE: dict[Relation, RnaKind] = {
    Relation.LNCRNA_MIRNA: RnaKind.LNCRNA,
    Relation.CIRCRNA_MIRNA: RnaKind.CIRCRNA,
    Relation.MIRNA_MRNA: RnaKind.MIRNA,
}

_TARGET: dict[Relation, RnaKind] = {
    Relation.LNCRNA_MIRNA: RnaKind.MIRNA,
    Relation.CIRCRNA_MIRNA: RnaKind.MIRNA,
    Relation.MIRNA_MRNA: RnaKind.MRNA,
}

ALL_RELATIONS: tuple[Relation, ...] = (
    Relation.LNCRNA_MIRNA,
    Relation.CIRCRNA_MIRNA,
    Relation.MIRNA_MRNA,
)

ALL_KINDS: tuple[RnaKind, ...] = (
    RnaKind.LNCRNA,
    RnaKind.CIRCRNA,
    RnaKind.MIRNA,
    RnaKind.MRNA,
)
