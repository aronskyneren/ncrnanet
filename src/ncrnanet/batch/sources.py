from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..gaffer.relations import ALL_KINDS, ALL_RELATIONS, Relation, RnaKind
from .cohort import Cohort

_KIND_BY_NAME = {kind.value: kind for kind in RnaKind}
_RELATION_BY_PAIR = {(r.source_kind, r.target_kind): r for r in ALL_RELATIONS}
_REQUIRED = {
    'src_id', 'src_kind', 'src_seq', 'src_expression', 'src_pathway',
    'dst_id', 'dst_kind', 'dst_seq', 'dst_expression', 'dst_pathway', 'label',
}


def load_pairs_csv(path: Path) -> Cohort:
    nodes: dict[tuple[RnaKind, str], tuple[str, float, int]] = {}
    pairs: dict[tuple[tuple[RnaKind, str], tuple[RnaKind, str]], int] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Missing CSV columns: {", ".join(sorted(missing))}')
        for line, row in enumerate(reader, start=2):
            keys = []
            for prefix in ('src', 'dst'):
                kind = _KIND_BY_NAME[row[f'{prefix}_kind'].strip().lower()]
                identifier = row[f'{prefix}_id'].strip()
                sequence = row[f'{prefix}_seq'].strip().upper()
                expression = float(row[f'{prefix}_expression'])
                pathway = int(row[f'{prefix}_pathway'])
                if (
                    not identifier or not sequence or not np.isfinite(expression)
                    or not -1 <= pathway < 6
                ):
                    raise ValueError(f'Invalid node data at line {line}')
                key = (kind, identifier)
                value = (sequence, expression, pathway)
                if key in nodes and nodes[key] != value:
                    raise ValueError(f'Inconsistent node data at line {line}')
                nodes[key] = value
                keys.append(key)
            src, dst = keys
            if (src[0], dst[0]) not in _RELATION_BY_PAIR:
                raise ValueError(f'Unsupported relation at line {line}')
            label = int(row['label'])
            if label not in (0, 1):
                raise ValueError(f'Invalid label at line {line}')
            pair = (src, dst)
            if pair in pairs and pairs[pair] != label:
                raise ValueError(f'Conflicting interaction labels at line {line}')
            pairs[pair] = label
    if not pairs:
        raise ValueError(f'No interaction records in {path}')
    ordered = sorted(nodes, key=lambda key: (ALL_KINDS.index(key[0]), key[1]))
    indices = {key: i for i, key in enumerate(ordered)}
    offsets = {}
    for kind in ALL_KINDS:
        members = [i for i, key in enumerate(ordered) if key[0] == kind]
        if members:
            offsets[kind] = (members[0], members[-1] + 1)
    positives: dict[Relation, list[tuple[int, int]]] = {r: [] for r in ALL_RELATIONS}
    negatives: dict[Relation, list[tuple[int, int]]] = {r: [] for r in ALL_RELATIONS}
    for (src, dst), label in pairs.items():
        relation = _RELATION_BY_PAIR[(src[0], dst[0])]
        (positives if label == 1 else negatives)[relation].append((indices[src], indices[dst]))
    return Cohort(
        ids=[f'{kind.value}:{identifier}' for kind, identifier in ordered],
        sequences=[nodes[key][0] for key in ordered],
        kinds=[key[0] for key in ordered],
        expression=np.array([nodes[key][1] for key in ordered], dtype=np.float64),
        pathways=np.array([nodes[key][2] for key in ordered], dtype=np.int64),
        positives=positives,
        negatives=negatives,
        offsets=offsets,
    )


def resolve_cohort(name: str, root: Path | None, seed: int) -> Cohort:
    if root is None:
        raise ValueError('A data root containing cohort files is required.')
    candidate = root / f'{name}.csv'
    if not candidate.is_file():
        raise FileNotFoundError(f'Cohort file not found: {candidate}')
    return load_pairs_csv(candidate)
