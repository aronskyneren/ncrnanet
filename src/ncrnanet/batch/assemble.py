from __future__ import annotations

import numpy as np
import torch

from ..gaffer.relations import ALL_KINDS, Relation
from .cohort import Cohort
from .descriptors import DomainDescriptor
from .encoder_api import SequenceEncoder
from .hetero_graph import HeteroGraph, InteractionSet, relation_code


def build_graph(
    cohort: Cohort, encoder: SequenceEncoder, descriptor: DomainDescriptor
) -> HeteroGraph:
    fm = encoder.embed_many(cohort.sequences)
    dom = descriptor.encode_many(cohort.sequences)
    inputs = np.concatenate([fm, dom], axis=1)
    kind_index = {kind: position for position, kind in enumerate(ALL_KINDS)}
    kinds = np.array([kind_index[k] for k in cohort.kinds], dtype=np.int64)
    edges: dict[Relation, torch.Tensor] = {}
    for relation, pairs in cohort.positives.items():
        arr = np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), dtype=np.int64)
        edges[relation] = torch.from_numpy(arr)
    graph = HeteroGraph(
        inputs=torch.from_numpy(inputs).float(),
        kinds=torch.from_numpy(kinds),
        edges=edges,
    )
    graph.validate()
    return graph


def build_interactions(cohort: Cohort) -> InteractionSet:
    src_all: list[int] = []
    dst_all: list[int] = []
    rel_all: list[int] = []
    lab_all: list[float] = []
    for label, relations in ((1.0, cohort.positives), (0.0, cohort.negatives)):
        for relation, pairs in relations.items():
            for src, dst in pairs:
                src_all.append(src)
                dst_all.append(dst)
                rel_all.append(relation_code(relation))
                lab_all.append(label)
    return InteractionSet(
        src=torch.tensor(src_all, dtype=torch.long),
        dst=torch.tensor(dst_all, dtype=torch.long),
        relation=torch.tensor(rel_all, dtype=torch.long),
        label=torch.tensor(lab_all, dtype=torch.float32),
    )
