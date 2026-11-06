from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..batch.assemble import build_graph, build_interactions
from ..batch.cohort import Cohort
from ..batch.descriptors import DomainDescriptor
from ..batch.hetero_graph import HeteroGraph, InteractionSet
from ..batch.sources import resolve_cohort
from ..coldwork.pathways import PathwayPrior
from ..furnace.foundation import FoundationEncoder, load_foundation_encoder
from ..gaffer.config import ExperimentConfig


@dataclass
class Dataset:
    cohort: Cohort
    graph: HeteroGraph
    interactions: InteractionSet
    prior: PathwayPrior
    prior_vector: torch.Tensor


def build_dataset(config: ExperimentConfig, data_root: Path | None, seed: int) -> Dataset:
    cohort = resolve_cohort(config.dataset, data_root, seed)
    if config.model.use_foundation:
        assert data_root is not None
        encoder = load_foundation_encoder(
            data_root / f"{config.dataset}.embeddings.npz",
            cohort.ids, cohort.sequences, config.model.fm_dim,
        )
    else:
        encoder = FoundationEncoder(
            cohort.sequences, np.zeros((cohort.n_nodes, config.model.fm_dim), dtype=np.float64)
        )
    descriptor = DomainDescriptor(
        dim=config.model.dom_dim,
        kmer_sizes=config.data.kmer_sizes,
        seed_length=config.data.seed_length,
    )
    graph = build_graph(cohort, encoder, descriptor)
    interactions = build_interactions(cohort)
    prior = PathwayPrior.from_expression(cohort.pathways, cohort.expression)
    prior_vector = torch.from_numpy(
        prior.prior_vector(interactions.src.numpy(), interactions.dst.numpy())
    ).float()
    return Dataset(
        cohort=cohort,
        graph=graph,
        interactions=interactions,
        prior=prior,
        prior_vector=prior_vector,
    )
