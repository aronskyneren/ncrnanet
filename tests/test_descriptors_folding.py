from __future__ import annotations

import numpy as np

from ncrnanet.batch.descriptors import DomainDescriptor
from ncrnanet.batch.folding import folding_energy_proxy, nussinov_pairs


def test_descriptor_dimension_matches_request() -> None:
    descriptor = DomainDescriptor(dim=128)
    vector = descriptor.encode("ACGUACGUACGUACGUACGU")
    assert vector.shape == (128,)


def test_descriptor_is_deterministic() -> None:
    descriptor = DomainDescriptor(dim=96)
    a = descriptor.encode("ACGUACGUACGUAAAA")
    b = descriptor.encode("ACGUACGUACGUAAAA")
    assert np.array_equal(a, b)


def test_gc_content_slot() -> None:
    descriptor = DomainDescriptor(dim=64)
    all_gc = descriptor.encode("GCGCGCGCGCGCGCGC")
    no_gc = descriptor.encode("AUAUAUAUAUAUAUAU")
    assert all_gc[0] == 1.0
    assert no_gc[0] == 0.0


def test_nussinov_hairpin_pairs() -> None:
    assert nussinov_pairs("GGGGAAAACCCC") >= 3
    assert nussinov_pairs("AAAAAAAA") == 0


def test_folding_proxy_is_non_positive() -> None:
    assert folding_energy_proxy("GGGGAAAACCCC") <= 0.0
    assert folding_energy_proxy("") == 0.0
