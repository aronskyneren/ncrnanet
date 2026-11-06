from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import confuse

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    fm_dim: int = 768
    dom_dim: int = 128
    hidden_dim: int = 256
    heads: int = 8
    layers: int = 3
    dropout: float = 0.2
    use_foundation: bool = True
    use_pathway_prior: bool = True


@dataclass(frozen=True)
class DataConfig:
    kmer_sizes: tuple[int, ...] = (3, 4, 5)
    seed_length: int = 8
    min_length: int = 18
    max_length: int = 256
    expression_percentile: float = 25.0


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 512
    grad_clip: float = 1.0
    lambda_pathway: float = 0.1
    ema_decay: float = 0.0
    seed: int = 0
    folds: int = 5


@dataclass(frozen=True)
class FusionConfig:
    beta_pred: float = 0.5
    beta_db: float = 0.3
    beta_pathway: float = 0.2
    tau: float = 0.6
    min_component: int = 3
    high_tier: float = 0.8
    gamma_betw: float = 0.4
    gamma_path: float = 0.3
    gamma_conc: float = 0.3


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "main"
    dataset: str = "cmi9905"
    foundation: str = "ernie_rna"
    transfer_target: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


def _pick(section: Mapping[str, Any], key: str, default: T) -> T:
    if key not in section:
        return default
    value = section[key]
    if isinstance(default, tuple):
        return tuple(value)  # type: ignore[return-value]
    return type(default)(value)  # type: ignore[call-arg]


def _model(raw: Mapping[str, Any]) -> ModelConfig:
    base = ModelConfig()
    return ModelConfig(
        fm_dim=_pick(raw, "fm_dim", base.fm_dim),
        dom_dim=_pick(raw, "dom_dim", base.dom_dim),
        hidden_dim=_pick(raw, "hidden_dim", base.hidden_dim),
        heads=_pick(raw, "heads", base.heads),
        layers=_pick(raw, "layers", base.layers),
        dropout=_pick(raw, "dropout", base.dropout),
        use_foundation=_pick(raw, "use_foundation", base.use_foundation),
        use_pathway_prior=_pick(raw, "use_pathway_prior", base.use_pathway_prior),
    )


def _data(raw: Mapping[str, Any]) -> DataConfig:
    base = DataConfig()
    return DataConfig(
        kmer_sizes=_pick(raw, "kmer_sizes", base.kmer_sizes),
        seed_length=_pick(raw, "seed_length", base.seed_length),
        min_length=_pick(raw, "min_length", base.min_length),
        max_length=_pick(raw, "max_length", base.max_length),
        expression_percentile=_pick(raw, "expression_percentile", base.expression_percentile),
    )


def _train(raw: Mapping[str, Any]) -> TrainConfig:
    base = TrainConfig()
    return TrainConfig(
        lr=_pick(raw, "lr", base.lr),
        weight_decay=_pick(raw, "weight_decay", base.weight_decay),
        warmup_epochs=_pick(raw, "warmup_epochs", base.warmup_epochs),
        max_epochs=_pick(raw, "max_epochs", base.max_epochs),
        patience=_pick(raw, "patience", base.patience),
        batch_size=_pick(raw, "batch_size", base.batch_size),
        grad_clip=_pick(raw, "grad_clip", base.grad_clip),
        lambda_pathway=_pick(raw, "lambda_pathway", base.lambda_pathway),
        ema_decay=_pick(raw, "ema_decay", base.ema_decay),
        seed=_pick(raw, "seed", base.seed),
        folds=_pick(raw, "folds", base.folds),
    )


def _fusion(raw: Mapping[str, Any]) -> FusionConfig:
    base = FusionConfig()
    return FusionConfig(
        beta_pred=_pick(raw, "beta_pred", base.beta_pred),
        beta_db=_pick(raw, "beta_db", base.beta_db),
        beta_pathway=_pick(raw, "beta_pathway", base.beta_pathway),
        tau=_pick(raw, "tau", base.tau),
        min_component=_pick(raw, "min_component", base.min_component),
        high_tier=_pick(raw, "high_tier", base.high_tier),
        gamma_betw=_pick(raw, "gamma_betw", base.gamma_betw),
        gamma_path=_pick(raw, "gamma_path", base.gamma_path),
        gamma_conc=_pick(raw, "gamma_conc", base.gamma_conc),
    )


def from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    base = ExperimentConfig()
    return ExperimentConfig(
        name=_pick(raw, "name", base.name),
        dataset=_pick(raw, "dataset", base.dataset),
        foundation=_pick(raw, "foundation", base.foundation),
        transfer_target=_pick(raw, "transfer_target", base.transfer_target),
        model=_model(raw.get("model", {})),
        data=_data(raw.get("data", {})),
        train=_train(raw.get("train", {})),
        fusion=_fusion(raw.get("fusion", {})),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    store = confuse.Configuration("ncrnanet", read=False)
    store.set_file(str(path))
    flat = store.get()
    if not isinstance(flat, dict):
        raise ValueError(f"config file is not a mapping: {path}")
    return flat


def load_experiment(path: str | Path) -> ExperimentConfig:
    location = Path(path)
    top = _read_yaml(location)
    includes = top.pop("include", {}) or {}
    recipes_root = location.parent.parent
    merged: dict[str, Any] = {}
    for section, relative in includes.items():
        merged[section] = _read_yaml(recipes_root / relative)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return from_mapping(merged)
