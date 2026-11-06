from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from ..calipers.scores import evaluate_interactions
from ..forming.checkpoint import load_checkpoint, save_checkpoint
from ..forming.folds import holdout_split
from ..forming.trainer import Trainer
from ..furnace.ncrnanet import build_model
from ..gaffer.config import ExperimentConfig, load_experiment
from ..gaffer.logbook import get_logger
from ..gaffer.seed import set_seed
from ..inspection.evaluator import run_cross_validation
from ..inspection.preparation import build_dataset
from ..inspection.report import format_cross_validation

_LOG = get_logger("bench")
_RECIPE_ROOT = Path("recipes/pour")


def _load(experiment: str) -> ExperimentConfig:
    candidate = Path(experiment)
    if candidate.is_file():
        return load_experiment(candidate)
    stem = experiment.split("/")[-1]
    fallback = _RECIPE_ROOT / f"{stem}.yaml"
    if fallback.is_file():
        return load_experiment(fallback)
    raise FileNotFoundError(f"no recipe found for '{experiment}'")


def _root(data_root: str) -> Path | None:
    return Path(data_root) if data_root else None


def train(
    experiment: str, *, data_root: str = "", out_dir: str = "runs", device: str = "cpu"
) -> None:
    config = _load(experiment)
    set_seed(config.train.seed)
    dataset = build_dataset(config, _root(data_root), config.train.seed)
    split = holdout_split(len(dataset.interactions), config.train.seed)
    model = build_model(config.model)
    trainer = Trainer(config, torch.device(device))
    result = trainer.train(
        model,
        dataset.graph,
        dataset.interactions,
        split.train,
        split.val,
        split.test,
        dataset.prior_vector,
    )
    destination = Path(out_dir) / f"{config.name}.pt"
    save_checkpoint(destination, model, config.train.seed, {"experiment": config.name})
    _LOG.info("checkpoint written to %s", destination)
    print(
        f"val AUC {result.best_val_auc:.4f} | "
        f"test AUC {result.test_scores.auc:.4f} AUPR {result.test_scores.aupr:.4f} "
        f"Acc {result.test_scores.accuracy:.4f} F1 {result.test_scores.f1:.4f}"
    )


def evaluate(experiment: str, *, data_root: str = "", device: str = "cpu") -> None:
    config = _load(experiment)
    result = run_cross_validation(config, _root(data_root), torch.device(device))
    print(format_cross_validation(config.name, config.dataset, result))


def infer(
    experiment: str,
    *,
    data_root: str = "",
    device: str = "cpu",
    checkpoint: str = "",
    top: int = 10,
) -> None:
    raise NotImplementedError(
        "Network inference requires an implementation that loads independent database evidence."
    )


def transfer(experiment: str, *, data_root: str = "", device: str = "cpu") -> None:
    config = _load(experiment)
    if not config.transfer_target:
        raise ValueError("experiment has no transfer_target")
    set_seed(config.train.seed)
    root = _root(data_root)
    source = build_dataset(config, root, config.train.seed)
    split = holdout_split(len(source.interactions), config.train.seed)
    model = build_model(config.model)
    trainer = Trainer(config, torch.device(device))
    trainer.train(
        model,
        source.graph,
        source.interactions,
        split.train,
        split.val,
        split.test,
        source.prior_vector,
    )

    target_config = replace(config, dataset=config.transfer_target)
    target = build_dataset(target_config, root, config.train.seed)
    model.eval()
    device_obj = torch.device(device)
    graph = target.graph.to(device_obj)
    inter = target.interactions.to(device_obj)
    with torch.no_grad():
        nodes = model.to(device_obj).encode(graph)
        probs = model.score_pairs(nodes, inter.src, inter.dst, inter.relation).cpu().numpy()
    scores = evaluate_interactions(inter.label.cpu().numpy(), probs)
    print(
        f"transfer {config.dataset} -> {config.transfer_target} | "
        f"AUC {scores.auc:.4f} AUPR {scores.aupr:.4f} Acc {scores.accuracy:.4f} F1 {scores.f1:.4f}"
    )


def export(
    experiment: str,
    *,
    output: str = "ncrnanet_embedding.onnx",
    device: str = "cpu",
    checkpoint: str = "",
) -> None:
    config = _load(experiment)
    model = build_model(config.model)
    if checkpoint:
        load_checkpoint(Path(checkpoint), model)
    model.eval()
    example = torch.zeros(4, config.model.fm_dim + config.model.dom_dim, device=device)
    torch.onnx.export(
        model.embedding,
        (example,),
        output,
        input_names=["features"],
        output_names=["node_state"],
        dynamic_axes={"features": {0: "nodes"}, "node_state": {0: "nodes"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported node embedding graph to {output}")
