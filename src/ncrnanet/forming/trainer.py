from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..batch.hetero_graph import HeteroGraph, InteractionSet
from ..calipers.scores import InteractionScores, evaluate_interactions
from ..furnace.ncrnanet import NcRNANet
from ..gaffer.config import ExperimentConfig
from ..gaffer.logbook import get_logger
from ..tempering.objective import composite_loss
from .ema import ParameterAverage
from .schedule import cosine_warmup

_LOG = get_logger("forming.trainer")


@dataclass
class FoldResult:
    best_val_auc: float
    test_scores: InteractionScores
    train_loss: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)


class Trainer:
    def __init__(self, config: ExperimentConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")

    def _evaluate(
        self, model: NcRNANet, graph: HeteroGraph, inter: InteractionSet, index: np.ndarray
    ) -> InteractionScores:
        model.eval()
        rows = torch.from_numpy(index).long().to(self.device)
        with torch.no_grad():
            nodes = model.encode(graph)
            logits = model.predictor(nodes, inter.src[rows], inter.dst[rows], inter.relation[rows])
            probs = torch.sigmoid(logits).cpu().numpy()
        labels = inter.label[rows].cpu().numpy()
        return evaluate_interactions(labels, probs)

    def train(
        self,
        model: NcRNANet,
        graph: HeteroGraph,
        inter: InteractionSet,
        train_index: np.ndarray,
        val_index: np.ndarray,
        test_index: np.ndarray,
        prior: torch.Tensor | None = None,
    ) -> FoldResult:
        cfg = self.config.train
        model.to(self.device)
        graph = graph.to(self.device)
        inter = inter.to(self.device)
        if prior is not None:
            prior = prior.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = cosine_warmup(optimizer, cfg.warmup_epochs, cfg.max_epochs)
        averager = ParameterAverage(model, cfg.ema_decay)

        rng = np.random.default_rng(cfg.seed)
        best_auc = float("-inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        stale = 0
        result = FoldResult(best_val_auc=best_auc, test_scores=InteractionScores(0, 0, 0, 0))

        lam = cfg.lambda_pathway if self.config.model.use_pathway_prior else 0.0

        for epoch in range(cfg.max_epochs):
            model.train()
            order = rng.permutation(train_index)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(order), cfg.batch_size):
                rows = (
                    torch.from_numpy(order[start : start + cfg.batch_size]).long().to(self.device)
                )
                optimizer.zero_grad()
                logits = model(graph, inter.src[rows], inter.dst[rows], inter.relation[rows])
                batch_prior = prior[rows] if prior is not None else None
                loss = composite_loss(logits, inter.label[rows], lam, batch_prior).total
                torch.autograd.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                averager.update(model)
                epoch_loss += float(loss.detach())
                steps += 1
            scheduler.step()
            result.train_loss.append(epoch_loss / max(steps, 1))

            val_scores = self._evaluate(model, graph, inter, val_index)
            current = 0.0 if np.isnan(val_scores.auc) else val_scores.auc
            result.val_auc.append(current)
            if current > best_auc:
                best_auc = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= cfg.patience:
                _LOG.info("early stop at epoch %d (best val AUC %.4f)", epoch, best_auc)
                break

        model.load_state_dict(best_state)
        if averager.active:
            averager.copy_to(model)
        result.best_val_auc = best_auc
        result.test_scores = self._evaluate(model, graph, inter, test_index)
        return result
