from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


@dataclass
class PairMemoryConfig:
    num_classes: int = 19
    ema_decay: float = 0.9
    scarcity_power: float = 0.5
    violation_weight: float = 0.7
    confusion_weight: float = 0.3
    strength: float = 1.0
    min_weight: float = 0.5
    max_weight: float = 3.0
    min_count: float = 4.0
    eps: float = 1e-6


class PairReliabilityMemory:
    """Server-side 19x19 directed adjacency memory.

    Clients send count/violation/confusion sufficient statistics. The server
    aggregates them, applies EMA, and broadcasts a clipped pair-weight matrix.
    No client feature coordinates are exchanged.
    """

    def __init__(self, config: PairMemoryConfig | None = None):
        self.cfg = config or PairMemoryConfig()
        c = self.cfg.num_classes
        self.count = torch.zeros(c, c, dtype=torch.float64)
        self.violation = torch.zeros(c, c, dtype=torch.float64)
        self.confusion = torch.zeros(c, c, dtype=torch.float64)
        self.weights = torch.ones(c, c, dtype=torch.float32)
        self.round = 0

    def update(self, client_stats: Iterable[Dict[str, torch.Tensor]]) -> torch.Tensor:
        c = self.cfg.num_classes
        total_count = torch.zeros(c, c, dtype=torch.float64)
        total_vsum = torch.zeros_like(total_count)
        total_csum = torch.zeros_like(total_count)
        for stats in client_stats:
            total_count += stats["pair_count"].detach().cpu().double()
            total_vsum += stats["violation_sum"].detach().cpu().double()
            total_csum += stats["confusion_sum"].detach().cpu().double()

        observed = total_count > 0
        round_v = torch.zeros_like(total_count)
        round_c = torch.zeros_like(total_count)
        round_v[observed] = total_vsum[observed] / total_count[observed]
        round_c[observed] = total_csum[observed] / total_count[observed]

        d = self.cfg.ema_decay if self.round > 0 else 0.0
        self.count = d * self.count + (1.0 - d) * total_count
        self.violation = torch.where(
            observed,
            d * self.violation + (1.0 - d) * round_v,
            self.violation,
        )
        self.confusion = torch.where(
            observed,
            d * self.confusion + (1.0 - d) * round_c,
            self.confusion,
        )
        self.round += 1
        self.weights = self._compute_weights()
        return self.weights.clone()

    def _compute_weights(self) -> torch.Tensor:
        cfg = self.cfg
        support = self.count.clone()
        valid = support >= cfg.min_count
        score = cfg.violation_weight * self.violation + cfg.confusion_weight * self.confusion
        scarcity = (support.sum().clamp_min(1.0) / support.clamp_min(1.0)).pow(cfg.scarcity_power)
        raw = score * scarcity
        offdiag = valid & ~torch.eye(cfg.num_classes, dtype=torch.bool)
        weights = torch.ones_like(raw)
        if offdiag.any():
            vals = raw[offdiag]
            median = vals.median()
            mad = (vals - median).abs().median().clamp_min(cfg.eps)
            normalized = (raw - median) / (1.4826 * mad + cfg.eps)
            weights = 1.0 + cfg.strength * torch.tanh(normalized)
            weights = weights.clamp(cfg.min_weight, cfg.max_weight)
            weights[~valid] = 1.0
        weights.fill_diagonal_(1.0)
        return weights.float()

    def state_dict(self) -> Dict[str, torch.Tensor | int]:
        return {
            "count": self.count,
            "violation": self.violation,
            "confusion": self.confusion,
            "weights": self.weights,
            "round": self.round,
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor | int]) -> None:
        self.count = state["count"].clone().double()
        self.violation = state["violation"].clone().double()
        self.confusion = state["confusion"].clone().double()
        self.weights = state["weights"].clone().float()
        self.round = int(state["round"])

    def summary(self) -> Dict[str, float]:
        offdiag = ~torch.eye(self.cfg.num_classes, dtype=torch.bool)
        return {
            "observed_pairs": float(((self.count >= self.cfg.min_count) & offdiag).sum().item()),
            "weight_mean": float(self.weights[offdiag].mean().item()),
            "weight_max": float(self.weights[offdiag].max().item()),
            "weight_min": float(self.weights[offdiag].min().item()),
        }
