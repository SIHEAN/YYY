from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch


@dataclass
class CGFAConfig:
    support_power: float = 0.5
    group_strength: float = 0.5
    factor_min: float = 0.75
    factor_max: float = 1.35
    eps: float = 1e-8


def _flatten_rows(weight: torch.Tensor) -> torch.Tensor:
    return weight.reshape(weight.shape[0], -1)


@torch.no_grad()
def aggregate_classifier_rows(
    previous_weight: torch.Tensor,
    client_weights: Sequence[torch.Tensor],
    class_counts: Sequence[torch.Tensor],
    sample_weights: Sequence[float],
    config: CGFAConfig | None = None,
) -> torch.Tensor:
    """Class-conditional row aggregation.

    A client contributes strongly to class c only when it has evidence for c.
    The aggregation is performed on *updates* from the shared round-start row.
    """
    cfg = config or CGFAConfig()
    prev = _flatten_rows(previous_weight).double().cpu()
    client = [_flatten_rows(w).double().cpu() for w in client_weights]
    counts = [x.double().cpu().clamp_min(0.0) for x in class_counts]
    sw = torch.tensor(sample_weights, dtype=torch.float64)
    sw = sw / sw.sum().clamp_min(cfg.eps)

    out = prev.clone()
    for c in range(prev.shape[0]):
        reliability = torch.stack([
            sw[i] * (counts[i][c] + 1.0).pow(cfg.support_power)
            for i in range(len(client))
        ])
        reliability = reliability / reliability.sum().clamp_min(cfg.eps)
        delta = sum(reliability[i] * (client[i][c] - prev[c]) for i in range(len(client)))
        out[c] = prev[c] + delta
    return out.reshape_as(previous_weight).to(previous_weight.device, previous_weight.dtype)


@torch.no_grad()
def class_group_rebalance(
    previous_weight: torch.Tensor,
    aggregated_weight: torch.Tensor,
    boundary_support: torch.Tensor,
    config: CGFAConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conservatively equalize rare/mid/common classifier-row update energy.

    The total Frobenius norm is preserved. Group factors are clipped, so the
    operation cannot arbitrarily amplify the classifier update.
    """
    cfg = config or CGFAConfig()
    prev = _flatten_rows(previous_weight).double()
    new = _flatten_rows(aggregated_weight).double()
    delta = new - prev
    base_norm = delta.norm().clamp_min(cfg.eps)

    support = boundary_support.double().to(delta.device)
    if support.numel() != delta.shape[0] or (support > 0).sum() < 3:
        return aggregated_weight, {"cgfa_applied": 0.0, "cgfa_norm": float(base_norm.item())}

    q1 = torch.quantile(support, 1.0 / 3.0)
    q2 = torch.quantile(support, 2.0 / 3.0)
    groups = [support <= q1, (support > q1) & (support <= q2), support > q2]
    rms = []
    for mask in groups:
        if mask.any():
            rms.append(delta[mask].norm(dim=1).square().mean().sqrt())
        else:
            rms.append(torch.tensor(0.0, device=delta.device, dtype=delta.dtype))
    positive = torch.stack([x for x in rms if x > cfg.eps])
    if positive.numel() == 0:
        return aggregated_weight, {"cgfa_applied": 0.0, "cgfa_norm": float(base_norm.item())}
    target = positive.median()

    adjusted = delta.clone()
    factors = []
    for mask, value in zip(groups, rms):
        if not mask.any() or value <= cfg.eps:
            factors.append(1.0)
            continue
        desired = (target / value).clamp(cfg.factor_min, cfg.factor_max)
        factor = 1.0 + cfg.group_strength * (desired - 1.0)
        adjusted[mask] *= factor
        factors.append(float(factor.item()))

    adjusted *= base_norm / adjusted.norm().clamp_min(cfg.eps)
    result = (prev + adjusted).reshape_as(aggregated_weight)
    return result.to(aggregated_weight.dtype), {
        "cgfa_applied": 1.0,
        "cgfa_norm": float(adjusted.norm().item()),
        "rare_factor": factors[0],
        "mid_factor": factors[1],
        "common_factor": factors[2],
    }
