from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FAMConfig:
    num_classes: int = 19
    ignore_index: int = 255
    margin: float = 0.75
    hard_fraction: float = 0.35
    entropy_weight: float = 0.25
    max_pixels: int = 4096
    eps: float = 1e-6


def _shift_labels(labels: torch.Tensor, dy: int, dx: int, fill: int) -> torch.Tensor:
    """Shift BxHxW labels without wraparound."""
    out = torch.full_like(labels, fill)
    h, w = labels.shape[-2:]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    yd0, yd1 = max(0, dy), min(h, h + dy)
    xd0, xd1 = max(0, dx), min(w, w + dx)
    if ys1 > ys0 and xs1 > xs0:
        out[:, yd0:yd1, xd0:xd1] = labels[:, ys0:ys1, xs0:xs1]
    return out


def resize_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(labels[:, None].float(), size=size, mode="nearest")[:, 0].long()


def semantic_boundary_mask(labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Four-neighbour semantic boundary mask."""
    valid = labels != ignore_index
    boundary = torch.zeros_like(valid)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n = _shift_labels(labels, dy, dx, ignore_index)
        boundary |= valid & (n != ignore_index) & (n != labels)
    return boundary


def boundary_ce_dice(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tau: int = 1,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cheap E1 loss: boundary CE plus class-macro boundary Dice."""
    labels = resize_labels(labels, logits.shape[-2:])
    boundary = semantic_boundary_mask(labels, ignore_index)
    if tau > 1:
        boundary = F.max_pool2d(boundary[:, None].float(), 2 * tau - 1, 1, tau - 1)[:, 0] > 0
    valid = boundary & (labels != ignore_index)
    if not valid.any():
        zero = logits.sum() * 0.0
        return zero, {"boundary_pixels": 0.0, "boundary_ce": 0.0, "boundary_dice": 0.0}

    per_pixel = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    ce = per_pixel[valid].mean()
    probs = logits.softmax(1)
    dice_terms = []
    for c in torch.unique(labels[valid]).tolist():
        if c < 0 or c >= logits.shape[1]:
            continue
        target = (labels == c) & valid
        pred = probs[:, c] * valid
        denom = pred.sum() + target.sum()
        if denom > 0:
            dice_terms.append(1.0 - (2.0 * (pred * target).sum() + eps) / (denom + eps))
    dice = torch.stack(dice_terms).mean() if dice_terms else logits.sum() * 0.0
    loss = ce + dice
    return loss, {
        "boundary_pixels": float(valid.sum().item()),
        "boundary_ce": float(ce.detach().item()),
        "boundary_dice": float(dice.detach().item()),
    }


def fast_adjacency_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    config: FAMConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | float]]:
    """Vectorized direct-adjacency margin loss and sufficient pair statistics.

    For each boundary pixel, the competitor is the highest-logit class among the
    *actual four-neighbour ground-truth classes*. This avoids all connected
    components, profile extraction and grid sampling.
    """
    cfg = config or FAMConfig(num_classes=logits.shape[1])
    b, c, h, w = logits.shape
    labels = resize_labels(labels, (h, w))
    valid_y = (labels >= 0) & (labels < c)
    safe_y = labels.clamp(0, c - 1)
    gt_logit = logits.gather(1, safe_y[:, None])[:, 0]

    best_score = torch.full_like(gt_logit, -torch.inf)
    best_label = torch.full_like(labels, -1)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neigh = _shift_labels(labels, dy, dx, cfg.ignore_index)
        valid = valid_y & (neigh >= 0) & (neigh < c) & (neigh != labels)
        safe_n = neigh.clamp(0, c - 1)
        score = logits.gather(1, safe_n[:, None])[:, 0]
        take = valid & (score > best_score)
        best_score = torch.where(take, score, best_score)
        best_label = torch.where(take, neigh, best_label)

    boundary = best_label >= 0
    if not boundary.any():
        zero = logits.sum() * 0.0
        empty = torch.zeros((c, c), device=logits.device, dtype=torch.float32)
        return zero, {
            "pair_count": empty,
            "violation_sum": empty.clone(),
            "confusion_sum": empty.clone(),
            "active_pixels": 0.0,
            "mean_margin": 0.0,
        }

    margin = gt_logit - best_score
    violation = F.softplus(cfg.margin - margin)
    probs = logits.softmax(1)
    entropy = -(probs.clamp_min(cfg.eps).log() * probs).sum(1) / torch.log(
        torch.tensor(float(c), device=logits.device)
    )
    hard_score = violation.detach() + cfg.entropy_weight * entropy.detach()

    flat_boundary = boundary.flatten()
    indices = flat_boundary.nonzero(as_tuple=False)[:, 0]
    n_keep = max(1, int(indices.numel() * cfg.hard_fraction))
    n_keep = min(n_keep, cfg.max_pixels, indices.numel())
    chosen_local = torch.topk(hard_score.flatten()[indices], k=n_keep, sorted=False).indices
    chosen = indices[chosen_local]

    y = safe_y.flatten()[chosen]
    n = best_label.flatten()[chosen]
    v = violation.flatten()[chosen]
    m = margin.flatten()[chosen]
    confusion = probs.permute(0, 2, 3, 1).reshape(-1, c)[chosen, n]

    if pair_weights is None:
        weights = torch.ones_like(v)
    else:
        weights = pair_weights.to(logits.device, logits.dtype)[y, n]
    loss = (weights * v).sum() / weights.sum().clamp_min(cfg.eps)

    pair_index = y * c + n
    ones = torch.ones_like(v, dtype=torch.float32)
    counts = torch.zeros(c * c, device=logits.device, dtype=torch.float32)
    vsum = torch.zeros_like(counts)
    csum = torch.zeros_like(counts)
    counts.scatter_add_(0, pair_index, ones)
    vsum.scatter_add_(0, pair_index, v.detach().float())
    csum.scatter_add_(0, pair_index, confusion.detach().float())

    return loss, {
        "pair_count": counts.view(c, c),
        "violation_sum": vsum.view(c, c),
        "confusion_sum": csum.view(c, c),
        "active_pixels": float(n_keep),
        "mean_margin": float(m.detach().mean().item()),
    }
