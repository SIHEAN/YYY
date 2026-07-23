from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from .boundary import semantic_boundary_mask


@dataclass
class SegmentationMetrics:
    num_classes: int = 19
    ignore_index: int = 255
    boundary_tau: int = 1

    def __post_init__(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        self.boundary_inter = torch.zeros(self.num_classes, dtype=torch.float64)
        self.boundary_union = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pair_match = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        self.pair_union = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if tuple(logits.shape[-2:]) != tuple(labels.shape[-2:]):
            logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu()
        labels = labels.cpu()
        valid = labels != self.ignore_index
        index = labels[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(
            index, minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes).double()
        self._update_boundary(pred, labels)
        self._update_pairs(pred, labels)

    def _update_boundary(self, pred: torch.Tensor, labels: torch.Tensor) -> None:
        gt_boundary = semantic_boundary_mask(labels, self.ignore_index)
        pr_boundary = semantic_boundary_mask(pred, self.ignore_index)
        if self.boundary_tau > 1:
            k = 2 * self.boundary_tau - 1
            gt_boundary = F.max_pool2d(gt_boundary[:, None].float(), k, 1, self.boundary_tau - 1)[:, 0] > 0
            pr_boundary = F.max_pool2d(pr_boundary[:, None].float(), k, 1, self.boundary_tau - 1)[:, 0] > 0
        for c in range(self.num_classes):
            gt = gt_boundary & (labels == c)
            pr = pr_boundary & (pred == c)
            self.boundary_inter[c] += (gt & pr).sum().item()
            self.boundary_union[c] += (gt | pr).sum().item()

    def _pair_edge_map(self, labels: torch.Tensor) -> Dict[Tuple[int, int], torch.Tensor]:
        result: Dict[Tuple[int, int], torch.Tensor] = {}
        b, h, w = labels.shape
        for dy, dx in ((1, 0), (0, 1)):
            current = labels[:, : h - dy if dy else h, : w - dx if dx else w]
            neigh = labels[:, dy:, dx:]
            valid = (
                (current >= 0) & (current < self.num_classes)
                & (neigh >= 0) & (neigh < self.num_classes)
                & (current != neigh)
            )
            if not valid.any():
                continue
            lo = torch.minimum(current, neigh)
            hi = torch.maximum(current, neigh)
            for code in torch.unique((lo * self.num_classes + hi)[valid]).tolist():
                a, c = divmod(int(code), self.num_classes)
                mask_small = valid & (lo == a) & (hi == c)
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[:, : h - dy if dy else h, : w - dx if dx else w] |= mask_small
                if dy:
                    mask[:, dy:, :] |= mask_small
                else:
                    mask[:, :, dx:] |= mask_small
                result[(a, c)] = result.get((a, c), torch.zeros_like(mask)) | mask
        return result

    def _update_pairs(self, pred: torch.Tensor, labels: torch.Tensor) -> None:
        gt = self._pair_edge_map(labels)
        pr = self._pair_edge_map(pred)
        keys = set(gt) | set(pr)
        for a, b in keys:
            gm = gt.get((a, b), torch.zeros_like(labels, dtype=torch.bool))
            pm = pr.get((a, b), torch.zeros_like(labels, dtype=torch.bool))
            if self.boundary_tau > 0:
                k = 2 * self.boundary_tau + 1
                gd = F.max_pool2d(gm[:, None].float(), k, 1, self.boundary_tau)[:, 0] > 0
                pd = F.max_pool2d(pm[:, None].float(), k, 1, self.boundary_tau)[:, 0] > 0
            else:
                gd, pd = gm, pm
            matched_gt = gm & pd
            matched_pr = pm & gd
            inter = 0.5 * (matched_gt.sum().item() + matched_pr.sum().item())
            union = gm.sum().item() + pm.sum().item() - inter
            self.pair_match[a, b] += inter
            self.pair_union[a, b] += max(union, 0.0)

    def compute(self) -> Dict[str, float]:
        tp = self.confusion.diag()
        union = self.confusion.sum(0) + self.confusion.sum(1) - tp
        iou = tp / union.clamp_min(1.0)
        present = union > 0
        boundary_iou = self.boundary_inter / self.boundary_union.clamp_min(1.0)
        boundary_present = self.boundary_union > 0
        pair_iou = self.pair_match / self.pair_union.clamp_min(1.0)
        pair_present = self.pair_union > 0
        return {
            "miou": float(iou[present].mean().item()),
            "pixel_accuracy": float(tp.sum().item() / self.confusion.sum().clamp_min(1.0).item()),
            "boundary_miou": float(boundary_iou[boundary_present].mean().item()),
            "pair_boundary_iou": float(pair_iou[pair_present].mean().item()) if pair_present.any() else 0.0,
            "evaluated_pairs": float(pair_present.sum().item()),
        }
