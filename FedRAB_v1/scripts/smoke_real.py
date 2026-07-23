#!/usr/bin/env python3
"""Real MIT-B2 + Cityscapes smoke test for FedRAB-v1.

Runs one deterministic batch for ranks 4/8/16, verifies E1 and FAM backward,
measures the incremental FAM cost, and performs one synthetic server aggregation.
It does not train a full federated round.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedrab.boundary import FAMConfig, boundary_ce_dice, fast_adjacency_margin
from fedrab.data import build_datasets
from fedrab.model import ModelConfig, build_model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--partition_json", required=True)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--client_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    clients, _ = build_datasets(args.data_root, args.partition_json)
    if args.client_id not in clients:
        raise KeyError(f"Client {args.client_id} not present; available={sorted(clients)}")
    loader = DataLoader(clients[args.client_id], batch_size=1, shuffle=False, num_workers=0)
    images, labels, sample_ids = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    for rank in (4, 8, 16):
        model = build_model(ModelConfig(args.pretrained_path, 19, rank)).to(device).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        sync(device)
        t0 = time.perf_counter()
        low = model(pixel_values=images).logits
        full = F.interpolate(low, labels.shape[-2:], mode="bilinear", align_corners=False)
        seg = F.cross_entropy(full, labels, ignore_index=255)
        bd, _ = boundary_ce_dice(full, labels, tau=1)
        base_loss = seg + 0.5 * bd
        sync(device)
        base_forward = time.perf_counter() - t0

        t1 = time.perf_counter()
        fam, stats = fast_adjacency_margin(
            low,
            labels,
            config=FAMConfig(num_classes=19, max_pixels=4096),
        )
        sync(device)
        fam_forward = time.perf_counter() - t1

        loss = base_loss + 0.15 * fam
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"non-finite loss at rank {rank}"
        assert stats["active_pixels"] > 0, f"no FAM pixels at rank {rank}"
        print(
            f"rank={rank} sample={sample_ids[0]} loss={loss.item():.4f} "
            f"base_forward={base_forward:.4f}s fam_extra={fam_forward:.4f}s "
            f"fam_ratio={fam_forward/max(base_forward,1e-9):.3f} "
            f"active={stats['active_pixels']:.0f}"
        )
        del model, optimizer, low, full, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("REAL_SMOKE_PASS")


if __name__ == "__main__":
    main()
