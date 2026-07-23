#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedrab.aggregation import CGFAConfig, aggregate_classifier_rows, class_group_rebalance
from fedrab.boundary import FAMConfig, boundary_ce_dice, fast_adjacency_margin
from fedrab.data import build_datasets
from fedrab.lora import LoRALinear, lora_modules
from fedrab.memory import PairMemoryConfig, PairReliabilityMemory
from fedrab.metrics import SegmentationMetrics
from fedrab.model import ModelConfig, build_model, parameter_report


TIER = {
    0: ("medium", 8, 2, 2), 1: ("medium", 8, 2, 2),
    2: ("weak", 4, 1, 1), 3: ("weak", 4, 1, 1),
    4: ("medium", 8, 2, 2), 5: ("strong", 16, 3, 4),
    6: ("strong", 16, 3, 4), 7: ("weak", 4, 1, 1),
    8: ("medium", 8, 2, 2), 9: ("strong", 16, 3, 4),
}


@dataclass
class ClientUpdate:
    client_id: int
    num_samples: int
    non_lora: Dict[str, torch.Tensor]
    lora_delta: Dict[str, torch.Tensor]
    classifier_weight: torch.Tensor
    class_count: torch.Tensor
    pair_stats: Dict[str, torch.Tensor]
    mean_loss: float
    seconds: float


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "train.log", mode="w")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)


def cosine_lr(base: float, minimum: float, round_idx: int, rounds: int) -> float:
    if rounds <= 1:
        return minimum
    phase = (round_idx - 1) / (rounds - 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * phase))


def classifier_name(model: torch.nn.Module) -> str:
    for name, parameter in model.named_parameters():
        if name.endswith("decode_head.classifier.weight"):
            return name
    raise KeyError("classifier weight not found")


@torch.no_grad()
def sync_from_global(local: torch.nn.Module, global_model: torch.nn.Module) -> None:
    local_state = local.state_dict()
    global_state = global_model.state_dict()
    for name, tensor in local_state.items():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        if name in global_state and tensor.shape == global_state[name].shape:
            tensor.copy_(global_state[name].to(tensor.device, tensor.dtype))
    gl = lora_modules(global_model)
    ll = lora_modules(local)
    for name in ll:
        ll[name].set_effective_delta(gl[name].effective_delta())


def collect_update(
    model: torch.nn.Module,
    client_id: int,
    num_samples: int,
    class_count: torch.Tensor,
    pair_stats: Dict[str, torch.Tensor],
    mean_loss: float,
    seconds: float,
) -> ClientUpdate:
    lora_parameter_names = {
        f"{name}.lora_A" for name in lora_modules(model)
    } | {f"{name}.lora_B" for name in lora_modules(model)}
    non_lora = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and name not in lora_parameter_names
    }
    deltas = {
        name: module.effective_delta().detach().cpu().clone()
        for name, module in lora_modules(model).items()
    }
    cname = classifier_name(model)
    return ClientUpdate(
        client_id=client_id,
        num_samples=num_samples,
        non_lora=non_lora,
        lora_delta=deltas,
        classifier_weight=non_lora[cname].clone(),
        class_count=class_count.detach().cpu().float(),
        pair_stats={k: v.detach().cpu().float() for k, v in pair_stats.items()},
        mean_loss=mean_loss,
        seconds=seconds,
    )


def train_client(
    model: torch.nn.Module,
    loader: DataLoader,
    client_id: int,
    epochs: int,
    lr: float,
    device: torch.device,
    pair_weights: torch.Tensor,
    variant: str,
    round_idx: int,
    args,
) -> ClientUpdate:
    start = time.time()
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    c = args.num_classes
    class_count = torch.zeros(c, device=device)
    pair_count = torch.zeros(c, c, device=device)
    violation_sum = torch.zeros_like(pair_count)
    confusion_sum = torch.zeros_like(pair_count)
    losses: List[float] = []
    fam_ramp = min(1.0, max(0.0, (round_idx - 1) / max(args.fam_ramp_rounds, 1)))
    full = variant == "full"

    for _ in range(epochs):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                low = model(pixel_values=images).logits
                full_logits = F.interpolate(low, labels.shape[-2:], mode="bilinear", align_corners=False)
                seg = F.cross_entropy(full_logits, labels, ignore_index=255)
                bd, _ = boundary_ce_dice(low, labels, tau=args.boundary_tau)
                loss = seg + args.lambda_boundary * bd
                if full and fam_ramp > 0:
                    fam, stats = fast_adjacency_margin(
                        low,
                        labels,
                        pair_weights=pair_weights,
                        config=FAMConfig(
                            num_classes=c,
                            margin=args.fam_margin,
                            hard_fraction=args.fam_hard_fraction,
                            max_pixels=args.fam_max_pixels,
                        ),
                    )
                    loss = loss + args.lambda_fam * fam_ramp * fam
                    pair_count += stats["pair_count"]
                    violation_sum += stats["violation_sum"]
                    confusion_sum += stats["confusion_sum"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            valid = (labels >= 0) & (labels < c)
            class_count += torch.bincount(labels[valid], minlength=c).float()

    pair_stats = {
        "pair_count": pair_count,
        "violation_sum": violation_sum,
        "confusion_sum": confusion_sum,
    }
    update = collect_update(
        model, client_id, len(loader.dataset), class_count, pair_stats,
        float(np.mean(losses)) if losses else 0.0, time.time() - start,
    )
    model.to("cpu")
    torch.cuda.empty_cache()
    return update


@torch.no_grad()
def aggregate_round(global_model: torch.nn.Module, updates: List[ClientUpdate], memory: PairReliabilityMemory, variant: str):
    sample = torch.tensor([u.num_samples for u in updates], dtype=torch.float64)
    sample /= sample.sum().clamp_min(1.0)
    named = dict(global_model.named_parameters())
    cname = classifier_name(global_model)
    previous_classifier = named[cname].detach().clone()

    for name, parameter in named.items():
        if not parameter.requires_grad or name.endswith("lora_A") or name.endswith("lora_B") or name == cname:
            continue
        avg = sum(sample[i] * updates[i].non_lora[name].double() for i in range(len(updates)))
        parameter.copy_(avg.to(parameter.device, parameter.dtype))

    global_lora = lora_modules(global_model)
    for name, module in global_lora.items():
        delta = sum(sample[i] * updates[i].lora_delta[name].double() for i in range(len(updates)))
        module.set_effective_delta(delta)

    row_aggregated = aggregate_classifier_rows(
        previous_classifier,
        [u.classifier_weight for u in updates],
        [u.class_count for u in updates],
        [u.num_samples for u in updates],
    )
    cgfa_stats = {"cgfa_applied": 0.0}
    if variant == "full":
        pair_weights = memory.update([u.pair_stats for u in updates])
        support = memory.count.sum(1).float().to(row_aggregated.device)
        row_aggregated, cgfa_stats = class_group_rebalance(
            previous_classifier, row_aggregated, support, CGFAConfig()
        )
    else:
        pair_weights = torch.ones(memory.cfg.num_classes, memory.cfg.num_classes)
    named[cname].copy_(row_aggregated.to(named[cname].device, named[cname].dtype))
    return pair_weights, cgfa_stats


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.to(device).eval()
    metrics = SegmentationMetrics(num_classes=num_classes, boundary_tau=1)
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(pixel_values=images).logits
        metrics.update(logits, labels)
    model.to("cpu")
    torch.cuda.empty_cache()
    return metrics.compute()


def save_checkpoint(path: Path, model, memory, round_idx: int, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "round": round_idx,
        "model": model.state_dict(),
        "pair_memory": memory.state_dict(),
        "history": history,
    }, path)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output_dir)
    configure_logging(output)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("variant=%s device=%s output=%s", args.variant, device, output)

    client_sets, val_set = build_datasets(args.data_root, args.partition_json, (args.crop_h, args.crop_w))
    models = {}
    for rank in (4, 8, 16):
        seed_everything(args.seed)
        models[rank] = build_model(ModelConfig(args.pretrained_path, args.num_classes, rank))
    global_model = models[16]
    for rank in (4, 8):
        sync_from_global(models[rank], global_model)
    logging.info("model=%s", parameter_report(global_model))

    loaders = {}
    for client_id, dataset in client_sets.items():
        _, _, _, batch = TIER.get(client_id, ("medium", 8, 2, 2))
        generator = torch.Generator().manual_seed(args.seed + client_id)
        loaders[client_id] = DataLoader(
            dataset, batch_size=batch, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=False, generator=generator,
        )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    memory = PairReliabilityMemory(PairMemoryConfig(num_classes=args.num_classes))
    pair_weights = torch.ones(args.num_classes, args.num_classes)
    history = []
    best_joint = -1.0
    client_ids = sorted(client_sets)
    rng = random.Random(args.seed)

    for round_idx in range(1, args.rounds + 1):
        n_select = max(1, int(math.ceil(len(client_ids) * args.participation)))
        selected = sorted(rng.sample(client_ids, n_select))
        lr = cosine_lr(args.lr, args.lr_min, round_idx, args.rounds)
        logging.info("Round %d/%d clients=%s lr=%.3e", round_idx, args.rounds, selected, lr)
        updates = []
        for client_id in selected:
            tier, rank, epochs, _ = TIER.get(client_id, ("medium", 8, 2, 2))
            local = models[rank]
            sync_from_global(local, global_model)
            update = train_client(
                local, loaders[client_id], client_id, epochs, lr, device,
                pair_weights, args.variant, round_idx, args,
            )
            updates.append(update)
            logging.info(
                "client=%d tier=%s rank=%d loss=%.4f time=%.1fs samples=%d",
                client_id, tier, rank, update.mean_loss, update.seconds, update.num_samples,
            )
        pair_weights, cgfa = aggregate_round(global_model, updates, memory, args.variant)
        record = {
            "round": round_idx,
            "lr": lr,
            "clients": selected,
            "train_loss": float(np.mean([u.mean_loss for u in updates])),
            "train_seconds": float(sum(u.seconds for u in updates)),
            "memory": memory.summary() if args.variant == "full" else {},
            "cgfa": cgfa,
        }
        if round_idx % args.eval_every == 0 or round_idx == args.rounds:
            result = evaluate(global_model, val_loader, device, args.num_classes)
            record.update(result)
            joint = result["miou"] + result["boundary_miou"]
            logging.info("eval=%s joint=%.4f", result, joint)
            if joint > best_joint:
                best_joint = joint
                save_checkpoint(output / "best_joint_model.pt", global_model, memory, round_idx, history + [record])
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if round_idx % args.save_every == 0:
            save_checkpoint(output / f"round_{round_idx}.pt", global_model, memory, round_idx, history)

    save_checkpoint(output / "last_model.pt", global_model, memory, args.rounds, history)
    logging.info("Training completed. best_joint=%.4f", best_joint)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--partition_json", required=True)
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", choices=("e1", "full"), default="full")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--participation", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--lambda_boundary", type=float, default=0.5)
    p.add_argument("--boundary_tau", type=int, default=1)
    p.add_argument("--lambda_fam", type=float, default=0.15)
    p.add_argument("--fam_margin", type=float, default=0.75)
    p.add_argument("--fam_hard_fraction", type=float, default=0.35)
    p.add_argument("--fam_max_pixels", type=int, default=4096)
    p.add_argument("--fam_ramp_rounds", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--amp", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
