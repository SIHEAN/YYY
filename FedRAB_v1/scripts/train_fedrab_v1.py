#!/usr/bin/env python3
"""Corrected FedRAB-v1 entry point.

Two protocol invariants are enforced here:
1. the rank-16 local client model is separate from the immutable round-start
   global model, so sequential client training cannot leak into later clients;
2. repeated local epochs do not count as independent class/pair evidence.
"""

import json
import logging
import math
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_fedrab as base
from fedrab.data import build_datasets
from fedrab.memory import PairMemoryConfig, PairReliabilityMemory
from fedrab.model import ModelConfig, build_model, parameter_report


_original_train_client = base.train_client


def evidence_normalized_train_client(
    model, loader, client_id, epochs, lr, device, pair_weights, variant, round_idx, args
):
    update = _original_train_client(
        model, loader, client_id, epochs, lr, device,
        pair_weights, variant, round_idx, args,
    )
    repeats = float(max(int(epochs), 1))
    update.class_count.div_(repeats)
    for value in update.pair_stats.values():
        value.div_(repeats)
    return update


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output_dir)
    base.configure_logging(output)
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("variant=%s device=%s output=%s", args.variant, device, output)

    client_sets, val_set = build_datasets(
        args.data_root, args.partition_json, (args.crop_h, args.crop_w)
    )

    base.seed_everything(args.seed)
    global_model = build_model(ModelConfig(args.pretrained_path, args.num_classes, 16))
    local_models = {}
    for rank in (4, 8, 16):
        base.seed_everything(args.seed)
        local_models[rank] = build_model(
            ModelConfig(args.pretrained_path, args.num_classes, rank)
        )
        base.sync_from_global(local_models[rank], global_model)
    assert local_models[16] is not global_model
    logging.info("server_model=%s", parameter_report(global_model))

    loaders = {}
    for client_id, dataset in client_sets.items():
        _, _, _, batch = base.TIER.get(client_id, ("medium", 8, 2, 2))
        generator = torch.Generator().manual_seed(args.seed + client_id)
        loaders[client_id] = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            generator=generator,
        )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    memory = PairReliabilityMemory(PairMemoryConfig(num_classes=args.num_classes))
    pair_weights = torch.ones(args.num_classes, args.num_classes)
    history = []
    best_joint = -1.0
    client_ids = sorted(client_sets)
    rng = random.Random(args.seed)

    for round_idx in range(1, args.rounds + 1):
        n_select = max(1, int(math.ceil(len(client_ids) * args.participation)))
        selected = sorted(rng.sample(client_ids, n_select))
        lr = base.cosine_lr(args.lr, args.lr_min, round_idx, args.rounds)
        logging.info("Round %d/%d clients=%s lr=%.3e", round_idx, args.rounds, selected, lr)
        updates = []

        for client_id in selected:
            tier, rank, epochs, _ = base.TIER.get(client_id, ("medium", 8, 2, 2))
            local = local_models[rank]
            base.sync_from_global(local, global_model)
            update = evidence_normalized_train_client(
                local,
                loaders[client_id],
                client_id,
                epochs,
                lr,
                device,
                pair_weights,
                args.variant,
                round_idx,
                args,
            )
            updates.append(update)
            logging.info(
                "client=%d tier=%s rank=%d loss=%.4f time=%.1fs samples=%d",
                client_id, tier, rank, update.mean_loss, update.seconds, update.num_samples,
            )

        pair_weights, cgfa = base.aggregate_round(
            global_model, updates, memory, args.variant
        )
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
            result = base.evaluate(global_model, val_loader, device, args.num_classes)
            record.update(result)
            joint = result["miou"] + result["boundary_miou"]
            logging.info("eval=%s joint=%.4f", result, joint)
            if joint > best_joint:
                best_joint = joint
                base.save_checkpoint(
                    output / "best_joint_model.pt",
                    global_model,
                    memory,
                    round_idx,
                    history + [record],
                )
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if round_idx % args.save_every == 0:
            base.save_checkpoint(
                output / f"round_{round_idx}.pt",
                global_model,
                memory,
                round_idx,
                history,
            )

    base.save_checkpoint(
        output / "last_model.pt", global_model, memory, args.rounds, history
    )
    logging.info("Training completed. best_joint=%.4f", best_joint)


if __name__ == "__main__":
    run(base.parse_args())
