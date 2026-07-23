#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def final_eval(history):
    rows = [row for row in history if "miou" in row]
    if not rows:
        raise ValueError("No evaluation rows in history")
    return rows[-1]


def total_train_seconds(history):
    return sum(float(row.get("train_seconds", 0.0)) for row in history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    records = {}
    for variant in ("e1", "full"):
        path = root / f"{variant}_seed{args.seed}" / "history.json"
        history = json.loads(path.read_text())
        records[variant] = (final_eval(history), total_train_seconds(history))

    e1, e1_time = records["e1"]
    full, full_time = records["full"]
    keys = ("miou", "boundary_miou", "pair_boundary_iou", "pixel_accuracy")
    print(f"E1 train seconds:   {e1_time:.1f}")
    print(f"Full train seconds: {full_time:.1f}")
    print(f"Overhead:           {(full_time / max(e1_time, 1e-9) - 1.0) * 100:.2f}%")
    print()
    for key in keys:
        a = float(e1.get(key, 0.0))
        b = float(full.get(key, 0.0))
        print(f"{key:24s} E1={a:.6f} Full={b:.6f} Delta={(b-a)*100:+.3f} pp")


if __name__ == "__main__":
    main()
