#!/usr/bin/env python3
"""Validated FedRAB-v1 entry point.

This wrapper keeps the main trainer readable while enforcing a critical
federated-evidence invariant: repeated local epochs must not be counted as
independent class/pair observations. Model optimization still uses the full
1/2/3 local epochs; uploaded support statistics are normalized to one local
pass before server reliability weighting.
"""

import train_fedrab as base


_original_train_client = base.train_client


def _evidence_normalized_train_client(
    model,
    loader,
    client_id,
    epochs,
    lr,
    device,
    pair_weights,
    variant,
    round_idx,
    args,
):
    update = _original_train_client(
        model,
        loader,
        client_id,
        epochs,
        lr,
        device,
        pair_weights,
        variant,
        round_idx,
        args,
    )
    repeats = float(max(int(epochs), 1))
    update.class_count.div_(repeats)
    for value in update.pair_stats.values():
        value.div_(repeats)
    return update


base.train_client = _evidence_normalized_train_client


if __name__ == "__main__":
    base.run(base.parse_args())
