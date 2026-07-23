#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${PARTITION:?Set PARTITION}"
: "${PRETRAIN:?Set PRETRAIN}"
: "${OUT_ROOT:?Set OUT_ROOT}"

GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-20}"
SEED="${SEED:-42}"
PARTICIPATION="${PARTICIPATION:-0.3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_ROOT}"
for VARIANT in e1 full; do
  echo "================================================================"
  echo "FedRAB short experiment: ${VARIANT}"
  echo "================================================================"
  "${PYTHON_BIN}" scripts/train_fedrab_v1.py \
    --gpu "${GPU}" \
    --data_root "${DATA_ROOT}" \
    --partition_json "${PARTITION}" \
    --pretrained_path "${PRETRAIN}" \
    --output_dir "${OUT_ROOT}/${VARIANT}_seed${SEED}" \
    --variant "${VARIANT}" \
    --rounds "${ROUNDS}" \
    --participation "${PARTICIPATION}" \
    --seed "${SEED}"
done
