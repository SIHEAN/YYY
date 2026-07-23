# FedRAB-v1

**Federated Relation-Adaptive Boundary Segmentation** is a lightweight replacement for the expensive N-OIMC/RIGF prototype.

It keeps the reliable parts of Fixed-v2 (heterogeneous LoRA, function-space aggregation, GroupNorm, explicit boundary supervision) and adds three jointly-trained modules:

1. **FAM — Fast Adjacency Margin**: vectorized four-neighbour class-pair margin loss on hard boundary pixels. No connected components, normal profiles, `grid_sample`, or extra model forward.
2. **PRM — Pair Reliability Memory**: clients upload only `19 x 19` directed pair statistics (count, violation, confusion). The server forms an EMA pair-weight memory and broadcasts it for the next round.
3. **CGFA — Class-Grouped Function-space Aggregation**: correct effective-update LoRA aggregation plus conservative class-row update rebalancing based on global boundary support.

The method has no inference-time branch and the pair-memory communication is under 5 KiB/client/round in FP32.

## Install

```bash
cd FedRAB_v1
pip install -r requirements.txt
```

## Short experiment

```bash
nohup env \
  GPU=0 \
  DATA_ROOT=/home/zkpk/zxy-2/over/dataset \
  PARTITION=/home/zkpk/zxy-2/over/dataset/partitions/alpha_0.1_K10_seed42/partition.json \
  PRETRAIN=/home/zkpk/zxy-2/over/pretrained/mit-b2 \
  OUT_ROOT=/home/zkpk/zxy-2/FedRAB/short_seed42 \
  ROUNDS=20 SEED=42 PARTICIPATION=0.3 \
  bash scripts/run_short.sh \
  > fedrab_short_seed42.log 2>&1 &
```

The script independently trains `e1` and `full` from the same seed. Do not claim improvement until the real Cityscapes runs finish.

## Expected speed

FAM uses tensor shifts, gather and scatter operations at decoder resolution. PRM and CGFA are server-side matrix operations over 19 classes. The intended overhead is below 20% relative to E1, but this must be measured on the user's GPU.
