# FedRAB-v1 method and result audit

## Why the earliest Fixed-v2 chain looked best

The strongest archived numbers came from a staged chain rather than a fair equal-budget comparison. The Fixed-v2 audit reports B0 0.6366/0.3104, E1 0.6527/0.3462, E2 0.6639/0.3617 and E3 0.6713/0.3622 for mIoU/boundary-mIoU. However, E2 started from E1 and changed participation, learning rate, losses, class weighting, decoder learning-rate multiplier and aggregation, then trained another 100 rounds. E3 added another 100 rounds. Therefore the apparent superiority mixes useful mechanisms with extra optimization budget.

The audit nevertheless gives two reliable signals:

1. explicit boundary supervision is the dominant boundary contributor;
2. class-aware global consistency and classifier-row aggregation are more promising than an expensive geometric profile loss.

FedRAB retains these two signals and removes the mechanisms that were slow, collapsed, or difficult to attribute.

## FedRAB-v1

### 1. FAM: Fast Adjacency Margin

For a ground-truth class `y(x)`, inspect only the four direct neighbours. Among the neighbouring classes that differ from `y`, choose the one with the largest current logit, `n(x)`. The loss is

```
softplus(gamma - (z_y - z_n)).
```

Only the hardest fraction of boundary pixels is used. The implementation is composed of tensor shifts, `gather`, `topk`, and `scatter_add`; it does not use connected components, signed-distance normals, multi-point profiles, or `grid_sample`.

### 2. PRM: Pair Reliability Memory

Each client uploads three directed `19 x 19` sufficient-statistic matrices:

- observed pair counts;
- summed margin violations;
- summed competitor probabilities.

The server forms EMA estimates and a robust clipped pair-weight matrix. Rare and consistently difficult pairs receive more weight in the next round. No client feature coordinates are exchanged.

### 3. CGFA: Class-Grouped Function-space Aggregation

The implementation keeps effective-update LoRA aggregation: it averages `scale * B @ A`, rather than separately averaging incompatible low-rank factors. Classifier rows are aggregated using class evidence. Rare/mid/common groups are then conservatively equalized, with clipped factors and exact preservation of the total classifier-update norm.

## Relationship to recent work

- BRDG (CVPR 2026) supports boundary-responsive selective refinement and adjacency-focused hard negatives, but FedRAB does not copy its superpixel architecture.
- FedBCS (AAAI 2026) supports style-aware contextual prototype alignment, but FedRAB deliberately avoids feature-prototype exchange because the current driving-scene code already showed that expensive feature geometry did not justify its cost.
- FedSaaS (IJCAI 2025) supports class-level global supervision for federated semantic segmentation.
- FedCGNM (2026) supports class-grouped gradient normalization for federated class imbalance. FedRAB uses a different server-side segmentation-specific classifier-row rebalance and does not claim the general normalization idea as new.
- UniFLoW (ICML 2026) independently reinforces the need to aggregate LoRA in effective-update/function space.

## Claims

FedRAB is a research implementation designed to maximize the probability of a useful accuracy/efficiency trade-off. It is not claimed to be guaranteed to improve Cityscapes before real controlled runs. The mandatory decision test is an equal-seed, equal-round comparison of `e1` and `full`, with both accuracy and wall-clock overhead reported.
