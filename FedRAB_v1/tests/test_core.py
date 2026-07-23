import torch
from torch import nn

from fedrab.aggregation import aggregate_classifier_rows, class_group_rebalance
from fedrab.boundary import FAMConfig, boundary_ce_dice, fast_adjacency_margin
from fedrab.lora import LoRALinear
from fedrab.memory import PairMemoryConfig, PairReliabilityMemory


def _fam_case():
    logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
    logits.data[:, 0, :, :4] = 2.0
    logits.data[:, 1, :, 4:] = 2.0
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, :, 4:] = 1
    loss, stats = fast_adjacency_margin(
        logits, labels, config=FAMConfig(num_classes=3, max_pixels=64)
    )
    return logits, loss, stats


def test_fam_forward_is_finite():
    _, loss, _ = _fam_case()
    assert bool(torch.isfinite(loss).item())


def test_fam_selects_boundary_pixels():
    _, _, stats = _fam_case()
    assert float(stats["active_pixels"]) > 0.0


def test_fam_records_direct_pair():
    _, _, stats = _fam_case()
    observed = stats["pair_count"][0, 1] + stats["pair_count"][1, 0]
    assert float(observed.item()) > 0.0


def test_fam_backward_is_finite():
    logits, loss, _ = _fam_case()
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())


def test_non_adjacent_class_is_not_used_as_competitor():
    logits = torch.zeros(1, 3, 6, 6, requires_grad=True)
    logits.data[:, 2] = 20.0
    labels = torch.zeros(1, 6, 6, dtype=torch.long)
    labels[:, :, 3:] = 1
    _, stats = fast_adjacency_margin(logits, labels, config=FAMConfig(num_classes=3))
    assert float(stats["pair_count"][:, 2].sum().item()) == 0.0


def test_boundary_loss_handles_empty_boundary():
    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    loss, stats = boundary_ce_dice(logits, labels)
    assert loss.item() == 0.0
    assert stats["boundary_pixels"] == 0.0


def test_pair_memory_upweights_reliable_hard_pair():
    memory = PairReliabilityMemory(PairMemoryConfig(num_classes=3, ema_decay=0.0, min_count=1))
    count = torch.zeros(3, 3)
    violation = torch.zeros(3, 3)
    confusion = torch.zeros(3, 3)
    count[0, 1] = 10
    count[1, 2] = 10
    violation[0, 1] = 20
    violation[1, 2] = 2
    confusion[0, 1] = 8
    confusion[1, 2] = 1
    weights = memory.update([{
        "pair_count": count,
        "violation_sum": violation,
        "confusion_sum": confusion,
    }])
    assert weights[0, 1] > weights[1, 2]
    assert weights.diag().eq(1).all()


def test_classifier_rows_use_class_evidence():
    prev = torch.zeros(3, 2)
    w1 = torch.tensor([[2.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
    w2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    c1 = torch.tensor([100.0, 1.0, 0.0])
    c2 = torch.tensor([0.0, 1.0, 100.0])
    out = aggregate_classifier_rows(prev, [w1, w2], [c1, c2], [1.0, 1.0])
    assert out[0, 0] > 1.5
    assert out[2, 0] > 3.0


def test_group_rebalance_preserves_total_update_norm():
    torch.manual_seed(7)
    prev = torch.zeros(6, 4)
    new = torch.randn(6, 4)
    support = torch.tensor([1.0, 2.0, 5.0, 20.0, 50.0, 100.0])
    out, stats = class_group_rebalance(prev, new, support)
    assert stats["cgfa_applied"] == 1.0
    assert torch.allclose((out - prev).norm(), (new - prev).norm(), rtol=1e-5, atol=1e-6)


def test_lora_effective_delta_round_trip():
    torch.manual_seed(11)
    base = nn.Linear(5, 4, bias=False)
    layer = LoRALinear(base, rank=3, alpha=6.0)
    target = torch.randn(4, 5)
    layer.set_effective_delta(target)
    reconstructed = layer.effective_delta()
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    expected = (u[:, :3] * s[:3]) @ vh[:3]
    assert torch.allclose(reconstructed, expected, rtol=1e-4, atol=1e-5)
