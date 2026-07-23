from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scale * update

    def effective_delta(self) -> torch.Tensor:
        return self.scale * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def set_effective_delta(self, delta: torch.Tensor) -> None:
        delta = delta.to(self.lora_A.device, torch.float32)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(self.rank, s.numel())
        u, s, vh = u[:, :r], s[:r], vh[:r]
        root = torch.sqrt(s.clamp_min(0.0) / max(self.scale, 1e-12))
        b = u * root[None, :]
        a = root[:, None] * vh
        self.lora_A.zero_()
        self.lora_B.zero_()
        self.lora_A[:r].copy_(a.to(self.lora_A.dtype))
        self.lora_B[:, :r].copy_(b.to(self.lora_B.dtype))


@dataclass
class LoRAConfig:
    rank: int
    alpha: float = 16.0
    encoder_targets: Tuple[str, ...] = ("query", "value")
    decoder_token: str = "decode_head.linear_c"


def inject_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    replaced: List[str] = []

    def visit(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            eligible_encoder = isinstance(child, nn.Linear) and name in config.encoder_targets
            eligible_decoder = isinstance(child, nn.Linear) and config.decoder_token in full
            if eligible_encoder or eligible_decoder:
                setattr(module, name, LoRALinear(child, config.rank, config.alpha))
                replaced.append(full)
            else:
                visit(child, full)

    visit(model)
    return replaced


def lora_modules(model: nn.Module) -> Dict[str, LoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def trainable_non_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    lora_param_names = {
        f"{name}.lora_A" for name in lora_modules(model)
    } | {f"{name}.lora_B" for name in lora_modules(model)}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name not in lora_param_names:
            result[name] = parameter.detach().cpu().clone()
    return result


@torch.no_grad()
def aggregate_models_function_space(
    client_models: Sequence[nn.Module],
    weights: Sequence[float],
    target_models: Sequence[nn.Module],
) -> None:
    """FedAvg for shared trainables and exact effective-update averaging for LoRA."""
    if not client_models:
        return
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum().clamp_min(1e-12)

    client_named = [dict(m.named_parameters()) for m in client_models]
    target_named = [dict(m.named_parameters()) for m in target_models]
    lora_names = set()
    for name in lora_modules(client_models[0]):
        lora_names.add(f"{name}.lora_A")
        lora_names.add(f"{name}.lora_B")

    for name, p0 in client_named[0].items():
        if name in lora_names or not p0.requires_grad:
            continue
        avg = torch.zeros_like(p0, device="cpu", dtype=torch.float64)
        for wi, params in zip(w, client_named):
            avg += wi * params[name].detach().cpu().double()
        for params in target_named:
            params[name].copy_(avg.to(params[name].device, params[name].dtype))

    client_lora = [lora_modules(m) for m in client_models]
    target_lora = [lora_modules(m) for m in target_models]
    for name in client_lora[0]:
        delta = None
        for wi, modules in zip(w, client_lora):
            current = modules[name].effective_delta().detach().cpu().double()
            delta = wi * current if delta is None else delta + wi * current
        assert delta is not None
        for modules in target_lora:
            modules[name].set_effective_delta(delta)


def classifier_weight_parameter(model: nn.Module) -> nn.Parameter:
    for name, module in model.named_modules():
        if name.endswith("decode_head.classifier") and isinstance(module, nn.Conv2d):
            return module.weight
        if name.endswith("decode_head.classifier") and isinstance(module, nn.Linear):
            return module.weight
    raise KeyError("decode_head.classifier was not found")
