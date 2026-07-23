from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation

from .lora import LoRAConfig, inject_lora


@dataclass
class ModelConfig:
    pretrained_path: str
    num_classes: int = 19
    rank: int = 8
    lora_alpha: float = 16.0
    group_norm_groups: int = 32


def _replace_decoder_norm(model: nn.Module, groups: int) -> None:
    head = model.decode_head
    if hasattr(head, "batch_norm"):
        old = head.batch_norm
        channels = old.num_features
        valid_groups = min(groups, channels)
        while channels % valid_groups != 0 and valid_groups > 1:
            valid_groups -= 1
        head.batch_norm = nn.GroupNorm(valid_groups, channels)


def build_model(config: ModelConfig) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.pretrained_path,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,
    )
    _replace_decoder_norm(model, config.group_norm_groups)

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = inject_lora(
        model,
        LoRAConfig(rank=config.rank, alpha=config.lora_alpha),
    )
    if not replaced:
        raise RuntimeError("No LoRA target modules were found in SegFormer")

    for parameter in model.decode_head.parameters():
        parameter.requires_grad = True
    # Keep the frozen base inside LoRA wrappers frozen after unfreezing the decoder.
    from .lora import LoRALinear
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    return model


def parameter_report(model: nn.Module) -> Dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def logits_from_model(model: nn.Module, images: torch.Tensor, output_size=None) -> torch.Tensor:
    output = model(pixel_values=images)
    logits = output.logits
    if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
        logits = torch.nn.functional.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
    return logits
