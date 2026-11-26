"""AdamW construction with weight-decay groups."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def split_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into (decay, no_decay). Exercise for Todd (bài 10).

    Convention for this project:
      * decay:    every parameter with 2 or more dimensions (Linear weights, embeddings;
                  the tied tok_emb / lm_head matrix appears once).
      * no_decay: every 1-D parameter (biases, LayerNorm weights/biases).
    Every trainable parameter must be in exactly one list. Write in a comment WHY 1-D params
    are excluded (you will be asked in the oral exam).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    return decay, no_decay
    


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """AdamW with two parameter groups; uses the fused CUDA kernel when available."""
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)
