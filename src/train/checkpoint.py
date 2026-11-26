"""Checkpoint save/load following interface contract v0.

Contract keys (read by member 3's GPTAdapter): state_dict, config, tokenizer_meta, step, val_loss.
Extra keys for resuming: optimizer, scaler, batcher, torch_rng, train_args, best_val_loss.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.model.config import GPTConfig
from src.model.gpt import GPT


def unwrap(model: nn.Module) -> nn.Module:
    """Return the original module if it was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    tokenizer_meta: dict[str, Any],
    step: int,
    val_loss: float | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    batcher_state: dict[str, Any] | None = None,
    train_args: dict[str, Any] | None = None,
    best_val_loss: float | None = None,
) -> Path:
    """Write atomically (temp file + rename) so a Colab disconnect never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = unwrap(model)
    payload: dict[str, Any] = {
        "state_dict": raw.state_dict(),
        "config": raw.config.to_dict(),
        "tokenizer_meta": dict(tokenizer_meta),
        "step": int(step),
        "val_loss": None if val_loss is None else float(val_loss),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "batcher": batcher_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_args": train_args or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> GPT:
    model = GPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model


def gpt_model_builder(cfg: dict[str, Any] | GPTConfig) -> GPT:
    """Pass this as `model_builder` to GPTAdapter.from_checkpoint (member 3)."""
    return GPT.from_config_dict(cfg)
