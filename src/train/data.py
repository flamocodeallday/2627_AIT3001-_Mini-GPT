"""Training-side data access: read meta.json, check the tokenizer hash, sample random batches.

Reuses member 1's `MemmapDataset` (one shared reader for the whole repo, see week-3 plan).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.pack import MemmapDataset
from src.data.tokenizer_base import compute_file_sha256


def load_meta(data_dir: str | Path) -> dict[str, Any]:
    """Read `meta.json` of a released data version (never hard-code vocab_size / eos_id)."""
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Put the released data under data/processed/<version>/ (see README)."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def verify_tokenizer_hash(data_dir: str | Path, meta: dict[str, Any] | None = None) -> str:
    """Check that tokenizer.json matches meta['tokenizer_hash']; return the hash."""
    meta = meta if meta is not None else load_meta(data_dir)
    tok_path = Path(data_dir) / "tokenizer.json"
    expected = meta["tokenizer_hash"]
    if tok_path.exists():
        actual = compute_file_sha256(tok_path)
        if actual != expected:
            raise ValueError(f"tokenizer.json hash {actual[:12]}... != meta.json {expected[:12]}...")
    return expected


def tokenizer_meta_from(meta: dict[str, Any]) -> dict[str, Any]:
    """Subset of meta.json stored in every checkpoint (contract: checkpoint['tokenizer_meta'])."""
    return {
        "version": meta["version"],
        "tokenizer_type": meta["tokenizer_type"],
        "vocab_size": int(meta["vocab_size"]),
        "eos_id": int(meta["eos_id"]),
        "tokenizer_hash": meta["tokenizer_hash"],
    }


class TokenBatcher:
    """Draw random (x, y) windows of length block_size from one split.

    y is x shifted by one token. Windows may cross an `eos` boundary on purpose: the model learns
    that after eos a new, unrelated document starts (simplest choice, same as GPT-2 style training).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        block_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if split == "test":
            raise ValueError("test.bin is frozen: never train or tune on it (only member 3's final eval).")
        self.dataset = MemmapDataset(Path(data_dir) / f"{split}.bin", block_size)
        self.split = split
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Restart the random stream (used for val so every evaluation sees the same batches)."""
        self.rng = np.random.default_rng(self.seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x, y of shape (batch_size, block_size), dtype int64, on self.device."""
        starts = self.rng.integers(0, len(self.dataset), size=self.batch_size)
        pairs = [self.dataset[int(s)] for s in starts]
        x = torch.from_numpy(np.stack([p[0] for p in pairs]))
        y = torch.from_numpy(np.stack([p[1] for p in pairs]))
        if self.device.type == "cuda":
            # pinned host memory + non_blocking lets the copy overlap with GPU compute
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.rng.bit_generator.state = state["rng"]
