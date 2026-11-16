"""ByteTokenizer (v0), streaming binary `.bin` packer, and sample MemmapDataset."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.tokenizer_base import EOS_TOKEN, BaseTokenizer, compute_file_sha256


class ByteTokenizer(BaseTokenizer):
    """Raw UTF-8 byte-level tokenizer (IDs 0..255) plus <|endoftext|> at ID 256."""

    def __init__(self) -> None:
        self._vocab_size: int = 257
        self._eos_id: int = 256
        self.special_tokens: dict[str, int] = {EOS_TOKEN: self._eos_id}
        self._id_to_special: dict[int, str] = {self._eos_id: EOS_TOKEN}
        self._special_re = re.compile(f"({re.escape(EOS_TOKEN)})")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def train(self, texts: Iterable[str], vocab_size: int = 257) -> None:
        """ByteTokenizer has a fixed vocabulary of 257; validate contract argument."""
        if vocab_size != 257:
            raise ValueError(f"ByteTokenizer requires vocab_size=257, got {vocab_size}")

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode string to byte IDs; literal special text stays ordinary bytes unless allowed."""
        if not text:
            return []
        if not allow_special:
            return list(text.encode("utf-8"))

        ids: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            special_id = self.special_tokens.get(part)
            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(part.encode("utf-8"))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back to UTF-8 text, safely handling partial bytes and special tokens."""
        out_parts: list[str] = []
        byte_buf = bytearray()

        for tid in ids:
            tid_int = int(tid)
            if 0 <= tid_int < 256:
                byte_buf.append(tid_int)
            elif tid_int in self._id_to_special:
                if byte_buf:
                    out_parts.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                out_parts.append(self._id_to_special[tid_int])
            else:
                raise ValueError(f"Token ID {tid_int} out of range [0, {self._vocab_size - 1}]")

        if byte_buf:
            out_parts.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(out_parts)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "byte_level",
            "vocab_size": self._vocab_size,
            "eos_token": self.eos_token,
            "eos_id": self._eos_id,
            "special_tokens": dict(self.special_tokens),
        }

    def _load_json_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("tokenizer_type") != "byte_level":
            raise ValueError(f"Expected tokenizer_type='byte_level', got {payload.get('tokenizer_type')!r}")
        vocab_size = int(payload["vocab_size"])
        eos_id = int(payload["eos_id"])
        special_tokens = {str(k): int(v) for k, v in payload["special_tokens"].items()}
        if vocab_size != 257 or eos_id != 256:
            raise ValueError(f"Invalid ByteTokenizer config: vocab_size={vocab_size}, eos_id={eos_id}")
        self._vocab_size = vocab_size
        self._eos_id = eos_id
        self.eos_token = str(payload.get("eos_token", EOS_TOKEN))
        self.special_tokens = special_tokens
        self._id_to_special = {v: k for k, v in special_tokens.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(k) for k in special_tokens) + ")")


def slice_docs_by_utf8_bytes(
    docs: Iterable[str],
    max_bytes: int | None,
) -> Iterator[str]:
    """Yield whole documents until cumulative UTF-8 size reaches `max_bytes` (if set)."""
    if max_bytes is None:
        yield from docs
        return

    used_bytes = 0
    for doc in docs:
        doc_bytes = len(doc.encode("utf-8"))
        yield doc
        used_bytes += doc_bytes
        if used_bytes >= max_bytes:
            break


def _pack_single_split(
    docs: Iterable[str],
    tokenizer: BaseTokenizer,
    out_bin_path: Path,
    flush_tokens: int = 1_000_000,
) -> dict[str, int]:
    """Stream-encode documents, insert 1 `eos_id` between documents, and write uint16 `.bin`."""
    eos_id = int(tokenizer.eos_id)
    n_docs = 0
    n_tokens = 0
    n_chars = 0
    buffer: list[int] = []

    out_bin_path.parent.mkdir(parents=True, exist_ok=True)
    with out_bin_path.open("wb") as f:
        for doc in docs:
            if n_docs > 0:
                buffer.append(eos_id)
            encoded = tokenizer.encode(doc, allow_special=False)
            buffer.extend(encoded)

            n_docs += 1
            n_chars += len(doc)

            if len(buffer) >= flush_tokens:
                arr = np.asarray(buffer, dtype="<u2")
                f.write(arr.tobytes())
                n_tokens += len(buffer)
                buffer.clear()

        if buffer:
            arr = np.asarray(buffer, dtype="<u2")
            f.write(arr.tobytes())
            n_tokens += len(buffer)
            buffer.clear()

    return {"n_docs": n_docs, "n_tokens": n_tokens, "n_chars": n_chars}


def build_bins(
    splits: Mapping[str, Iterable[str]],
    tokenizer: BaseTokenizer,
    output_dir: str | Path,
    version: str,
    tokenizer_type: str,
    seed: int,
    target_mb: Mapping[str, float] | None = None,
    tokenizer_filename: str = "tokenizer.json",
    flush_tokens: int = 1_000_000,
) -> dict[str, Any]:
    """Pack train/val/test streams into uint16 `.bin` files and write `meta.json`.

    Args:
        splits: Mapping of split name ("train", "val", "test") to document iterables.
        tokenizer: Tokenizer implementing `encode(str, allow_special=False)`, `save`, `vocab_size`, `eos_id`.
        output_dir: Destination directory (e.g. `data/processed/v0_byte`).
        version: Dataset version tag (e.g. `"v0_byte"` or `"v1_bpe8192"`).
        tokenizer_type: Identifier written to `meta.json` (e.g. `"byte_level"`, `"custom_regex_bpe"`).
        seed: Random seed used when splitting the corpus.
        target_mb: Optional per-split UTF-8 byte cap in MiB (e.g. `{"train": 10.0, "val": 1.0, "test": 1.0}`).
        tokenizer_filename: Filename for saved tokenizer JSON inside `output_dir`.
        flush_tokens: Buffer size in tokens before flushing binary chunk to disk.

    Returns:
        The complete `meta.json` dictionary.
    """
    assert tokenizer.vocab_size <= 65535, f"vocab_size {tokenizer.vocab_size} exceeds uint16 max (65535)"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_path = tokenizer.save(out_dir / tokenizer_filename)
    tok_hash = compute_file_sha256(tok_path)

    split_stats: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise KeyError(f"Missing required split '{split_name}' in splits mapping")

        max_bytes: int | None = None
        if target_mb and split_name in target_mb and target_mb[split_name] is not None:
            max_bytes = int(float(target_mb[split_name]) * 1024 * 1024)

        doc_stream = slice_docs_by_utf8_bytes(splits[split_name], max_bytes=max_bytes)
        split_stats[split_name] = _pack_single_split(
            doc_stream,
            tokenizer=tokenizer,
            out_bin_path=out_dir / f"{split_name}.bin",
            flush_tokens=flush_tokens,
        )

    meta: dict[str, Any] = {
        "version": version,
        "tokenizer_type": tokenizer_type,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_id": int(tokenizer.eos_id),
        "seed": int(seed),
        "tokenizer_hash": tok_hash,
        "splits": split_stats,
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


class MemmapDataset:
    """Minimal zero-copy memmap reader yielding `(x, y)` int64 context windows from `.bin`."""

    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.data = np.memmap(bin_path, dtype="<u2", mode="r")
        self.block_size = int(block_size)
        if len(self.data) <= self.block_size:
            raise ValueError(f"Binary file {bin_path} has {len(self.data)} tokens, needs > {self.block_size}")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        return chunk[:-1], chunk[1:]
