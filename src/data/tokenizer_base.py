"""Shared base class and descriptor utilities for custom tokenizers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

EOS_TOKEN = "<|endoftext|>"
T = TypeVar("T", bound="BaseTokenizer")


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming 64 KB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


class _DualLoadDescriptor:
    """Descriptor allowing `Tokenizer.load(path)` and `tok.load(path)` on the same method."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.__doc__ = func.__doc__

    def __get__(self, instance: T | None, owner: type[T]) -> Callable[[str | Path], T]:
        def _loader(path: str | Path) -> T:
            return self._func(owner, instance, path)

        _loader.__name__ = self._func.__name__
        _loader.__doc__ = self._func.__doc__
        return _loader


class BaseTokenizer:
    """Common interface contract for ByteTokenizer (v0) and CustomBPE (v1)."""

    eos_token: str = EOS_TOKEN

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        raise NotImplementedError

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        raise NotImplementedError

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def to_json_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load_json_dict(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        """Serialize tokenizer into a single JSON file and return its Path."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_dict()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

    @_DualLoadDescriptor
    def load(cls: type[T], instance: T | None, path: str | Path) -> T:
        """Load tokenizer state from a single JSON file.

        Works both as `Tokenizer.load(path)` (returns a new instance) and
        `tok.load(path)` (populates `tok` in-place and returns `tok`).
        """
        in_path = Path(path)
        with in_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        target = instance if instance is not None else cls()
        target._load_json_dict(payload)
        return target
