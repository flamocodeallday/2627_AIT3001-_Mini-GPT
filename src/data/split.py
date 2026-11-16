"""Document-level streaming splitting, JSONL/Zstd I/O, and corpus statistics."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def _normalize_ratios(
    ratios: Mapping[str, float] | Sequence[float],
) -> tuple[float, float, float]:
    """Validate and convert split ratios into (train, val, test) floats summing to 1.0."""
    if isinstance(ratios, Mapping):
        vals = (float(ratios["train"]), float(ratios["val"]), float(ratios["test"]))
    elif len(ratios) == 3:
        vals = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    else:
        raise ValueError(f"Expected 3 split ratios (train, val, test), got {ratios!r}")

    if any(r <= 0.0 for r in vals):
        raise ValueError(f"All split ratios must be positive, got {vals}")
    total = sum(vals)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total} ({vals})")
    return vals


def assign_split(
    docs: Iterable[str],
    ratios: Mapping[str, float] | Sequence[float] = (0.98, 0.01, 0.01),
    seed: int = 42,
) -> Iterator[tuple[str, str]]:
    """Assign each document to 'train', 'val', or 'test' in O(1) memory using fixed seed."""
    r_train, r_val, _ = _normalize_ratios(ratios)
    val_cutoff = r_train + r_val
    rng = random.Random(seed)

    for doc in docs:
        u = rng.random()
        if u < r_train:
            yield "train", doc
        elif u < val_cutoff:
            yield "val", doc
        else:
            yield "test", doc


def split_by_document(
    docs: Iterable[str],
    ratios: Mapping[str, float] | Sequence[float] = (0.98, 0.01, 0.01),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Materialize split buckets in memory (intended for small fixtures/tests only)."""
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split_name, doc in assign_split(docs, ratios=ratios, seed=seed):
        buckets[split_name].append(doc)
    return buckets


class SplitStatsAccumulator:
    """Lightweight accumulator storing only document lengths (not document strings)."""

    def __init__(self) -> None:
        self.n_docs: int = 0
        self.total_chars: int = 0
        self._lengths: list[int] = []

    def update(self, doc: str) -> None:
        length = len(doc)
        self.n_docs += 1
        self.total_chars += length
        self._lengths.append(length)

    def finalize(self) -> dict[str, int | float]:
        if self.n_docs == 0:
            return {
                "n_docs": 0,
                "total_chars": 0,
                "min_chars": 0,
                "median_chars": 0.0,
                "p95_chars": 0.0,
                "max_chars": 0,
                "mean_chars": 0.0,
            }

        sorted_lens = sorted(self._lengths)
        n = len(sorted_lens)
        mid = n // 2
        median = float(sorted_lens[mid]) if n % 2 == 1 else (sorted_lens[mid - 1] + sorted_lens[mid]) / 2.0

        # Nearest-rank / linear interpolation for 95th percentile.
        p95_pos = 0.95 * (n - 1)
        lo = int(math.floor(p95_pos))
        hi = int(math.ceil(p95_pos))
        frac = p95_pos - lo
        p95 = sorted_lens[lo] * (1.0 - frac) + sorted_lens[hi] * frac

        return {
            "n_docs": self.n_docs,
            "total_chars": self.total_chars,
            "min_chars": sorted_lens[0],
            "median_chars": round(median, 2),
            "p95_chars": round(p95, 2),
            "max_chars": sorted_lens[-1],
            "mean_chars": round(self.total_chars / self.n_docs, 2),
        }


def corpus_stats(splits: Mapping[str, Iterable[str]]) -> dict[str, dict[str, int | float]]:
    """Compute per-split and total corpus length statistics in streaming fashion."""
    stats: dict[str, dict[str, int | float]] = {}
    total_acc = SplitStatsAccumulator()

    for name, docs in splits.items():
        acc = SplitStatsAccumulator()
        for doc in docs:
            acc.update(doc)
            total_acc.update(doc)
        stats[name] = acc.finalize()

    stats["total"] = total_acc.finalize()
    return stats


def _open_text_writer(path: Path, compress: bool) -> Any:
    if compress:
        import zstandard as zstd

        raw_f = path.open("wb")
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.stream_writer(raw_f, closefd=True), True
    return path.open("w", encoding="utf-8"), False


def iter_jsonl_texts(path: str | Path, text_field: str = "text") -> Iterator[str]:
    """Stream document strings from a `.jsonl` or `.jsonl.zst` file."""
    file_path = Path(path)
    if file_path.suffix == ".zst":
        import io
        import zstandard as zstd

        with file_path.open("rb") as raw_f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw_f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if line:
                        yield json.loads(line)[text_field]
    else:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)[text_field]


def write_splits_streaming(
    docs: Iterable[str],
    output_dir: str | Path,
    ratios: Mapping[str, float] | Sequence[float] = (0.98, 0.01, 0.01),
    seed: int = 42,
    compress: bool = False,
) -> dict[str, Any]:
    """Stream documents directly to split JSONL(.zst) files and compute statistics without OOM."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jsonl.zst" if compress else ".jsonl"

    writers: dict[str, tuple[Any, bool]] = {}
    accs: dict[str, SplitStatsAccumulator] = {}
    total_acc = SplitStatsAccumulator()
    file_paths: dict[str, str] = {}

    try:
        for name in SPLIT_NAMES:
            p = out_dir / f"{name}{ext}"
            file_paths[name] = str(p)
            writers[name] = _open_text_writer(p, compress=compress)
            accs[name] = SplitStatsAccumulator()

        for split_name, doc in assign_split(docs, ratios=ratios, seed=seed):
            accs[split_name].update(doc)
            total_acc.update(doc)
            line = json.dumps({"text": doc}, ensure_ascii=False) + "\n"
            writer, is_binary = writers[split_name]
            if is_binary:
                writer.write(line.encode("utf-8"))
            else:
                writer.write(line)
    finally:
        for writer, _ in writers.values():
            writer.close()

    stats = {name: accs[name].finalize() for name in SPLIT_NAMES}
    stats["total"] = total_acc.finalize()

    stats_path = out_dir / "split_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "ratios": _normalize_ratios(ratios), "files": file_paths, "stats": stats},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {"files": file_paths, "stats_file": str(stats_path), "stats": stats}
