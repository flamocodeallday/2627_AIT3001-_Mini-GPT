"""Unit tests for streaming splitting, stats computation, and JSONL(.zst) I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.clean import clean_and_dedup_stream
from src.data.split import (
    assign_split,
    corpus_stats,
    iter_jsonl_texts,
    split_by_document,
    write_splits_streaming,
)


def test_split_reproducibility_and_ratios() -> None:
    docs = [f"Tài liệu số {i} với nội dung tiếng Việt đầy đủ." for i in range(1000)]
    s1 = split_by_document(docs, ratios=(0.90, 0.05, 0.05), seed=42)
    s2 = split_by_document(docs, ratios={"train": 0.90, "val": 0.05, "test": 0.05}, seed=42)

    assert s1 == s2
    assert sum(len(v) for v in s1.values()) == 1000
    assert 870 <= len(s1["train"]) <= 930
    assert 25 <= len(s1["val"]) <= 75
    assert 25 <= len(s1["test"]) <= 75

    # Ensure disjointness across splits
    train_set, val_set, test_set = set(s1["train"]), set(s1["val"]), set(s1["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_invalid_ratios_raise() -> None:
    with pytest.raises(ValueError):
        list(assign_split(["a"], ratios=(0.5, 0.5, 0.1)))
    with pytest.raises(ValueError):
        list(assign_split(["a"], ratios=(1.0, 0.0, 0.0)))


def test_corpus_stats_values() -> None:
    splits = {
        "train": ["a" * 10, "b" * 20, "c" * 30],
        "val": ["d" * 40],
        "test": [],
    }
    stats = corpus_stats(splits)
    assert stats["train"]["n_docs"] == 3
    assert stats["train"]["total_chars"] == 60
    assert stats["train"]["min_chars"] == 10
    assert stats["train"]["median_chars"] == 20.0
    assert stats["train"]["max_chars"] == 30
    assert stats["test"]["n_docs"] == 0
    assert stats["total"]["n_docs"] == 4
    assert stats["total"]["total_chars"] == 100


@pytest.mark.parametrize("compress", [False, True])
def test_write_splits_streaming_and_iter_roundtrip(tmp_path: Path, compress: bool) -> None:
    docs = [f"Bài viết thử nghiệm số {i} có tiếng Việt chuẩn." for i in range(60)]
    res = write_splits_streaming(
        docs,
        output_dir=tmp_path,
        ratios=(0.8, 0.1, 0.1),
        seed=123,
        compress=compress,
    )
    stats_file = Path(res["stats_file"])
    assert stats_file.exists()
    loaded_stats = json.loads(stats_file.read_text(encoding="utf-8"))
    assert loaded_stats["stats"]["total"]["n_docs"] == 60

    recovered = []
    for name in ("train", "val", "test"):
        recovered.extend(iter_jsonl_texts(res["files"][name]))
    assert sorted(recovered) == sorted(docs)


def test_fixtures_end_to_end(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    assert fixture_path.exists()
    assert fixture_path.stat().st_size >= 400_000

    cleaned = clean_and_dedup_stream(iter_jsonl_texts(fixture_path))
    res = write_splits_streaming(
        cleaned,
        output_dir=tmp_path,
        ratios=(0.90, 0.05, 0.05),
        seed=42,
        compress=False,
    )
    assert res["stats"]["train"]["n_docs"] > 0
    assert res["stats"]["val"]["n_docs"] > 0
    assert res["stats"]["test"]["n_docs"] > 0
