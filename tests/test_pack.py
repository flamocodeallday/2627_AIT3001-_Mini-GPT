"""Unit tests for ByteTokenizer (v0), dual load descriptor, binary packing, and MemmapDataset."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from src.data.bpe import CustomBPE
from src.data.clean import clean_and_dedup_stream
from src.data.pack import ByteTokenizer, MemmapDataset, build_bins
from src.data.split import iter_jsonl_texts, split_by_document, write_splits_streaming
from src.data.tokenizer_base import compute_file_sha256


def test_byte_tokenizer_roundtrip_and_special_handling() -> None:
    tok = ByteTokenizer()
    assert tok.vocab_size == 257
    assert tok.eos_id == 256
    tok.train(["mẫu"], vocab_size=257)

    with pytest.raises(ValueError):
        tok.train(["mẫu"], vocab_size=512)

    samples = [
        "",
        "Xin chào Việt Nam! Tiếng Việt chuẩn NFC.",
        unicodedata.normalize("NFD", "Tiếng Việt tổ hợp NFD có dấu."),
        "Biểu tượng 🚀🔥 và ký tự đặc biệt \n\t — …",
        "Văn bản chứa chuỗi <|endoftext|> bình thường.",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert tok.eos_id not in ids
        assert tok.decode(ids) == text

    # allow_special=True emits eos_id directly
    special_ids = tok.encode("A<|endoftext|>B", allow_special=True)
    assert special_ids == [65, 256, 66]
    assert tok.decode(special_ids) == "A<|endoftext|>B"


def test_byte_tokenizer_decode_resilience_and_errors() -> None:
    tok = ByteTokenizer()
    # Truncated UTF-8 sequence should not raise
    decoded = tok.decode([0xE1, 0xBB])
    assert "\ufffd" in decoded

    with pytest.raises(ValueError):
        tok.decode([257])
    with pytest.raises(ValueError):
        tok.decode([-1])


def test_byte_tokenizer_save_and_dual_load(tmp_path: Path) -> None:
    tok = ByteTokenizer()
    tok_file = tmp_path / "tokenizer.json"
    saved_path = tok.save(tok_file)
    assert saved_path == tok_file
    assert tok_file.is_file()

    # 1) Class-method call: ByteTokenizer.load(path) -> new instance
    loaded_cls = ByteTokenizer.load(tok_file)
    assert isinstance(loaded_cls, ByteTokenizer)
    assert loaded_cls is not tok
    assert loaded_cls.vocab_size == 257
    assert loaded_cls.eos_id == 256
    assert loaded_cls.decode(loaded_cls.encode("Hà Nội")) == "Hà Nội"

    # 2) Instance-method call: existing_tok.load(path) -> populates in-place
    existing = ByteTokenizer()
    existing._vocab_size = 999  # mutate to verify reload resets state
    ret = existing.load(tok_file)
    assert ret is existing
    assert existing.vocab_size == 257
    assert existing.eos_id == 256


def test_build_bins_and_memmap_dataset_on_fixtures(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    splits = split_by_document(cleaned, ratios=(0.90, 0.05, 0.05), seed=42)

    out_dir = tmp_path / "v0_byte"
    tok = ByteTokenizer()
    meta = build_bins(
        splits=splits,
        tokenizer=tok,
        output_dir=out_dir,
        version="v0_byte",
        tokenizer_type="byte_level",
        seed=42,
        target_mb={"train": 0.15, "val": 0.05, "test": 0.05},
        flush_tokens=1024,
    )

    # Check meta.json schema & SHA-256 hash
    meta_on_disk = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta_on_disk == meta
    assert meta["version"] == "v0_byte"
    assert meta["tokenizer_type"] == "byte_level"
    assert meta["vocab_size"] == 257
    assert meta["eos_id"] == 256
    assert meta["seed"] == 42
    assert meta["tokenizer_hash"] == compute_file_sha256(out_dir / "tokenizer.json")

    for split_name in ("train", "val", "test"):
        bin_path = out_dir / f"{split_name}.bin"
        assert bin_path.exists()
        arr = np.memmap(bin_path, dtype="<u2", mode="r")
        s_meta = meta["splits"][split_name]

        assert len(arr) == s_meta["n_tokens"]
        assert s_meta["n_docs"] >= 1
        assert int(arr.max()) < meta["vocab_size"]

        # Strictly 1 eos_id between adjacent documents, none inside literal doc text
        eos_positions = np.where(arr == tok.eos_id)[0]
        assert len(eos_positions) == s_meta["n_docs"] - 1
        if len(eos_positions) > 1:
            assert np.all(np.diff(eos_positions) > 1)

        # Round-trip decode over entire split matches joined docs with <|endoftext|> separator
        decoded = tok.decode(arr.tolist())
        expected_docs = splits[split_name][: s_meta["n_docs"]]
        assert s_meta["n_chars"] == sum(len(d) for d in expected_docs)
        assert decoded == "<|endoftext|>".join(expected_docs)

    # Verify MemmapDataset window slicing
    ds = MemmapDataset(out_dir / "train.bin", block_size=64)
    assert len(ds) == meta["splits"]["train"]["n_tokens"] - 64
    x, y = ds[0]
    assert x.shape == (64,) and y.shape == (64,)
    assert x.dtype == np.int64 and y.dtype == np.int64
    assert np.array_equal(x[1:], y[:-1])


def test_build_bins_parallel_matches_single_worker_bit_for_bit(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    splits = split_by_document(cleaned, ratios=(0.90, 0.05, 0.05), seed=42)

    bpe = CustomBPE()
    bpe.train(splits["train"], vocab_size=512)
    prebuilt_tok = tmp_path / "custom_bpe512.json"
    bpe.save(prebuilt_tok)
    expected_hash = compute_file_sha256(prebuilt_tok)

    out_single = tmp_path / "out_w1"
    out_parallel = tmp_path / "out_w2"

    meta_w1 = build_bins(
        splits=splits,
        tokenizer=bpe,
        output_dir=out_single,
        version="v1_bpe512",
        tokenizer_type="custom_regex_bpe",
        seed=42,
        tokenizer_source_path=prebuilt_tok,
        workers=1,
        batch_size=25,
    )
    meta_w2 = build_bins(
        splits=splits,
        tokenizer=bpe,
        output_dir=out_parallel,
        version="v1_bpe512",
        tokenizer_type="custom_regex_bpe",
        seed=42,
        tokenizer_source_path=prebuilt_tok,
        workers=2,
        batch_size=25,
    )

    assert meta_w1 == meta_w2
    assert meta_w2["tokenizer_hash"] == expected_hash
    assert (out_parallel / "tokenizer.json").read_bytes() == prebuilt_tok.read_bytes()

    for split_name in ("train", "val", "test"):
        b1 = (out_single / f"{split_name}.bin").read_bytes()
        b2 = (out_parallel / f"{split_name}.bin").read_bytes()
        assert b1 == b2
        arr = np.memmap(out_parallel / f"{split_name}.bin", dtype="<u2", mode="r")
        assert int(np.sum(arr == bpe.eos_id)) == len(splits[split_name]) - 1


def test_pack_data_cli_bpe_parallel_and_sanity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pack_data import main as pack_main

    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    interim_dir = tmp_path / "interim"
    write_splits_streaming(cleaned, output_dir=interim_dir, ratios=(0.90, 0.05, 0.05), seed=42)

    bpe = CustomBPE()
    bpe.train(iter_jsonl_texts(interim_dir / "train.jsonl"), vocab_size=400)
    tok_path = tmp_path / "custom_bpe400.json"
    bpe.save(tok_path)

    out_dir = tmp_path / "v1_bpe400"
    monkeypatch.setattr(
        "sys.argv",
        [
            "scripts.pack_data",
            "--config",
            "configs/data.yaml",
            "--interim-dir",
            str(interim_dir),
            "--output-dir",
            str(out_dir),
            "--tokenizer-path",
            str(tok_path),
            "--workers",
            "2",
            "--batch-size",
            "25",
            "--sanity-check",
        ],
    )
    pack_main()

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "v1_bpe400"
    assert meta["tokenizer_type"] == "custom_regex_bpe"
    assert meta["vocab_size"] == 400
    assert meta["tokenizer_hash"] == compute_file_sha256(tok_path)
    assert (out_dir / "tokenizer.json").read_bytes() == tok_path.read_bytes()
