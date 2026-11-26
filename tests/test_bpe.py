"""Unit tests for CustomBPE: training invariants, Vietnamese round-trips, Base64 save/load, and speed."""

from __future__ import annotations

import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data.bpe import CustomBPE
from src.data.clean import clean_and_dedup_stream
from src.data.pack import build_bins
from src.data.split import iter_jsonl_texts, split_by_document
from src.data.tokenizer_base import compute_file_sha256


def _naive_bpe_merges(texts: list[str], pattern_re, num_merges: int) -> list[tuple[int, int, int]]:
    """Simple reference BPE loop (no heap/index) used strictly to verify CustomBPE correctness."""
    freqs: Counter[bytes] = Counter()
    for t in texts:
        for m in pattern_re.finditer(t):
            freqs[m.group(0).encode("utf-8")] += 1

    seqs = {chunk: list(chunk) for chunk in freqs if len(chunk) >= 2}
    merges: list[tuple[int, int, int]] = []
    next_id = 257

    for _ in range(num_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for chunk, seq in seqs.items():
            w = freqs[chunk]
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += w
        if not pair_counts:
            break
        best_pair, best_cnt = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))
        if best_cnt < 2:
            break

        merged_id = next_id
        next_id += 1
        merges.append((best_pair[0], best_pair[1], merged_id))
        a, b = best_pair
        for chunk, seq in seqs.items():
            out: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(merged_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seqs[chunk] = out

    return merges


def test_pretokenization_lossless_and_training_matches_reference() -> None:
    samples = [
        "aaaaab aaaaab ac abab abab Việt Nam 2026!   \n\nDòng\t2 🚀",
        unicodedata.normalize("NFD", "Tiếng Việt tổ hợp"),
        "",
        "   \r\n  <|endoftext|>  1234567 ",
    ]
    bpe = CustomBPE()
    for text in samples:
        reconstructed = "".join(m.group(0) for m in bpe._chunk_re.finditer(text))
        assert reconstructed == text

    corpus = [
        "aaaaab aaaaab ac abab abab ab",
        "Việt Nam Việt Nam Hà Nội Hà Nội 2026 2026",
        "học sâu học sâu mô hình ngôn ngữ mô hình ngôn ngữ",
    ]
    bpe.train(corpus, vocab_size=275)
    ref_merges = _naive_bpe_merges(corpus, bpe._chunk_re, num_merges=275 - 257)
    assert bpe.merges == ref_merges
    assert bpe.vocab_size == 257 + len(ref_merges)


def test_bpe_roundtrip_edge_cases_and_special_tokens() -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    train_texts = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))[:80]

    bpe = CustomBPE()
    bpe.train(train_texts, vocab_size=512)
    assert bpe.vocab_size == 512
    assert bpe.eos_id == 256

    edge_strings = [
        "",
        "Xin chào Việt Nam! Tiếng Việt chuẩn NFC: ă â đ ê ô ơ ư Ắ Ệ Ử.",
        unicodedata.normalize("NFD", "Tiếng Việt tổ hợp NFD: Trường Đại học Công nghệ."),
        "Biểu tượng cảm xúc 🚀🔥🇻🇳👨‍👩‍👧‍👦 và chữ Nhật 日本語.",
        "Ký tự điều khiển \x00\x01\t\r\n và dấu câu — … “ ” « » °.",
        "Văn bản chứa chuỗi <|endoftext|> xuất hiện tự nhiên ở giữa câu.",
        "<|endoftext|>",
    ]

    for text in edge_strings:
        ids = bpe.encode(text)
        assert bpe.eos_id not in ids
        assert bpe.decode(ids) == text

    # Repeated encode exercises memoization cache and stays identical
    long_text = edge_strings[1] * 5
    assert bpe.encode(long_text) == bpe.encode(long_text)

    # allow_special=True emits eos_id for <|endoftext|>
    special_text = "Đoạn 1<|endoftext|>Đoạn 2"
    special_ids = bpe.encode(special_text, allow_special=True)
    assert bpe.eos_id in special_ids
    assert bpe.decode(special_ids) == special_text


def test_bpe_save_base64_and_dual_load(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    train_texts = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))[:100]

    bpe = CustomBPE()
    bpe.train(train_texts, vocab_size=600)
    assert bpe.vocab_size == 600

    tok_file = tmp_path / "tokenizer.json"
    saved = bpe.save(tok_file)
    assert saved == tok_file

    raw_json = json.loads(tok_file.read_text(encoding="utf-8"))
    assert raw_json["tokenizer_type"] == "custom_regex_bpe"
    assert raw_json["vocab_size"] == 600
    assert raw_json["eos_id"] == 256
    assert len(raw_json["merges"]) == 600 - 257
    assert "bytes_b64" in raw_json["merges"][0]

    # 1) Class-method load
    loaded_cls = CustomBPE.load(tok_file)
    assert isinstance(loaded_cls, CustomBPE)
    assert loaded_cls is not bpe
    assert loaded_cls.vocab_size == bpe.vocab_size
    assert loaded_cls.eos_id == bpe.eos_id
    assert loaded_cls.merges == bpe.merges

    # 2) Instance-method load (into an initially untrained or differently trained instance)
    existing = CustomBPE()
    existing.train(["abc abc"], vocab_size=258)
    ret = existing.load(tok_file)
    assert ret is existing
    assert existing.vocab_size == bpe.vocab_size
    assert existing.merges == bpe.merges

    probe_texts = [
        "Hà Nội và Thành phố Hồ Chí Minh là hai đô thị lớn 🚀.",
        unicodedata.normalize("NFD", "Văn học Việt Nam thế kỷ XX."),
        "Chuỗi <|endoftext|> kiểm tra \x00\t\n.",
    ]
    for text in probe_texts:
        ids_orig = bpe.encode(text)
        assert loaded_cls.encode(text) == ids_orig
        assert existing.encode(text) == ids_orig
        assert loaded_cls.decode(ids_orig) == text
        assert existing.decode(ids_orig) == text

    # Corrupt merge bytes must be detected on load
    raw_json["merges"][0]["bytes_b64"] = "AAAA"
    bad_file = tmp_path / "bad_tokenizer.json"
    bad_file.write_text(json.dumps(raw_json), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted merge entry"):
        CustomBPE.load(bad_file)


def test_bpe_validation_decode_errors_and_build_bins_integration(tmp_path: Path) -> None:
    bpe = CustomBPE()
    with pytest.raises(ValueError):
        bpe.train(["abc"], vocab_size=256)
    with pytest.raises(ValueError):
        bpe.train(["abc"], vocab_size=65536)

    with pytest.raises(ValueError):
        bpe.decode([9999])
    with pytest.raises(ValueError):
        bpe.decode([-1])

    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    splits = split_by_document(cleaned, ratios=(0.90, 0.05, 0.05), seed=42)

    t0 = time.perf_counter()
    bpe.train(splits["train"], vocab_size=1024)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0
    assert bpe.vocab_size == 1024

    sample_val = splits["val"][0]
    val_ids = bpe.encode(sample_val)
    assert len(val_ids) < len(sample_val.encode("utf-8"))
    assert bpe.decode(val_ids) == sample_val

    out_dir = tmp_path / "v1_bpe1024"
    meta = build_bins(
        splits=splits,
        tokenizer=bpe,
        output_dir=out_dir,
        version="v1_bpe1024",
        tokenizer_type="custom_regex_bpe",
        seed=42,
    )
    assert meta["vocab_size"] == 1024
    assert meta["eos_id"] == 256
    assert meta["tokenizer_hash"] == compute_file_sha256(out_dir / "tokenizer.json")

    test_arr = np.memmap(out_dir / "test.bin", dtype="<u2", mode="r")
    assert len(test_arr) == meta["splits"]["test"]["n_tokens"]
    assert int(test_arr.max()) < 1024
    assert bpe.decode(test_arr.tolist()) == "<|endoftext|>".join(splits["test"])
