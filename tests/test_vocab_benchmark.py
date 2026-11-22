"""Unit tests for Gói 4 vocab-size benchmarking and HF tokenizers comparison."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.bpe import CustomBPE
from src.data.clean import clean_and_dedup_stream
from src.data.pack import build_bins
from src.data.split import iter_jsonl_texts, split_by_document, write_splits_streaming
from src.data.tokenizer_base import compute_file_sha256
from src.data.vocab_eval import (
    analyze_token_usage,
    compute_corpus_text_totals,
    embedding_param_cost,
    inspect_tokenizer_vocab,
    run_vocab_benchmark,
)


def test_embedding_param_cost_and_corpus_totals() -> None:
    costs = embedding_param_cost(vocab_size=8192, d_models=(256, 384, 512))
    assert [c["d_model"] for c in costs] == [256, 384, 512]
    assert costs[0]["embedding_params"] == 8192 * 256
    assert costs[1]["embedding_params"] == 8192 * 384
    assert costs[2]["embedding_params"] == 8192 * 512
    assert costs[0]["pct_of_10M"] == round((8192 * 256 / 10_000_000) * 100.0, 2)
    assert costs[2]["pct_of_30M"] == round((8192 * 512 / 30_000_000) * 100.0, 2)

    totals = compute_corpus_text_totals(["Hà Nội đẹp", "Việt Nam"])
    assert totals["n_docs"] == 2
    assert totals["n_chars"] == len("Hà Nội đẹp") + len("Việt Nam")
    assert totals["n_words"] == 5
    assert totals["n_bytes"] == len("Hà Nội đẹp".encode("utf-8")) + len("Việt Nam".encode("utf-8"))


def test_vocab_benchmark_end_to_end_on_fixtures(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    interim_dir = tmp_path / "interim"
    write_splits_streaming(cleaned, output_dir=interim_dir, ratios=(0.90, 0.05, 0.05), seed=42)

    splits = split_by_document(cleaned, ratios=(0.90, 0.05, 0.05), seed=42)
    out_dir = tmp_path / "benchmark_out"

    report = run_vocab_benchmark(
        train_docs_iter=splits["train"],
        val_docs_iter=splits["val"],
        out_dir=out_dir,
        train_mb=0.5,
        vocab_sizes=[300, 350],
        d_models=[256, 384, 512],
        include_hf=True,
    )

    assert report["vocab_sizes"] == [300, 350]
    assert len(report["custom_bpe_results"]) == 2
    assert len(report["hf_bpe_results"]) == 2

    # Larger vocab must achieve equal or better compression (higher bytes_per_token, lower n_tokens)
    r300, r350 = report["custom_bpe_results"]
    assert r300["actual_vocab_size"] == 300
    assert r350["actual_vocab_size"] == 350
    assert r350["n_tokens"] < r300["n_tokens"]
    assert r350["bytes_per_token"] > r300["bytes_per_token"]
    assert r350["fertility_tokens_per_word"] < r300["fertility_tokens_per_word"]

    for row in report["custom_bpe_results"]:
        v = row["actual_vocab_size"]
        tok_path = out_dir / f"custom_bpe{v}.json"
        assert tok_path.is_file()
        assert row["tokenizer_hash"] == compute_file_sha256(tok_path)
        reloaded = CustomBPE.load(tok_path)
        assert reloaded.vocab_size == v
        assert len(row["embedding_cost"]) == 3

    for hf_row in report["hf_bpe_results"]:
        assert hf_row["implementation"] == "hf_tokenizers_byte_bpe"
        assert hf_row["n_tokens"] > 0
        assert hf_row["bytes_per_token"] > 0
        assert hf_row["fertility_tokens_per_word"] > 0

    json_report = json.loads((out_dir / "vocab_benchmark.json").read_text(encoding="utf-8"))
    assert len(json_report["comparisons"]) == 4
    md_text = (out_dir / "vocab_benchmark.md").read_text(encoding="utf-8")
    assert "custom_regex_bpe" in md_text
    assert "hf_tokenizers_byte_bpe" in md_text
    assert "Embedding Params" in md_text


def test_inspect_tokenizer_vocab_and_analyze_token_usage(tmp_path: Path) -> None:
    fixture_path = Path("fixtures/sample_vi_wiki.jsonl")
    cleaned = list(clean_and_dedup_stream(iter_jsonl_texts(fixture_path)))
    splits = split_by_document(cleaned, ratios=(0.90, 0.05, 0.05), seed=42)

    bpe = CustomBPE()
    bpe.train(splits["train"], vocab_size=512)

    vocab_report = inspect_tokenizer_vocab(bpe, sample_limit=20)
    assert vocab_report["vocab_size"] == 512
    assert vocab_report["n_learned_tokens"] == 512 - 257
    assert vocab_report["avg_learned_byte_length"] >= 2.0
    assert 0.0 < vocab_report["letter_token_ratio"] <= 1.0
    assert sum(vocab_report["category_counts"].values()) == vocab_report["n_learned_tokens"]
    assert len(vocab_report["first_learned_tokens"]) == 20
    assert len(vocab_report["longest_letter_tokens"]) > 0

    out_dir = tmp_path / "v1_bpe512"
    meta = build_bins(
        splits=splits,
        tokenizer=bpe,
        output_dir=out_dir,
        version="v1_bpe512",
        tokenizer_type="custom_regex_bpe",
        seed=42,
        workers=1,
    )

    usage_train = analyze_token_usage(out_dir / "train.bin", vocab_size=512, tokenizer=bpe, top_k=10)
    assert usage_train["total_tokens"] == meta["splits"]["train"]["n_tokens"]
    assert usage_train["vocab_size"] == 512
    assert 0 <= usage_train["dead_token_count"] < 512
    assert 0 <= usage_train["learned_dead_token_count"] < (512 - 257)
    assert len(usage_train["top_tokens"]) == 10
    assert sum(t["count"] for t in usage_train["top_tokens"]) <= usage_train["total_tokens"]
    assert all("text" in t for t in usage_train["top_tokens"])

    # Empty bin returns graceful zero stats
    empty_bin = tmp_path / "empty.bin"
    empty_bin.write_bytes(b"")
    usage_empty = analyze_token_usage(empty_bin, vocab_size=512)
    assert usage_empty["total_tokens"] == 0
    assert usage_empty["dead_token_count"] == 512
    assert usage_empty["dead_token_ratio"] == 1.0
