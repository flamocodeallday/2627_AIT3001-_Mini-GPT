"""Reusable vocabulary-size benchmarking and HF `tokenizers` comparison helpers."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.bpe import CustomBPE
from src.data.pack import slice_docs_by_utf8_bytes
from src.data.tokenizer_base import EOS_TOKEN, BaseTokenizer, compute_file_sha256

DEFAULT_VOCAB_SIZES: tuple[int, ...] = (4096, 8192, 16384)
DEFAULT_D_MODELS: tuple[int, ...] = (256, 384, 512)


def compute_corpus_text_totals(docs: Iterable[str]) -> dict[str, int]:
    """Compute document, char, UTF-8 byte, and whitespace-word counts on normalized docs."""
    n_docs = 0
    n_chars = 0
    n_bytes = 0
    n_words = 0
    for doc in docs:
        n_docs += 1
        n_chars += len(doc)
        n_bytes += len(doc.encode("utf-8"))
        n_words += len(doc.split())
    return {
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "n_words": n_words,
    }


def embedding_param_cost(
    vocab_size: int,
    d_models: Sequence[int] = DEFAULT_D_MODELS,
    budgets: Sequence[int] = (10_000_000, 30_000_000),
) -> list[dict[str, Any]]:
    """Compute tied-embedding parameter counts (`vocab_size * d_model`) and budget shares."""
    rows: list[dict[str, Any]] = []
    for d_model in d_models:
        params = int(vocab_size) * int(d_model)
        row: dict[str, Any] = {
            "d_model": int(d_model),
            "embedding_params": params,
        }
        for budget in budgets:
            label = f"pct_of_{budget // 1_000_000}M"
            row[label] = round((params / budget) * 100.0, 2)
        rows.append(row)
    return rows


def evaluate_token_counts_on_val(
    encode_batch_fn: Callable[[Sequence[str]], Sequence[int]],
    val_docs: Sequence[str],
    val_totals: dict[str, int],
) -> dict[str, Any]:
    """Time validation encoding and compute compression ratio and fertility metrics."""
    t0 = time.perf_counter()
    token_lengths = encode_batch_fn(val_docs)
    encode_sec = time.perf_counter() - t0

    n_tokens = int(sum(token_lengths))
    if n_tokens <= 0:
        raise ValueError("Encoded validation set produced 0 tokens")

    n_bytes = val_totals["n_bytes"]
    n_chars = val_totals["n_chars"]
    n_words = val_totals["n_words"]

    return {
        "n_tokens": n_tokens,
        "val_encode_time_sec": round(encode_sec, 4),
        "bytes_per_token": round(n_bytes / n_tokens, 4),
        "chars_per_token": round(n_chars / n_tokens, 4),
        "fertility_tokens_per_word": round(n_tokens / n_words, 4) if n_words > 0 else 0.0,
    }


def _train_and_eval_custom_bpe(
    train_sample: Sequence[str],
    val_docs: Sequence[str],
    val_totals: dict[str, int],
    vocab_size: int,
    save_path: Path | None,
    d_models: Sequence[int],
) -> dict[str, Any]:
    bpe = CustomBPE()
    t0 = time.perf_counter()
    bpe.train(train_sample, vocab_size=vocab_size)
    train_sec = time.perf_counter() - t0

    tok_hash: str | None = None
    saved_str: str | None = None
    if save_path is not None:
        saved = bpe.save(save_path)
        tok_hash = compute_file_sha256(saved)
        saved_str = str(saved)

    val_metrics = evaluate_token_counts_on_val(
        lambda docs: [len(bpe.encode(d, allow_special=False)) for d in docs],
        val_docs=val_docs,
        val_totals=val_totals,
    )
    return {
        "implementation": "custom_regex_bpe",
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": int(bpe.vocab_size),
        "eos_id": int(bpe.eos_id),
        "train_time_sec": round(train_sec, 4),
        "tokenizer_path": saved_str,
        "tokenizer_hash": tok_hash,
        "embedding_cost": embedding_param_cost(bpe.vocab_size, d_models=d_models),
        **val_metrics,
    }


def _train_and_eval_hf_bpe(
    train_sample: Sequence[str],
    val_docs: Sequence[str],
    val_totals: dict[str, int],
    vocab_size: int,
) -> dict[str, Any]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    hf_tok = Tokenizer(models.BPE())
    hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf_tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=2,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    t0 = time.perf_counter()
    hf_tok.train_from_iterator(train_sample, trainer=trainer)
    train_sec = time.perf_counter() - t0

    def _encode_hf_batches(docs: Sequence[str], batch_size: int = 512) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(docs), batch_size):
            encs = hf_tok.encode_batch(docs[start : start + batch_size], add_special_tokens=False)
            lengths.extend(len(enc.ids) for enc in encs)
        return lengths

    val_metrics = evaluate_token_counts_on_val(
        _encode_hf_batches,
        val_docs=val_docs,
        val_totals=val_totals,
    )
    return {
        "implementation": "hf_tokenizers_byte_bpe",
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": int(hf_tok.get_vocab_size()),
        "train_time_sec": round(train_sec, 4),
        **val_metrics,
    }


def format_benchmark_markdown(report: dict[str, Any]) -> str:
    """Format vocabulary survey and HF comparison results into Markdown tables."""
    train_info = report["train_sample"]
    val_info = report["val_corpus"]

    lines: list[str] = [
        "# Khảo sát kích thước từ vựng & Đối chiếu thư viện chuẩn (Gói 4)",
        "",
        (
            f"- **Mẫu train:** {train_info['n_docs']:,} bài | "
            f"{train_info['n_bytes'] / (1024 * 1024):.2f} MiB | "
            f"{train_info['n_chars']:,} ký tự | {train_info['n_words']:,} từ (tách khoảng trắng)"
        ),
        (
            f"- **Tập val:** {val_info['n_docs']:,} bài | "
            f"{val_info['n_bytes'] / (1024 * 1024):.2f} MiB | "
            f"{val_info['n_chars']:,} ký tự | {val_info['n_words']:,} từ"
        ),
        "",
        "## 1. So sánh nén, Fertility và Tốc độ trên tập `val`",
        "",
        "| Tokenizer | Vocab (actual) | Train (s) | Val Encode (s) | Val Tokens | Bytes / Token | Chars / Token | Fertility (tok/word) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in report["comparisons"]:
        lines.append(
            f"| {row['implementation']} | {row['actual_vocab_size']} | "
            f"{row['train_time_sec']:.2f} | {row['val_encode_time_sec']:.2f} | "
            f"{row['n_tokens']:,} | {row['bytes_per_token']:.4f} | "
            f"{row['chars_per_token']:.4f} | {row['fertility_tokens_per_word']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Chi phí tham số Embedding (weight tying: $V \\times d_{\\text{model}}$)",
            "",
            "| Vocab Size | $d_{\\text{model}}$ | Embedding Params | % của 10M | % của 30M |",
            "|---:|---:|---:|---:|---:|",
        ]
    )

    for row in report["custom_bpe_results"]:
        v = row["actual_vocab_size"]
        for emb in row["embedding_cost"]:
            lines.append(
                f"| {v} | {emb['d_model']} | {emb['embedding_params']:,} | "
                f"{emb['pct_of_10M']:.2f}% | {emb['pct_of_30M']:.2f}% |"
            )

    lines.append("")
    return "\n".join(lines)


def run_vocab_benchmark(
    train_docs_iter: Iterable[str],
    val_docs_iter: Iterable[str],
    out_dir: str | Path,
    train_mb: float | None = 40.0,
    vocab_sizes: Sequence[int] = DEFAULT_VOCAB_SIZES,
    d_models: Sequence[int] = DEFAULT_D_MODELS,
    include_hf: bool = True,
) -> dict[str, Any]:
    """Run full vocab-size survey across `CustomBPE` and optional HF `tokenizers`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    max_bytes = int(float(train_mb) * 1024 * 1024) if train_mb is not None else None
    train_sample = list(slice_docs_by_utf8_bytes(train_docs_iter, max_bytes=max_bytes))
    if not train_sample:
        raise ValueError("Train sample is empty; check cleaned splits or train_mb")

    val_docs = list(val_docs_iter)
    if not val_docs:
        raise ValueError("Validation split is empty; check cleaned splits")

    train_totals = compute_corpus_text_totals(train_sample)
    val_totals = compute_corpus_text_totals(val_docs)

    custom_results: list[dict[str, Any]] = []
    hf_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for vocab_size in vocab_sizes:
        tok_file = out_path / f"custom_bpe{int(vocab_size)}.json"
        custom_res = _train_and_eval_custom_bpe(
            train_sample=train_sample,
            val_docs=val_docs,
            val_totals=val_totals,
            vocab_size=int(vocab_size),
            save_path=tok_file,
            d_models=d_models,
        )
        custom_results.append(custom_res)
        comparisons.append(custom_res)

        if include_hf:
            hf_res = _train_and_eval_hf_bpe(
                train_sample=train_sample,
                val_docs=val_docs,
                val_totals=val_totals,
                vocab_size=int(vocab_size),
            )
            hf_results.append(hf_res)
            comparisons.append(hf_res)

    report: dict[str, Any] = {
        "train_mb_limit": train_mb,
        "train_sample": train_totals,
        "val_corpus": val_totals,
        "vocab_sizes": [int(v) for v in vocab_sizes],
        "d_models": [int(d) for d in d_models],
        "custom_bpe_results": custom_results,
        "hf_bpe_results": hf_results,
        "comparisons": comparisons,
    }

    json_path = out_path / "vocab_benchmark.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_text = format_benchmark_markdown(report)
    md_path = out_path / "vocab_benchmark.md"
    md_path.write_text(md_text, encoding="utf-8")

    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    return report


def _classify_learned_token(raw_bytes: bytes) -> tuple[str, str]:
    """Classify a learned BPE token's byte sequence into quality categories."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return "partial_utf8", raw_bytes.decode("utf-8", errors="replace")

    for ch in text:
        if ch not in ("\n", "\t", "\r") and unicodedata.category(ch) in ("Cc", "Cf"):
            return "control_or_foreign", text

    if all(ch in " \t\r\n" for ch in text):
        return "whitespace_only", text

    has_latin_letter = False
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            if "LATIN" in unicodedata.name(ch, ""):
                has_latin_letter = True
            else:
                return "control_or_foreign", text

    if has_latin_letter:
        return "letter", text
    return "digit_or_punct", text


def inspect_tokenizer_vocab(
    tokenizer: CustomBPE,
    sample_limit: int = 30,
) -> dict[str, Any]:
    """Inspect learned merge tokens (IDs 257..vocab_size-1) for byte length and composition."""
    vocab_map = tokenizer.vocab_bytes
    learned_ids = range(257, int(tokenizer.vocab_size))
    n_learned = max(0, int(tokenizer.vocab_size) - 257)

    counts: dict[str, int] = {
        "letter": 0,
        "digit_or_punct": 0,
        "whitespace_only": 0,
        "partial_utf8": 0,
        "control_or_foreign": 0,
    }
    total_bytes = 0
    first_samples: list[dict[str, Any]] = []
    letter_entries: list[dict[str, Any]] = []

    for idx, tid in enumerate(learned_ids):
        raw = vocab_map[tid]
        byte_len = len(raw)
        total_bytes += byte_len
        category, text = _classify_learned_token(raw)
        counts[category] += 1

        entry = {
            "token_id": tid,
            "byte_len": byte_len,
            "category": category,
            "text": text,
        }
        if idx < sample_limit:
            first_samples.append(entry)
        if category == "letter":
            letter_entries.append(entry)

    longest_letter = sorted(
        letter_entries,
        key=lambda item: (item["byte_len"], -item["token_id"]),
        reverse=True,
    )[: min(20, sample_limit)]

    denom = n_learned if n_learned > 0 else 1
    valid_utf8_count = n_learned - counts["partial_utf8"]
    return {
        "vocab_size": int(tokenizer.vocab_size),
        "n_learned_tokens": n_learned,
        "avg_learned_byte_length": round(total_bytes / denom, 4) if n_learned > 0 else 0.0,
        "category_counts": counts,
        "valid_utf8_ratio": round(valid_utf8_count / denom, 4) if n_learned > 0 else 0.0,
        "letter_token_ratio": round(counts["letter"] / denom, 4) if n_learned > 0 else 0.0,
        "whitespace_only_ratio": round(counts["whitespace_only"] / denom, 4) if n_learned > 0 else 0.0,
        "partial_or_control_ratio": round(
            (counts["partial_utf8"] + counts["control_or_foreign"]) / denom, 4
        )
        if n_learned > 0
        else 0.0,
        "first_learned_tokens": first_samples,
        "longest_letter_tokens": longest_letter,
    }


def analyze_token_usage(
    bin_path: str | Path,
    vocab_size: int,
    tokenizer: BaseTokenizer | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Compute token usage frequencies and dead-token counts on a uint16 `.bin` file."""
    v_size = int(vocab_size)
    if v_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {v_size}")

    bin_file = Path(bin_path)
    if bin_file.stat().st_size == 0:
        return {
            "bin_path": str(bin_path),
            "vocab_size": v_size,
            "total_tokens": 0,
            "dead_token_count": v_size,
            "dead_token_ratio": 1.0,
            "learned_dead_token_count": max(0, v_size - 257),
            "learned_dead_token_ratio": 1.0 if v_size > 257 else 0.0,
            "dead_token_ids_sample": list(range(min(30, v_size))),
            "top_tokens": [],
        }

    arr = np.memmap(bin_file, dtype="<u2", mode="r")
    total_tokens = int(len(arr))

    max_id = int(arr.max())
    if max_id >= v_size:
        raise ValueError(f"Found token ID {max_id} >= vocab_size {v_size} in {bin_path}")

    freqs = np.bincount(arr, minlength=v_size)[:v_size]
    dead_ids = np.flatnonzero(freqs == 0)
    dead_count = int(len(dead_ids))

    learned_dead_count = int(np.sum(freqs[257:v_size] == 0)) if v_size > 257 else 0
    learned_total = max(0, v_size - 257)

    k = min(int(top_k), v_size)
    top_ids = np.argsort(freqs)[::-1][:k]
    top_tokens: list[dict[str, Any]] = []
    for tid_np in top_ids:
        tid = int(tid_np)
        cnt = int(freqs[tid])
        if cnt == 0:
            break
        entry: dict[str, Any] = {
            "token_id": tid,
            "count": cnt,
            "pct": round((cnt / total_tokens) * 100.0, 4),
        }
        if tokenizer is not None:
            entry["text"] = tokenizer.decode([tid])
        top_tokens.append(entry)

    return {
        "bin_path": str(bin_path),
        "vocab_size": v_size,
        "total_tokens": total_tokens,
        "dead_token_count": dead_count,
        "dead_token_ratio": round(dead_count / v_size, 4),
        "learned_dead_token_count": learned_dead_count,
        "learned_dead_token_ratio": round(learned_dead_count / learned_total, 4)
        if learned_total > 0
        else 0.0,
        "dead_token_ids_sample": [int(x) for x in dead_ids[:30]],
        "top_tokens": top_tokens,
    }
