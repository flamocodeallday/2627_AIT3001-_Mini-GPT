"""CLI entrypoint for Gói 4 (Step 1): benchmark vocab sizes and compare against HF tokenizers."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.clean import clean_and_dedup_stream
from src.data.config import load_data_config
from src.data.split import iter_jsonl_texts, resolve_split_files, split_by_document
from src.data.vocab_eval import (
    DEFAULT_D_MODELS,
    DEFAULT_VOCAB_SIZES,
    format_benchmark_markdown,
    run_vocab_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CustomBPE vocab sizes and compare with HF tokenizers.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data YAML config.")
    parser.add_argument(
        "--source",
        choices=("interim", "fixtures"),
        default="interim",
        help="Read cleaned splits from interim_dir/split_stats.json or clean fixtures on the fly.",
    )
    parser.add_argument("--interim-dir", type=str, default=None, help="Override directory containing split files.")
    parser.add_argument("--train-mb", type=float, default=None, help="Train sample budget in MiB (default: config).")
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_VOCAB_SIZES),
        help="List of vocab sizes to evaluate (default: 4096 8192 16384).",
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=list(DEFAULT_D_MODELS),
        help="Hidden sizes d_model for embedding parameter budget table.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/vocab_benchmark",
        help="Output directory for trained tokenizers (.json) and benchmark reports.",
    )
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace tokenizers comparison.")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    bpe_cfg = cfg.get("bpe", {})
    train_mb = args.train_mb if args.train_mb is not None else float(bpe_cfg.get("train_sample_mb", 40.0))

    if args.source == "fixtures":
        clean_cfg = cfg["cleaning"]
        split_cfg = cfg["split"]
        raw_iter = iter_jsonl_texts(cfg["paths"]["fixtures_file"])
        cleaned = clean_and_dedup_stream(
            raw_iter,
            min_chars=int(clean_cfg["min_chars"]),
            min_valid_char_ratio=float(clean_cfg["min_valid_char_ratio"]),
            min_letter_ratio=float(clean_cfg.get("min_letter_ratio", 0.45)),
            strip_sections=clean_cfg.get("strip_sections"),
        )
        splits = split_by_document(cleaned, ratios=split_cfg["ratios"], seed=int(split_cfg["seed"]))
        train_docs = splits["train"]
        val_docs = splits["val"]
    else:
        interim_dir = Path(args.interim_dir or cfg["paths"]["interim_dir"])
        split_files = resolve_split_files(interim_dir)
        train_docs = iter_jsonl_texts(split_files["train"])
        val_docs = iter_jsonl_texts(split_files["val"])

    report = run_vocab_benchmark(
        train_docs_iter=train_docs,
        val_docs_iter=val_docs,
        out_dir=args.out_dir,
        train_mb=train_mb,
        vocab_sizes=args.vocab_sizes,
        d_models=args.d_models,
        include_hf=not args.skip_hf,
    )
    print(format_benchmark_markdown(report))
    print(f"Saved JSON report: {report['artifacts']['json']}")
    print(f"Saved Markdown report: {report['artifacts']['markdown']}")


if __name__ == "__main__":
    main()
