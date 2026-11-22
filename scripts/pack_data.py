"""CLI entrypoint for Gói 2 & Gói 5: pack cleaned splits into uint16 `.bin` files and `meta.json`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.data.bpe import CustomBPE
from src.data.config import load_data_config
from src.data.pack import ByteTokenizer, build_bins
from src.data.split import iter_jsonl_texts, resolve_split_files
from src.data.tokenizer_base import BaseTokenizer, compute_file_sha256


def sanity_check_bins(processed_dir: Path, tokenizer: BaseTokenizer, preview_tokens: int = 120) -> None:
    """Verify `.bin` integrity against `meta.json`, hash consistency, and decode a preview slice."""
    meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab_size"])
    eos_id = int(meta["eos_id"])

    tok_file = processed_dir / "tokenizer.json"
    assert tok_file.is_file(), f"Missing {tok_file}"
    assert meta["tokenizer_hash"] == compute_file_sha256(tok_file), "tokenizer_hash mismatch"

    for split_name in ("train", "val", "test"):
        arr = np.memmap(processed_dir / f"{split_name}.bin", dtype="<u2", mode="r")
        split_meta = meta["splits"][split_name]
        assert len(arr) == split_meta["n_tokens"], f"{split_name}: len(arr) != meta n_tokens"
        if len(arr) > 0:
            assert int(arr.max()) < vocab_size, f"{split_name}: token ID >= vocab_size"
            eos_count = int(np.sum(arr == eos_id))
            expected_eos = max(0, int(split_meta["n_docs"]) - 1)
            assert eos_count == expected_eos, f"{split_name}: eos_count={eos_count} != {expected_eos}"
            preview = tokenizer.decode(arr[:preview_tokens].tolist())
            print(f"[{split_name}] tokens={len(arr):,}, eos_count={eos_count:,}, preview={preview[:120]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack cleaned splits into uint16 .bin files.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data YAML config.")
    parser.add_argument("--interim-dir", type=str, default=None, help="Directory with train/val/test.jsonl(.zst).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: processed_dir/version).")
    parser.add_argument("--version", type=str, default=None, help="Dataset version tag (default: v0_byte or v1_bpe<V>).")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional pre-trained CustomBPE JSON path (loads directly without retraining).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for parallel encoding (default: available CPU count).",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Documents per worker batch (default: 500).")
    parser.add_argument("--full", action="store_true", help="Ignore target_mb cap and encode entire splits.")
    parser.add_argument("--sanity-check", action="store_true", help="Run memmap verification after packing.")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    v0_cfg = cfg["v0_byte"]
    interim_dir = Path(args.interim_dir or cfg["paths"]["interim_dir"])

    if args.tokenizer_path:
        tok_src = Path(args.tokenizer_path)
        tokenizer: BaseTokenizer = CustomBPE.load(tok_src)
        tokenizer_type = "custom_regex_bpe"
        version = args.version or f"v1_bpe{tokenizer.vocab_size}"
        target_mb = None
    else:
        tok_src = None
        tokenizer = ByteTokenizer()
        tokenizer_type = "byte_level"
        version = args.version or v0_cfg["version"]
        target_mb = None if args.full else v0_cfg.get("target_mb")

    output_dir = Path(args.output_dir or (Path(cfg["paths"]["processed_dir"]) / version))

    split_files = resolve_split_files(interim_dir)
    splits = {
        name: iter_jsonl_texts(split_files[name])
        for name in ("train", "val", "test")
    }

    meta = build_bins(
        splits=splits,
        tokenizer=tokenizer,
        output_dir=output_dir,
        version=version,
        tokenizer_type=tokenizer_type,
        seed=int(cfg["split"]["seed"]),
        target_mb=target_mb,
        tokenizer_source_path=tok_src,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.sanity_check:
        sanity_check_bins(output_dir, tokenizer)


if __name__ == "__main__":
    main()
