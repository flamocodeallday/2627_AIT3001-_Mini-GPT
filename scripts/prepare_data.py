"""CLI entrypoint for Gói 1: stream, clean, deduplicate, and split Vietnamese Wikipedia."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path

from src.data.clean import clean_and_dedup_stream
from src.data.config import load_data_config
from src.data.split import iter_jsonl_texts, write_splits_streaming


def _iter_hf_wikipedia(dataset_cfg: dict, limit: int | None = None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config"],
        split=dataset_cfg.get("split", "train"),
        streaming=True,
    )
    text_field = dataset_cfg.get("text_field", "text")
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        yield row[text_field]


def _sample_jsonl_for_inspection(jsonl_path: Path, sample_n: int, seed: int) -> list[str]:
    """Reservoir-sample `sample_n` documents from a split file without loading all docs."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    for idx, text in enumerate(iter_jsonl_texts(jsonl_path)):
        if idx < sample_n:
            reservoir.append(text)
        else:
            j = rng.randint(0, idx)
            if j < sample_n:
                reservoir[j] = text
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and split corpus.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data YAML config.")
    parser.add_argument(
        "--source",
        choices=("fixtures", "hf"),
        default="fixtures",
        help="Use local fixtures JSONL or stream HuggingFace dataset.",
    )
    parser.add_argument("--input-jsonl", type=str, default=None, help="Optional custom input JSONL(.zst) file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override interim output directory.")
    parser.add_argument("--compress", action="store_true", help="Write .jsonl.zst instead of .jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max raw documents to read.")
    parser.add_argument("--inspect-n", type=int, default=5, help="Print N random cleaned train docs to inspect.")
    parser.add_argument("--inspect-out", type=str, default=None, help="Optional path to save sampled docs.")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    clean_cfg = cfg["cleaning"]
    split_cfg = cfg["split"]
    out_dir = Path(args.output_dir or cfg["paths"]["interim_dir"])

    if args.input_jsonl:
        raw_iter = iter_jsonl_texts(args.input_jsonl, text_field=cfg["dataset"].get("text_field", "text"))
    elif args.source == "fixtures":
        raw_iter = iter_jsonl_texts(cfg["paths"]["fixtures_file"])
    else:
        raw_iter = _iter_hf_wikipedia(cfg["dataset"], limit=args.limit)

    if args.limit is not None and (args.input_jsonl or args.source == "fixtures"):
        def _limited(it: Iterator[str], max_items: int) -> Iterator[str]:
            for idx, item in enumerate(it):
                if idx >= max_items:
                    break
                yield item
        raw_iter = _limited(raw_iter, args.limit)

    cleaned_stream = clean_and_dedup_stream(
        raw_iter,
        min_chars=int(clean_cfg["min_chars"]),
        min_valid_char_ratio=float(clean_cfg["min_valid_char_ratio"]),
        min_letter_ratio=float(clean_cfg.get("min_letter_ratio", 0.45)),
        strip_sections=clean_cfg.get("strip_sections"),
    )

    result = write_splits_streaming(
        cleaned_stream,
        output_dir=out_dir,
        ratios=split_cfg["ratios"],
        seed=int(split_cfg["seed"]),
        compress=args.compress,
    )
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))

    if args.inspect_n > 0:
        train_path = Path(result["files"]["train"])
        samples = _sample_jsonl_for_inspection(train_path, args.inspect_n, seed=int(split_cfg["seed"]))
        if args.inspect_out:
            inspect_path = Path(args.inspect_out)
            inspect_path.parent.mkdir(parents=True, exist_ok=True)
            with inspect_path.open("w", encoding="utf-8") as f:
                for i, doc in enumerate(samples, start=1):
                    f.write(f"=== SAMPLE {i} (len={len(doc)}) ===\n{doc}\n\n")
            print(f"Saved {len(samples)} inspection samples to {inspect_path}")
        else:
            for i, doc in enumerate(samples, start=1):
                preview = doc[:220].replace("\n", "\\n")
                print(f"[Sample {i} | len={len(doc)}] {preview}...")


if __name__ == "__main__":
    main()
