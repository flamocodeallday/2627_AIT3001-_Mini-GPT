import os
import csv
import argparse
from typing import List
from src.eval.base import DatasetBundle, EvalMetrics, evaluate_model
from src.eval.ngram import NGramLM
from src.eval.lstm import LSTMLM
from src.eval.gpt_adapter import GPTAdapter


def _build_gpt_from_src_model(cfg):
    """Tải lớp GPT và GPTConfig từ src.model của Thành viên 2."""
    try:
        from src.model.config import GPTConfig  # type: ignore
        import src.model as model_pkg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Không thể import src.model (code của TV2 chưa có trên nhánh này). "
            "Hãy merge nhánh TV2 trước khi đánh giá --gpt_ckpt."
        ) from exc

    gpt_cls = getattr(model_pkg, "MiniGPT", None) or getattr(model_pkg, "GPT", None)
    if gpt_cls is None:
        from src.model.gpt import GPT as gpt_cls  # type: ignore

    if isinstance(cfg, dict):
        cfg = GPTConfig(**cfg)
    return gpt_cls(cfg)


def export_summary_tables(results: List[EvalMetrics], out_prefix: str):
    """Xuất bảng so sánh dạng Markdown và CSV cho báo cáo ACL."""
    if not results:
        print("[Cảnh báo] Không có kết quả nào để xuất bảng.")
        return

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"

    headers = [
        "Model",
        "Data",
        "Split",
        "Tag",
        "Ctx",
        "Total Params",
        "Non-Emb Params",
        "Train Tokens",
        "Eval Tokens",
        "Avg NLL (nats)",
        "Perplexity",
        "BPC",
        "Git Commit",
    ]

    rows = []
    for r in results:
        commit_str = f"{r.git_commit}{'*' if r.git_dirty else ''}"
        train_tok_str = f"{r.train_tokens:,}" if r.train_tokens is not None else "N/A"
        rows.append([
            r.model_name,
            r.data_version,
            r.split,
            r.tag,
            str(r.ctx),
            f"{r.params.get('total_params', 0):,}",
            f"{r.params.get('non_embedding_params', 0):,}",
            train_tok_str,
            f"{r.num_tokens:,}",
            f"{r.avg_nll:.4f}",
            f"{r.perplexity:.2f}",
            f"{r.bpc:.4f}",
            commit_str,
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Bảng Tổng Hợp Kết Quả Đánh Giá ({results[0].data_version} - {results[0].split})\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\nĐã xuất bảng tổng hợp tại:\n  - {md_path}\n  - {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy đánh giá tổng hợp tất cả mô hình")
    parser.add_argument("--data_dir", type=str, default="data/processed/v1_bpe8192")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--ngram_train_tokens", type=int, default=10_000_000)
    parser.add_argument("--skip_ngram", action="store_true")
    parser.add_argument(
        "--eval_max_tokens",
        type=int,
        default=None,
        help="Mặc định None = đánh giá TOÀN BỘ split (official). Chỉ đặt số nhỏ khi debug.",
    )
    parser.add_argument("--lstm_ckpt", type=str, default=None, help="Đường dẫn 1 checkpoint LSTM")
    parser.add_argument("--lstm_ckpts", type=str, nargs="*", default=[], help="Danh sách checkpoint LSTM")
    parser.add_argument("--gpt_ckpt", type=str, default=None, help="Đường dẫn checkpoint GPT của TV2")
    parser.add_argument("--gpt_name", type=str, default="mini_gpt")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    ds = DatasetBundle(args.data_dir)
    all_results: List[EvalMetrics] = []

    # 1. Baseline N-gram (n=3 và n=4 với Interpolated Kneser-Ney smoothing)
    if not args.skip_ngram:
        train_tokens = ds.get_split_tokens("train")
        for order in [3, 4]:
            print(f"Đang huấn luyện & đánh giá N-gram (order={order}, fit={args.ngram_train_tokens:,} tokens)...")
            ngram = NGramLM(order=order, vocab_size=ds.vocab_size, tokenizer_hash=ds.tokenizer_hash)
            ngram.fit(train_tokens, max_train_tokens=args.ngram_train_tokens)
            res = evaluate_model(
                ngram,
                ds,
                split=args.split,
                ctx=args.ctx,
                max_tokens=args.eval_max_tokens,
                save_dir=args.save_dir,
            )
            all_results.append(res)

    # 2. Baseline LSTM (hỗ trợ cả LSTM nhỏ và LSTM lớn lstm_l3_h1024)
    lstm_paths: List[str] = list(args.lstm_ckpts)
    if args.lstm_ckpt and args.lstm_ckpt not in lstm_paths:
        lstm_paths.insert(0, args.lstm_ckpt)

    for ckpt_path in lstm_paths:
        if os.path.exists(ckpt_path):
            print(f"Đang đánh giá LSTM từ checkpoint: {ckpt_path}...")
            lstm = LSTMLM.from_checkpoint(ckpt_path)
            res_lstm = evaluate_model(
                lstm,
                ds,
                split=args.split,
                ctx=args.ctx,
                max_tokens=args.eval_max_tokens,
                save_dir=args.save_dir,
            )
            all_results.append(res_lstm)
        else:
            print(f"[Cảnh báo] Không tìm thấy checkpoint LSTM tại {ckpt_path}, bỏ qua.")

    # 3. Mô hình GPT chính của TV2
    if args.gpt_ckpt:
        if os.path.exists(args.gpt_ckpt):
            print(f"Đang đánh giá GPT từ checkpoint: {args.gpt_ckpt}...")
            gpt_adapter = GPTAdapter.from_checkpoint(
                ckpt_path=args.gpt_ckpt,
                model_builder=_build_gpt_from_src_model,
                name=args.gpt_name,
            )
            res_gpt = evaluate_model(
                gpt_adapter,
                ds,
                split=args.split,
                ctx=args.ctx,
                max_tokens=args.eval_max_tokens,
                save_dir=args.save_dir,
            )
            all_results.append(res_gpt)
        else:
            print(f"[Cảnh báo] Không tìm thấy checkpoint GPT tại {args.gpt_ckpt}, bỏ qua.")

    out_prefix = os.path.join(args.save_dir, f"summary_{ds.version}_{args.split}")
    export_summary_tables(all_results, out_prefix)


if __name__ == "__main__":
    main()
