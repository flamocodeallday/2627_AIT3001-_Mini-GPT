import os
import csv
import argparse
from typing import List
from src.eval.base import DatasetBundle, EvalMetrics, evaluate_model
from src.eval.ngram import NGramLM
from src.eval.lstm import LSTMLM


def export_summary_tables(results: List[EvalMetrics], out_prefix: str):
    """Xuất bảng so sánh dạng Markdown và CSV cho báo cáo ACL."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"

    headers = [
        "Model",
        "Data",
        "Split",
        "Tag",
        "Total Params",
        "Non-Emb Params",
        "Avg NLL (nats)",
        "Perplexity",
        "BPC",
    ]

    rows = []
    for r in results:
        rows.append([
            r.model_name,
            r.data_version,
            r.split,
            r.tag,
            f"{r.params.get('total_params', 0):,}",
            f"{r.params.get('non_embedding_params', 0):,}",
            f"{r.avg_nll:.4f}",
            f"{r.perplexity:.2f}",
            f"{r.bpc:.4f}",
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
    parser.add_argument("--data_dir", type=str, default="data/processed/v0_byte")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--ngram_train_tokens", type=int, default=1_000_000)
    parser.add_argument("--eval_max_tokens", type=int, default=100_000)
    parser.add_argument("--lstm_ckpt", type=str, default="results/checkpoints/lstm_l2_h256_v0_byte_best.pt")
    args = parser.parse_args()

    ds = DatasetBundle(args.data_dir)
    train_tokens = ds.get_split_tokens("train")
    all_results: List[EvalMetrics] = []

    # 1. Baseline N-gram (n=3 và n=4 với Kneser-Ney smoothing)
    for order in [3, 4]:
        print(f"Đang huấn luyện & đánh giá N-gram (order={order})...")
        ngram = NGramLM(order=order, vocab_size=ds.vocab_size, tokenizer_hash=ds.tokenizer_hash)
        ngram.fit(train_tokens, max_train_tokens=args.ngram_train_tokens)
        res = evaluate_model(
            ngram, ds, split=args.split, ctx=args.ctx, max_tokens=args.eval_max_tokens, save_dir="results"
        )
        all_results.append(res)

    # 2. Baseline LSTM (từ checkpoint đã huấn luyện)
    if os.path.exists(args.lstm_ckpt):
        print(f"Đang đánh giá LSTM từ checkpoint: {args.lstm_ckpt}...")
        lstm = LSTMLM.from_checkpoint(args.lstm_ckpt)
        res_lstm = evaluate_model(
            lstm, ds, split=args.split, ctx=args.ctx, max_tokens=args.eval_max_tokens, save_dir="results"
        )
        all_results.append(res_lstm)
    else:
        print(f"[Cảnh báo] Không tìm thấy checkpoint LSTM tại {args.lstm_ckpt}, bỏ qua.")

    # In bảng tóm tắt ra màn hình và lưu file
    out_prefix = f"results/summary_{ds.version}_{args.split}"
    export_summary_tables(all_results, out_prefix)


if __name__ == "__main__":
    main()