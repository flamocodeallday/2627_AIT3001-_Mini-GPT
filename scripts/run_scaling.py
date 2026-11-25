import os
import json
import math
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.eval.base import DatasetBundle, evaluate_model
from src.eval.lstm import LSTMLM, LSTMConfig
from scripts.train_lstm import get_batch


# 4 cấu hình tăng dần kích thước để khảo sát Model Scaling Laws
SCALING_PRESETS = [
    {"name": "scale_xs", "num_layers": 1, "hidden_dim": 64},
    {"name": "scale_s",  "num_layers": 2, "hidden_dim": 128},
    {"name": "scale_m",  "num_layers": 2, "hidden_dim": 256},
    {"name": "scale_l",  "num_layers": 3, "hidden_dim": 384},
]


def plot_scaling_curve(records: list, out_png: str):
    """Vẽ đồ thị Model Scaling trên thang log-log và fit đường Power-Law (Kaplan et al., 2020)."""
    non_emb = np.array([r["non_embedding_params"] for r in records], dtype=np.float64)
    val_loss = np.array([r["val_loss"] for r in records], dtype=np.float64)

    # Fit tuyến tính trong không gian log10: log10(L) = -alpha * log10(N) + c
    log_n = np.log10(non_emb)
    log_l = np.log10(val_loss)
    slope, intercept = np.polyfit(log_n, log_l, 1)
    alpha = -slope

    n_grid = np.logspace(np.log10(non_emb.min() * 0.8), np.log10(non_emb.max() * 1.2), 100)
    l_fit = (10 ** intercept) * (n_grid ** slope)

    plt.figure(figsize=(7, 5), dpi=150)
    plt.loglog(non_emb, val_loss, "o", markersize=8, color="#1f77b4", label="Observed Val Loss")
    plt.loglog(
        n_grid,
        l_fit,
        "--",
        color="#d62728",
        linewidth=2,
        label=f"Power-law fit ($\\alpha_N = {alpha:.4f}$)",
    )

    for r in records:
        plt.annotate(
            r["name"],
            (r["non_embedding_params"], r["val_loss"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    plt.xlabel("Non-Embedding Parameters ($N$)", fontsize=11)
    plt.ylabel("Validation Cross-Entropy Loss (nats)", fontsize=11)
    plt.title("Model Scaling Law (Log-Log Scale)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Đã lưu biểu đồ Scaling Law tại: {out_png} (alpha_N = {alpha:.4f})")


def run_scaling_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = DatasetBundle(args.data_dir)
    train_data = ds.get_split_tokens("train")
    records = []

    for preset in SCALING_PRESETS:
        cfg = LSTMConfig(
            vocab_size=ds.vocab_size,
            embed_dim=preset["hidden_dim"],
            hidden_dim=preset["hidden_dim"],
            num_layers=preset["num_layers"],
            dropout=0.1,
            tie_weights=True,
        )
        model = LSTMLM(cfg, tokenizer_hash=ds.tokenizer_hash, name=preset["name"])
        params = model.count_params()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        t0 = time.time()
        model.train()
        warmup_steps = max(1, int(0.05 * args.steps_per_model))
        for step in range(1, args.steps_per_model + 1):
            if step < warmup_steps:
                lr = args.lr * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / max(1, args.steps_per_model - warmup_steps)
                lr = args.lr * 0.1 + 0.9 * args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            xb, yb = get_batch(train_data, args.batch_size, args.ctx, model.device)
            _, loss, _ = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate_model(
            model, ds, split="val", ctx=args.ctx, max_tokens=args.eval_max_tokens, save_dir=None
        )
        tokens_seen = args.steps_per_model * args.batch_size * args.ctx
        approx_flops = 6 * params["non_embedding_params"] * tokens_seen

        rec = {
            "name": preset["name"],
            "num_layers": preset["num_layers"],
            "hidden_dim": preset["hidden_dim"],
            "total_params": params["total_params"],
            "non_embedding_params": params["non_embedding_params"],
            "tokens_seen": tokens_seen,
            "approx_flops": approx_flops,
            "val_loss": metrics.avg_nll,
            "val_ppl": metrics.perplexity,
            "val_bpc": metrics.bpc,
            "train_time_sec": round(time.time() - t0, 1),
        }
        records.append(rec)
        print(
            f"[{rec['name']:<8}] N_non_emb={rec['non_embedding_params']:>8,} | "
            f"Val Loss={rec['val_loss']:.4f} | PPL={rec['val_ppl']:.2f} | "
            f"Time={rec['train_time_sec']}s"
        )

    os.makedirs("results", exist_ok=True)
    json_path = f"results/scaling_{ds.version}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    plot_scaling_curve(records, f"results/scaling_law_{ds.version}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Model Scaling Laws")
    parser.add_argument("--data_dir", type=str, default="data/processed/v0_byte")
    parser.add_argument("--steps_per_model", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval_max_tokens", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_scaling_experiment(args)