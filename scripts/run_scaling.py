import os
import json
import math
import time
import argparse
from typing import List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.eval.base import DatasetBundle, evaluate_model, _get_git_info, _get_hardware_info
from src.eval.lstm import LSTMLM, LSTMConfig
from src.eval.gpt_adapter import GPTAdapter
from scripts.train_lstm import get_batch


# Đường so sánh phụ: Scaling trên LSTM (giữ lại để đối chiếu, ghi nhãn rõ)
LSTM_SCALING_PRESETS = [
    {"name": "lstm_xs", "num_layers": 1, "hidden_dim": 64},
    {"name": "lstm_s", "num_layers": 2, "hidden_dim": 128},
    {"name": "lstm_m", "num_layers": 2, "hidden_dim": 256},
    {"name": "lstm_l", "num_layers": 3, "hidden_dim": 384},
]


def _load_gpt_presets_and_cls(preset_names: List[str], vocab_size: int, ctx: int):
    """Đọc PRESETS, SCALING_LADDER và make_config từ src.model.config của Thành viên 2."""
    try:
        from src.model import GPT, PRESETS, SCALING_LADDER, make_config  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Không thể import src.model (PRESETS, SCALING_LADDER, GPT) của TV2. "
            "Hãy merge nhánh TV2 trước khi chạy --arch gpt, hoặc dùng --arch lstm để chạy đường phụ."
        ) from exc

    selected = preset_names if preset_names else list(SCALING_LADDER)
    configs = []
    for name in selected:
        if name not in PRESETS:
            raise ValueError(f"Preset '{name}' không có trong src.model.config.PRESETS: {list(PRESETS.keys())}")
        configs.append((name, make_config(name, vocab_size=vocab_size, block_size=ctx)))
    return configs, GPT


def plot_scaling_curve(records: list, out_png: str, arch: str):
    """Vẽ đồ thị Model Scaling trên thang log-log và fit đường Power-Law (Kaplan et al., 2020)."""
    non_emb = np.array([r["non_embedding_params"] for r in records], dtype=np.float64)
    val_loss = np.array([r["val_loss"] for r in records], dtype=np.float64)

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

    title_suffix = "GPT (Chính)" if arch == "gpt" else "LSTM (Đường so sánh phụ)"
    plt.xlabel("Non-Embedding Parameters ($N_{\\text{non-emb}}$)", fontsize=11)
    plt.ylabel("Validation Cross-Entropy Loss (nats)", fontsize=11)
    plt.title(f"Model Scaling Law — {title_suffix}", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Đã lưu biểu đồ Scaling Law tại: {out_png} (alpha_N = {alpha:.4f})")


def _train_one_run(
    model_nn: torch.nn.Module,
    eval_wrapper,
    name: str,
    ds: DatasetBundle,
    train_data: np.ndarray,
    args,
    is_gpt: bool,
) -> Dict[str, Any]:
    params = eval_wrapper.count_params()
    n_non_emb = params["non_embedding_params"]

    # Luật ngân sách đã chốt (Quyết định C): D = token_multiplier * N_non_emb (mặc định 20x)
    target_tokens = int(args.token_multiplier * n_non_emb)
    tokens_per_step = args.batch_size * args.grad_accum * args.ctx
    if args.max_steps_override is not None:
        total_steps = args.max_steps_override
    else:
        total_steps = max(1, math.ceil(target_tokens / tokens_per_step))

    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    min_lr = args.lr * args.min_lr_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn.to(device)

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    optimizer = torch.optim.AdamW(
        model_nn.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95) if is_gpt else (0.9, 0.999),
        weight_decay=args.weight_decay if is_gpt else 0.01,
    )

    print(
        f"\n>>> Bắt đầu [{args.arch.upper()} | {name}] N_non_emb={n_non_emb:,} | "
        f"Target D={target_tokens:,} | Steps={total_steps} | Device={device} | AMP={use_amp}"
    )

    t0 = time.time()
    model_nn.train()
    for step in range(1, total_steps + 1):
        if step < warmup_steps:
            lr = args.lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = min_lr + (args.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.batch_size, args.ctx, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if is_gpt:
                    _, loss = model_nn(xb, targets=yb)
                else:
                    _, loss, _ = model_nn(xb, yb)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.item()) / args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_interval == 0 or step == total_steps:
            print(
                f"  [{name}] Step {step:5d}/{total_steps} | Loss: {accum_loss:.4f} | "
                f"LR: {lr:.2e} | Elapsed: {time.time() - t0:.1f}s"
            )

    train_time_sec = round(time.time() - t0, 1)
    tokens_seen = total_steps * tokens_per_step
    approx_flops = 6 * n_non_emb * tokens_seen
    git_commit, git_dirty = _get_git_info()

    train_config = {
        "arch": args.arch,
        "preset": name,
        "ctx": args.ctx,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "token_multiplier": args.token_multiplier,
        "seed": args.seed,
    }

    metrics = evaluate_model(
        eval_wrapper,
        ds,
        split="val",
        ctx=args.ctx,
        max_tokens=args.eval_max_tokens,
        save_dir=None,
        train_tokens=tokens_seen,
        train_config=train_config,
    )

    rec = {
        "arch": args.arch,
        "name": name,
        "total_params": params["total_params"],
        "non_embedding_params": n_non_emb,
        "target_tokens": target_tokens,
        "tokens_seen": tokens_seen,
        "tokens_per_non_emb_param": round(tokens_seen / max(1, n_non_emb), 2),
        "approx_flops": approx_flops,
        "val_loss": metrics.avg_nll,
        "val_ppl": metrics.perplexity,
        "val_bpc": metrics.bpc,
        "eval_tokens": metrics.num_tokens,
        "is_full_split": metrics.is_full_split,
        "tag": metrics.tag,
        "train_config": train_config,
        "hardware": _get_hardware_info(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "train_time_sec": train_time_sec,
    }
    print(
        f"[{args.arch.upper()} | {name:<10}] N_non_emb={n_non_emb:>10,} | "
        f"D={tokens_seen:>11,} ({rec['tokens_per_non_emb_param']:.1f}x) | "
        f"Val Loss={rec['val_loss']:.4f} | PPL={rec['val_ppl']:.2f} | Time={train_time_sec}s"
    )
    return rec


def run_scaling_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = DatasetBundle(args.data_dir)
    train_data = ds.get_split_tokens("train")
    records = []

    if args.arch == "gpt":
        configs, gpt_cls = _load_gpt_presets_and_cls(args.presets, ds.vocab_size, args.ctx)
        for name, cfg in configs:
            gpt_model = gpt_cls(cfg)
            adapter = GPTAdapter(
                gpt_model=gpt_model,
                vocab_size=ds.vocab_size,
                tokenizer_hash=ds.tokenizer_hash,
                name=name,
            )
            rec = _train_one_run(gpt_model, adapter, name, ds, train_data, args, is_gpt=True)
            records.append(rec)
    else:
        for preset in LSTM_SCALING_PRESETS:
            if args.presets and preset["name"] not in args.presets:
                continue
            cfg = LSTMConfig(
                vocab_size=ds.vocab_size,
                embed_dim=preset["hidden_dim"],
                hidden_dim=preset["hidden_dim"],
                num_layers=preset["num_layers"],
                dropout=0.1,
                tie_weights=True,
            )
            model = LSTMLM(cfg, tokenizer_hash=ds.tokenizer_hash, name=preset["name"])
            rec = _train_one_run(model, model, preset["name"], ds, train_data, args, is_gpt=False)
            records.append(rec)

    os.makedirs(args.save_dir, exist_ok=True)
    json_path = os.path.join(args.save_dir, f"scaling_{args.arch}_{ds.version}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if len(records) >= 2:
        plot_scaling_curve(
            records,
            os.path.join(args.save_dir, f"scaling_law_{args.arch}_{ds.version}.png"),
            arch=args.arch,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Model Scaling Laws (GPT chính / LSTM phụ)")
    parser.add_argument("--arch", type=str, choices=["gpt", "lstm"], default="gpt")
    parser.add_argument("--data_dir", type=str, default="data/processed/v1_bpe8192")
    parser.add_argument("--presets", type=str, nargs="*", default=[], help="Danh sách preset cần chạy")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Bật BF16/FP16 Mixed Precision trên CUDA")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--token_multiplier",
        type=float,
        default=20.0,
        help="Luật ngân sách D = token_multiplier * N_non_emb (mặc định 20 theo Quyết định C)",
    )
    parser.add_argument("--max_steps_override", type=int, default=None, help="Chỉ dùng để debug nhanh")
    parser.add_argument("--eval_max_tokens", type=int, default=None, help="Mặc định None = toàn bộ val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    run_scaling_experiment(args)
