import os
import time
import math
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.eval.base import DatasetBundle, evaluate_model
from src.eval.lstm import LSTMLM, LSTMConfig


def get_batch(data: np.ndarray, batch_size: int, ctx: int, device: torch.device):
    ix = torch.randint(len(data) - ctx - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + ctx].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def train_lstm(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = DatasetBundle(args.data_dir)
    train_data = ds.get_split_tokens("train")

    cfg = LSTMConfig(
        vocab_size=ds.vocab_size,
        embed_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=True,
    )
    model = LSTMLM(
        config=cfg,
        tokenizer_hash=ds.tokenizer_hash,
        name=f"lstm_l{args.num_layers}_h{args.hidden_dim}",
    )
    params_info = model.count_params()
    print(f"Khởi tạo {model.name} trên {model.device} | Params: {params_info}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs("results/checkpoints", exist_ok=True)
    log_dir = os.path.join("runs", f"{model.name}_{ds.version}_s{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    ckpt_path = os.path.join("results/checkpoints", f"{model.name}_{ds.version}_best.pt")

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        model.train()
        # Cosine learning rate decay với warmup 5%
        warmup_steps = max(1, int(0.05 * args.max_steps))
        if step < warmup_steps:
            lr = args.lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
            lr = args.lr * 0.1 + 0.9 * args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch(train_data, args.batch_size, args.ctx, model.device)
        _, loss, _ = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"Train Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                f"Elapsed: {time.time() - t0:.1f}s"
            )

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_metrics = evaluate_model(
                model=model,
                dataset=ds,
                split="val",
                ctx=args.ctx,
                max_tokens=args.eval_max_tokens,
                save_dir=log_dir,
            )
            writer.add_scalar("val/loss", val_metrics.avg_nll, step)
            writer.add_scalar("val/perplexity", val_metrics.perplexity, step)
            writer.add_scalar("val/bpc", val_metrics.bpc, step)
            print(
                f"  >>> [EVAL Step {step}] Val Loss: {val_metrics.avg_nll:.4f} | "
                f"PPL: {val_metrics.perplexity:.2f} | BPC: {val_metrics.bpc:.4f} "
                f"[{val_metrics.tag}]"
            )

            if val_metrics.avg_nll < best_val_loss:
                best_val_loss = val_metrics.avg_nll
                model.save_checkpoint(
                    path=ckpt_path,
                    tokenizer_meta={
                        "version": ds.version,
                        "tokenizer_hash": ds.tokenizer_hash,
                        "vocab_size": ds.vocab_size,
                    },
                    step=step,
                    val_loss=best_val_loss,
                )

    writer.close()
    print(f"\nHoàn tất huấn luyện LSTM! Best Val Loss: {best_val_loss:.4f} | Saved: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline LSTMLM tự cài đặt")
    parser.add_argument("--data_dir", type=str, default="data/processed/v0_byte")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_max_tokens", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lstm(args)