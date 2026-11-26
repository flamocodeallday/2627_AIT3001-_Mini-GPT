"""Train the GPT on a released data version.

    python -m src.train.train --config configs/gpt/tiny_debug.yaml            # CPU, ~1 min
    python -m src.train.train --config configs/gpt/main_30m.yaml --device cuda
    python -m src.train.train --config configs/gpt/scaling.yaml --preset scale_3m
    python -m src.train.train --config configs/gpt/main_30m.yaml --resume checkpoints/<run_id>/last.pt

Who wrote what: `train_step` is Todd's exercise (bài 11). Everything else is scaffolding.
Budget (decision C): tokens = tokens_per_param x N_non_emb unless `max_steps` / `tokens` is set.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ContextManager

import torch
import torch.nn as nn
import yaml

from src.model.config import GPTConfig, estimate_params, make_config
from src.model.gpt import GPT
from src.train.checkpoint import load_checkpoint, save_checkpoint, unwrap
from src.train.data import TokenBatcher, load_meta, tokenizer_meta_from, verify_tokenizer_hash
from src.train.optim import build_optimizer
from src.train.schedule import get_lr


# --------------------------------------------------------------------------------------
# Todd's exercise
# --------------------------------------------------------------------------------------
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    next_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    grad_accum_steps: int,
    grad_clip: float,
    autocast_ctx: ContextManager,
) -> dict[str, float]:
    """One optimizer step built from `grad_accum_steps` micro-batches. Bài 11.

    Required behaviour, in order:
      1. Clear old gradients (`optimizer.zero_grad(set_to_none=True)`).
      2. For each micro-batch: x, y = next_batch(); forward under `autocast_ctx`;
         scale the loss so that the ACCUMULATED gradient equals the gradient of the MEAN loss
         over all micro-batches; backward through `scaler.scale(...)`.
      3. Unscale the gradients (`scaler.unscale_(optimizer)`) BEFORE clipping, then clip the
         global norm to `grad_clip` and keep the norm value that the clip function returns.
      4. `scaler.step(optimizer)` then `scaler.update()`.
    Returns {"loss": mean un-scaled loss over micro-batches (float), "grad_norm": pre-clip norm (float)}.
    With grad_accum_steps=4 and quarter-size batches the update must equal 1 step on the full batch.
    """
    # TODO(todd) bài 11
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    for _ in range(grad_accum_steps):
        x, y = next_batch()
        with autocast_ctx:
            _, loss = model(x, y)
            scaled_loss = loss / grad_accum_steps
        total_loss += loss.item()
        scaler.scale(scaled_loss).backward()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": total_loss / grad_accum_steps,
        "grad_norm": float(grad_norm),
    }


# --------------------------------------------------------------------------------------
# Scaffolding
# --------------------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "data", "train", "eval", "log"):
        cfg.setdefault(section, {})
    return cfg


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()

    try:
        return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_and_scaler(device: torch.device) -> tuple[ContextManager, torch.amp.GradScaler, str]:
    """bf16 autocast when supported, fp16 + GradScaler on older GPUs (T4), fp32 on CPU."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16), torch.amp.GradScaler("cuda", enabled=False), "bf16"
        return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda", enabled=True), "fp16"
    return contextlib.nullcontext(), torch.amp.GradScaler("cpu", enabled=False), "fp32"


def compute_budget(cfg: dict[str, Any], model_cfg: GPTConfig) -> dict[str, int]:
    """Number of optimizer steps from the token budget (decision C)."""
    t = cfg["train"]
    tokens_per_step = int(t["batch_size"]) * int(t.get("grad_accum_steps", 1)) * model_cfg.block_size
    n_non_emb = estimate_params(model_cfg)["non_embedding"]
    if t.get("max_steps"):
        total_steps = int(t["max_steps"])
    elif t.get("tokens"):
        total_steps = math.ceil(float(t["tokens"]) / tokens_per_step)
    else:
        total_steps = math.ceil(float(t.get("tokens_per_param", 20)) * n_non_emb / tokens_per_step)
    warmup = int(t["warmup_steps"]) if t.get("warmup_steps") is not None else max(1, int(float(t.get("warmup_frac", 0.02)) * total_steps))
    return {
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "warmup_steps": min(warmup, total_steps),
        "total_tokens": total_steps * tokens_per_step,
        "non_embedding_params": n_non_emb,
    }


@torch.no_grad()
def estimate_loss(model: nn.Module, batcher: TokenBatcher, iters: int, autocast_ctx: ContextManager) -> float:
    """Mean loss over `iters` fixed val batches (batcher.reset() -> same batches every time)."""
    was_training = model.training
    model.eval()
    batcher.reset()
    losses = []
    for _ in range(iters):
        x, y = batcher.next_batch()
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)


class Logger:
    """TensorBoard (default) and optional W&B; silently degrades if a backend is missing."""

    def __init__(self, run_id: str, log_dir: Path, use_wandb: bool, config: dict[str, Any]) -> None:
        self.tb = None
        self.wandb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(log_dir=str(log_dir / "tb" / run_id))
        except Exception as exc:  # tensorboard not installed
            print(f"[log] TensorBoard disabled: {exc}")
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(project="mini-gpt-vi", name=run_id, config=config)
            except Exception as exc:
                print(f"[log] W&B disabled: {exc}")

    def scalars(self, values: dict[str, float], step: int) -> None:
        if self.tb is not None:
            for k, v in values.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def text(self, tag: str, value: str, step: int) -> None:
        if self.tb is not None:
            self.tb.add_text(tag, value, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


def try_sample(model: nn.Module, data_dir: Path, prompt: str, device: torch.device, n_tokens: int = 60) -> str | None:
    """Generate a short sample for the log; skipped until generate() (bài 8) is implemented."""
    try:
        from src.data.pack import load_tokenizer

        tok = load_tokenizer(data_dir / "tokenizer.json")
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        gen = torch.Generator(device=device).manual_seed(0)
        out = unwrap(model).generate(ids, n_tokens, temperature=0.8, top_k=50, generator=gen)
        return tok.decode(out[0].tolist())  # decode the whole sequence: a single token may be half a UTF-8 char
    except NotImplementedError:
        return None
    except Exception as exc:
        return f"[sample failed: {exc}]"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Train the mini-GPT")
    ap.add_argument("--config", required=True)
    ap.add_argument("--preset", help="override model.preset (scaling runs)")
    ap.add_argument("--data_dir", help="override data.dir")
    ap.add_argument("--device")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--run_id")
    ap.add_argument("--resume", help="path to last.pt to continue a run")
    ap.add_argument("--max_steps", type=int, help="override the budget (debugging)")
    ap.add_argument("--compile", action="store_true", help="torch.compile (Linux/Colab only, needs Triton)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    if args.preset:
        cfg["model"]["preset"] = args.preset
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps

    t = cfg["train"]
    seed = int(t.get("seed", 42))
    torch.manual_seed(seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(cfg["data"]["dir"])
    meta = load_meta(data_dir)
    verify_tokenizer_hash(data_dir, meta)
    tok_meta = tokenizer_meta_from(meta)

    resume_ckpt = load_checkpoint(args.resume) if args.resume else None
    if resume_ckpt is not None:
        model_cfg = GPTConfig.from_dict(resume_ckpt["config"])
        if resume_ckpt["tokenizer_meta"]["tokenizer_hash"] != tok_meta["tokenizer_hash"]:
            raise ValueError("Checkpoint was trained with a different tokenizer than data.dir")
    else:
        m = cfg["model"]
        model_cfg = make_config(
            m.get("preset", "main_30m"),
            vocab_size=int(meta["vocab_size"]),
            block_size=int(m.get("block_size", 512)),
            dropout=float(m.get("dropout", 0.0)),
        )

    budget = compute_budget(cfg, model_cfg)
    run_id = args.run_id or (resume_ckpt["train_args"].get("run_id") if resume_ckpt else None) or (
        f"gpt_{cfg['model'].get('preset', 'custom')}_{meta['version']}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    out_dir = Path(args.out_dir) / run_id
    autocast_ctx, scaler, precision = autocast_and_scaler(device)

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(
        model, lr=float(t.get("max_lr", 6e-4)), weight_decay=float(t.get("weight_decay", 0.1)),
        betas=tuple(t.get("betas", (0.9, 0.95))), device_type=device.type,
    )
    train_batcher = TokenBatcher(data_dir, "train", model_cfg.block_size, int(t["batch_size"]), device, seed=seed)
    e = cfg["eval"]
    val_batcher = TokenBatcher(data_dir, "val", model_cfg.block_size, int(e.get("batch_size", t["batch_size"])), device, seed=1234)

    start_step, best_val = 0, float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        if resume_ckpt.get("scaler"):
            scaler.load_state_dict(resume_ckpt["scaler"])
        if resume_ckpt.get("batcher"):
            train_batcher.load_state_dict(resume_ckpt["batcher"])
        torch.set_rng_state(resume_ckpt["torch_rng"])
        start_step = int(resume_ckpt["step"])
        best_val = resume_ckpt.get("best_val_loss") or float("inf")
        print(f"[resume] from step {start_step}, best val {best_val:.4f}")

    train_model: nn.Module = torch.compile(model) if args.compile else model
    params = {"total": model.count_params(False), "non_embedding": model.count_params(True)}
    lg = cfg["log"]
    eval_interval, eval_iters = int(e.get("interval", 500)), int(e.get("iters", 50))
    log_interval, ckpt_interval = int(lg.get("interval", 20)), int(lg.get("ckpt_interval", 1000))
    grad_accum, grad_clip = int(t.get("grad_accum_steps", 1)), float(t.get("grad_clip", 1.0))

    run_info = {
        "run_id": run_id, "git": git_info(), "config_file": str(args.config), "config": cfg,
        "model_config": model_cfg.to_dict(), "params": params, "budget": budget,
        "tokens_per_param": budget["total_tokens"] / max(params["non_embedding"], 1),
        "epochs": budget["total_tokens"] / int(meta["splits"]["train"]["n_tokens"]),
        "data_version": meta["version"], "tokenizer_hash": tok_meta["tokenizer_hash"],
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "precision": precision, "torch": torch.__version__,
    }
    print(json.dumps({k: run_info[k] for k in ("run_id", "params", "budget", "tokens_per_param", "epochs", "device", "precision")}, indent=2, default=str))
    logger = Logger(run_id, Path(args.log_dir), args.wandb, run_info)

    def checkpoint(name: str, step: int, val_loss: float | None) -> None:
        save_checkpoint(
            out_dir / name, model=model, tokenizer_meta=tok_meta, step=step, val_loss=val_loss,
            optimizer=optimizer, scaler=scaler, batcher_state=train_batcher.state_dict(),
            train_args={"run_id": run_id, "config": cfg, "budget": budget}, best_val_loss=best_val,
        )

    total_steps = budget["total_steps"]
    model.train()
    t0, last_val = time.time(), None
    step = start_step
    for step in range(start_step, total_steps):
        lr = get_lr(step, max_lr=float(t.get("max_lr", 6e-4)), min_lr=float(t.get("min_lr", 6e-5)),
                    warmup_steps=budget["warmup_steps"], total_steps=total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % eval_interval == 0 and step > start_step:
            last_val = estimate_loss(train_model, val_batcher, eval_iters, autocast_ctx)
            logger.scalars({"val/loss": last_val, "val/ppl": math.exp(min(last_val, 20))}, step)
            print(f"  >>> step {step}: val loss {last_val:.4f} | ppl {math.exp(min(last_val, 20)):.2f}")
            if last_val < best_val:
                best_val = last_val
                checkpoint("best.pt", step, last_val)
            sample = try_sample(model, data_dir, str(lg.get("sample_prompt", "Việt Nam là")), device)
            if sample:
                logger.text("sample", sample, step)

        t_step = time.time()
        stats = train_step(train_model, optimizer, scaler, train_batcher.next_batch, grad_accum, grad_clip, autocast_ctx)

        if step % log_interval == 0 or step == total_steps - 1:
            dt = time.time() - t_step
            tok_s = budget["tokens_per_step"] / max(dt, 1e-9)
            eta_min = (total_steps - step - 1) * dt / 60
            logger.scalars({"train/loss": stats["loss"], "train/grad_norm": stats["grad_norm"], "train/lr": lr, "train/tok_per_s": tok_s}, step)
            print(f"step {step:6d}/{total_steps} | loss {stats['loss']:.4f} | gnorm {stats['grad_norm']:.2f} | lr {lr:.2e} | {tok_s:,.0f} tok/s | ETA {eta_min:.1f} min")

        if (step + 1) % ckpt_interval == 0:
            checkpoint("last.pt", step + 1, last_val)

    # final evaluation + checkpoint
    final_step = total_steps
    last_val = estimate_loss(train_model, val_batcher, eval_iters, autocast_ctx)
    if last_val < best_val:
        best_val = last_val
        checkpoint("best.pt", final_step, last_val)
    checkpoint("last.pt", final_step, last_val)
    logger.scalars({"val/loss": last_val}, final_step)
    logger.close()

    run_info.update({
        "final_val_loss": last_val, "best_val_loss": best_val, "train_time_sec": round(time.time() - t0, 1),
        "checkpoint_best": str(out_dir / "best.pt"), "finished_at": datetime.now().isoformat(timespec="seconds"),
        "note": "val loss = mean over fixed random val batches (training monitor). Official PPL/BPC come from member 3's harness.",
    })
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"train_{run_id}.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"done: val {last_val:.4f} (best {best_val:.4f}) -> {out_dir}")
    return run_info


if __name__ == "__main__":
    main()
