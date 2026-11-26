"""End-to-end smoke test for member 2's GPT: train -> checkpoint -> reload -> eval -> generate.

    python -m scripts.smoke_test            # CPU, < 2 minutes on v0_byte

PASSes only after all TODO(todd) exercises are done. Each stage prints PASS / FAIL.
If member 3's `src/eval` is present (after merge), stage 4 also runs GPTAdapter + evaluate_model.
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from pathlib import Path

import torch

DATA = Path("data/processed/v0_byte")


def stage(name: str, fn) -> object:
    try:
        out = fn()
        print(f"[PASS] {name}")
        return out
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    from src.train import train as train_mod
    from src.train.checkpoint import build_model_from_checkpoint, gpt_model_builder, load_checkpoint
    from src.train.data import TokenBatcher, load_meta

    if not (DATA / "train.bin").exists():
        print(f"[FAIL] missing {DATA}. Copy the released v0_byte folder there first.")
        sys.exit(1)
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    meta = load_meta(DATA)

    def run_train():
        return train_mod.main([
            "--config", "configs/gpt/tiny_debug.yaml", "--device", "cpu", "--max_steps", "60",
            "--run_id", "smoke", "--out_dir", str(tmp / "ckpt"), "--log_dir", str(tmp / "logs"),
            "--results_dir", str(tmp / "results"),
        ])

    info = stage("1. train 60 steps on v0_byte", run_train)
    assert info["final_val_loss"] < math.log(meta["vocab_size"]), "val loss did not go below ln(V)"

    ckpt_path = tmp / "ckpt" / "smoke" / "last.pt"
    ckpt = stage("2. checkpoint has contract keys", lambda: load_checkpoint(ckpt_path))
    for key in ("state_dict", "config", "tokenizer_meta", "step", "val_loss"):
        assert key in ckpt, key

    model = stage("3. rebuild model from checkpoint", lambda: build_model_from_checkpoint(ckpt).eval())

    def evaluate():
        val = TokenBatcher(DATA, "val", 128, 8, seed=0)
        with torch.no_grad():
            losses = [float(model(*val.next_batch())[1]) for _ in range(5)]
        loss = sum(losses) / len(losses)
        print(f"      reloaded model val loss {loss:.4f} (train log said {info['final_val_loss']:.4f})")
        try:
            from src.eval.base import DatasetBundle, evaluate_model
            from src.eval.gpt_adapter import GPTAdapter
        except ImportError:
            print("      (member 3's src/eval not on this branch yet: skipping GPTAdapter check)")
            return
        adapter = GPTAdapter.from_checkpoint(str(ckpt_path), model_builder=gpt_model_builder, device="cpu")
        res = evaluate_model(adapter, DatasetBundle(str(DATA)), split="val", ctx=128, max_tokens=20_000, save_dir=None)
        print(f"      GPTAdapter: PPL {res.perplexity:.2f} | BPC {res.bpc:.4f} [{res.tag}]")

    stage("4. evaluate reloaded model", evaluate)

    def generate():
        from src.data.pack import load_tokenizer

        tok = load_tokenizer(DATA / "tokenizer.json")
        prompt = torch.tensor([tok.encode("Việt Nam là")], dtype=torch.long)
        out = model.generate(prompt, 80, temperature=0.8, top_k=40, generator=torch.Generator().manual_seed(0))
        print("      sample:", tok.decode(out[0].tolist()).replace("\n", " ")[:200])

    stage("5. generate text", generate)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
