import json
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from src.eval.base import DatasetBundle, evaluate_model
from src.eval.gpt_adapter import GPTAdapter


class DummyGPT(nn.Module):
    """Mô hình GPT giả lập tuân thủ Hợp đồng giao diện v0 để kiểm thử GPTAdapter."""

    def __init__(self, vocab_size: int, n_embd: int = 32, block_size: int = 128, uniform_logits: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.uniform_logits = uniform_logits
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        if self.uniform_logits:
            logits = torch.zeros((B, T, self.vocab_size), dtype=torch.float32, device=idx.device)
        else:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            logits = self.lm_head(self.proj(x))
        return logits, None


def _make_dataset(tmp_path, vocab_size=50, n_val=300):
    rng = np.random.default_rng(99)
    val_arr = rng.integers(0, vocab_size, size=n_val, dtype=np.uint16)
    data_dir = tmp_path / "v1_dummy"
    data_dir.mkdir(parents=True, exist_ok=True)
    val_arr.tofile(data_dir / "val.bin")
    meta = {
        "version": "v1_dummy",
        "tokenizer_type": "bpe",
        "vocab_size": vocab_size,
        "eos_id": 0,
        "seed": 42,
        "tokenizer_hash": "dummy_gpt_hash_123456",
        "splits": {"val": {"n_tokens": n_val, "n_chars": n_val * 4}},
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return DatasetBundle(str(data_dir))


def test_gpt_adapter_uniform_nll_and_shape(tmp_path):
    ds = _make_dataset(tmp_path, vocab_size=50, n_val=300)
    gpt = DummyGPT(vocab_size=ds.vocab_size, n_embd=32, block_size=128, uniform_logits=True)
    adapter = GPTAdapter(gpt, vocab_size=ds.vocab_size, tokenizer_hash=ds.tokenizer_hash, device="cpu")

    # Kiểm tra shape forward
    x = torch.randint(0, ds.vocab_size, (2, 16))
    logits, loss = gpt(x)
    assert logits.shape == (2, 16, ds.vocab_size)
    assert loss is None

    # Phân phối đều -> avg_nll == ln(V) và PPL == V
    res = evaluate_model(adapter, ds, split="val", ctx=64, max_tokens=None, save_dir=None)
    assert res.num_tokens == 299
    assert math.isclose(res.avg_nll, math.log(ds.vocab_size), rel_tol=1e-5)
    assert math.isclose(res.perplexity, float(ds.vocab_size), rel_tol=1e-4)
    assert res.tag == "official"

    # Kiểm tra count_params loại trừ cả tok_emb và pos_emb
    params = adapter.count_params()
    assert params["total_params"] > params["non_embedding_params"] > 0


def test_gpt_adapter_from_checkpoint_and_hash_check(tmp_path):
    ds = _make_dataset(tmp_path, vocab_size=50, n_val=200)
    gpt = DummyGPT(vocab_size=ds.vocab_size, n_embd=32, block_size=128, uniform_logits=False)
    ckpt_path = str(tmp_path / "dummy_gpt.pt")
    torch.save(
        {
            "state_dict": gpt.state_dict(),
            "config": {"vocab_size": ds.vocab_size, "n_embd": 32, "block_size": 128},
            "tokenizer_meta": {
                "version": ds.version,
                "vocab_size": ds.vocab_size,
                "tokenizer_hash": ds.tokenizer_hash,
            },
            "step": 100,
            "val_loss": 3.5,
            "train_tokens": 50000,
            "train_config": {"lr": 3e-4, "ctx": 128},
        },
        ckpt_path,
    )

    def builder(cfg):
        return DummyGPT(vocab_size=cfg["vocab_size"], n_embd=cfg["n_embd"], block_size=cfg["block_size"])

    adapter = GPTAdapter.from_checkpoint(ckpt_path, model_builder=builder, name="dummy_gpt", device="cpu")
    assert adapter.tokenizer_hash == ds.tokenizer_hash
    assert adapter.train_tokens == 50000

    res = evaluate_model(adapter, ds, split="val", ctx=64, max_tokens=None, save_dir=None)
    assert res.train_tokens == 50000
    assert res.train_config["lr"] == 3e-4

    # Lệch tokenizer_hash phải raise ValueError
    adapter.tokenizer_hash = "mismatched_hash"
    with pytest.raises(ValueError, match="Lệch tokenizer_hash"):
        evaluate_model(adapter, ds, split="val", ctx=64, max_tokens=None, save_dir=None)


def test_gpt_adapter_with_real_tv2_gpt_and_checkpoint(tmp_path):
    """Kiểm thử tích hợp trực tiếp GPTAdapter với GPT thật và save_checkpoint của TV2."""
    from src.model import GPT, make_config, estimate_params
    from src.train.checkpoint import save_checkpoint, gpt_model_builder

    ds = _make_dataset(tmp_path, vocab_size=64, n_val=200)
    cfg = make_config("tiny_debug", vocab_size=ds.vocab_size, block_size=64)
    gpt = GPT(cfg)

    ckpt_path = tmp_path / "tv2_tiny.pt"
    save_checkpoint(
        ckpt_path,
        model=gpt,
        tokenizer_meta={
            "version": ds.version,
            "vocab_size": ds.vocab_size,
            "tokenizer_hash": ds.tokenizer_hash,
        },
        step=10,
        val_loss=4.1,
        train_args={"batch_size": 4, "grad_accum_steps": 2, "block_size": 64, "max_lr": 6e-4},
    )

    adapter = GPTAdapter.from_checkpoint(str(ckpt_path), model_builder=gpt_model_builder, name="tiny_gpt", device="cpu")
    expected_params = estimate_params(cfg)
    counted = adapter.count_params()
    assert counted["total_params"] == expected_params["total"]
    assert counted["non_embedding_params"] == expected_params["non_embedding"]
    assert adapter.train_tokens == 10 * 4 * 2 * 64

    res = evaluate_model(adapter, ds, split="val", ctx=64, max_tokens=None, save_dir=None)
    assert res.tag == "official"
    assert res.num_tokens == 199
    assert res.perplexity > 0
    assert res.train_tokens == 10 * 4 * 2 * 64
