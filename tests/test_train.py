"""Auto-grader for member 2's training exercises (bài 9-11) + checks for the training scaffold.

    pytest tests/test_train.py -k get_lr
    pytest tests/test_train.py -k split_decay
    pytest tests/test_train.py -k train_step
    pytest tests/test_train.py -k scaffold      # must pass from day one (data/checkpoint/budget)
"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from src.model.config import GPTConfig, estimate_params, make_config
from src.model.gpt import GPT
from src.train.checkpoint import build_model_from_checkpoint, gpt_model_builder, load_checkpoint, save_checkpoint
from src.train.data import TokenBatcher, load_meta, tokenizer_meta_from, verify_tokenizer_hash
from src.train.optim import split_decay_params
from src.train.schedule import get_lr
from src.train.train import compute_budget, train_step

V0 = Path("data/processed/v0_byte")
needs_v0 = pytest.mark.skipif(not (V0 / "train.bin").exists(), reason="data/processed/v0_byte not present")


def small_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=257, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# ----------------------------------------------------------------------------- bài 9: get_lr
LR = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, total_steps=100)


def test_get_lr_warmup() -> None:
    assert 0 < get_lr(0, **LR) <= 1e-3 / 10 + 1e-12, "step 0: nhỏ nhưng khác 0"
    assert get_lr(9, **LR) == pytest.approx(1e-3), "bước warmup cuối phải đạt max_lr"
    lrs = [get_lr(s, **LR) for s in range(10)]
    assert all(a < b for a, b in zip(lrs, lrs[1:])), "warmup phải tăng dần"


def test_get_lr_cosine_and_after() -> None:
    assert get_lr(10, **LR) == pytest.approx(1e-3)
    assert get_lr(55, **LR) == pytest.approx((1e-3 + 1e-4) / 2), "giữa pha cosine phải là trung bình max/min"
    assert get_lr(100, **LR) == pytest.approx(1e-4)
    assert get_lr(150, **LR) == pytest.approx(1e-4)
    lrs = [get_lr(s, **LR) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "pha cosine phải giảm đơn điệu"


# ----------------------------------------------------------------------------- bài 10: split_decay_params
@pytest.mark.parametrize("bias", [False, True])
def test_split_decay_params_partition(bias: bool) -> None:
    model = GPT(small_cfg(bias=bias))
    decay, no_decay = split_decay_params(model)
    ids_d, ids_n = {id(p) for p in decay}, {id(p) for p in no_decay}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids_d) == len(decay) and len(ids_n) == len(no_decay), "có tham số bị lặp (weight tying?)"
    assert not ids_d & ids_n, "một tham số nằm ở cả hai nhóm"
    assert ids_d | ids_n == all_ids, "thiếu tham số"
    assert all(p.dim() >= 2 for p in decay) and all(p.dim() < 2 for p in no_decay)
    assert id(model.tok_emb.weight) in ids_d


# ----------------------------------------------------------------------------- bài 11: train_step
def _fixed_batches(x: torch.Tensor, y: torch.Tensor, n_chunks: int):
    xs, ys = x.chunk(n_chunks), y.chunk(n_chunks)
    it = iter(zip(xs, ys))
    return lambda: next(it)


def _run_step(model, x, y, accum, clip):
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return train_step(model, opt, scaler, _fixed_batches(x, y, accum), accum, clip, contextlib.nullcontext())


def test_train_step_grad_accumulation_equals_full_batch() -> None:
    torch.manual_seed(0)
    base = GPT(small_cfg())
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    x = torch.randint(0, 257, (8, 16))
    y = torch.randint(0, 257, (8, 16))
    sa = _run_step(a, x, y, accum=1, clip=1e9)
    sb = _run_step(b, x, y, accum=4, clip=1e9)
    assert sa["loss"] == pytest.approx(sb["loss"], rel=1e-5), "loss trả về phải là trung bình qua micro-batch"
    assert sa["grad_norm"] == pytest.approx(sb["grad_norm"], rel=1e-4)
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), f"{n}: cập nhật với accum=4 khác accum=1 (chia loss đúng chưa?)"


def test_train_step_clips_and_reports_preclip_norm() -> None:
    torch.manual_seed(0)
    model = GPT(small_cfg())
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randint(0, 257, (4, 16))
    y = torch.randint(0, 257, (4, 16))
    stats = _run_step(model, x, y, accum=1, clip=0.01)
    assert isinstance(stats["loss"], float) and isinstance(stats["grad_norm"], float)
    assert stats["grad_norm"] > 0.01, "grad_norm phải là norm TRƯỚC khi clip"
    delta = torch.sqrt(sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), before)))
    assert delta.item() <= 0.01 * (1 + 1e-3), f"với SGD lr=1, ||Δθ|| = ||grad đã clip|| phải ≤ 0.01, nhận {delta.item():.4f}"


def test_train_step_clears_previous_gradients() -> None:
    torch.manual_seed(0)
    base = GPT(small_cfg())
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    for p in b.parameters():  # stale garbage gradients that must be discarded
        p.grad = torch.full_like(p, 100.0)
    x = torch.randint(0, 257, (4, 16))
    y = torch.randint(0, 257, (4, 16))
    _run_step(a, x, y, accum=1, clip=1e9)
    _run_step(b, x, y, accum=1, clip=1e9)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), "gradient cũ chưa được xóa trước khi backward"


# ----------------------------------------------------------------------------- scaffold
def test_scaffold_budget_follows_decision_c() -> None:
    model_cfg = make_config("main_30m", vocab_size=8192, block_size=512)
    cfg = {"train": {"batch_size": 32, "grad_accum_steps": 4, "tokens_per_param": 20, "warmup_frac": 0.03}}
    b = compute_budget(cfg, model_cfg)
    n = estimate_params(model_cfg)["non_embedding"]
    assert b["tokens_per_step"] == 32 * 4 * 512
    assert abs(b["total_tokens"] - 20 * n) < b["tokens_per_step"]
    assert b["warmup_steps"] == int(0.03 * b["total_steps"])
    assert compute_budget({"train": {"batch_size": 2, "max_steps": 7}}, model_cfg)["total_steps"] == 7


@needs_v0
def test_scaffold_batcher_shapes_shift_and_determinism() -> None:
    meta = load_meta(V0)
    bat = TokenBatcher(V0, "train", block_size=32, batch_size=4, seed=5)
    x, y = bat.next_batch()
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1]), "y phải là x dịch 1"
    assert int(x.max()) < meta["vocab_size"]
    bat.reset()
    x2, _ = bat.next_batch()
    assert torch.equal(x, x2)
    with pytest.raises(ValueError):
        TokenBatcher(V0, "test", 32, 4)


@needs_v0
def test_scaffold_tokenizer_hash_check(tmp_path: Path) -> None:
    meta = load_meta(V0)
    assert verify_tokenizer_hash(V0, meta) == meta["tokenizer_hash"]
    bad = tmp_path / "v0_bad"
    bad.mkdir()
    shutil.copy(V0 / "tokenizer.json", bad / "tokenizer.json")
    (bad / "meta.json").write_text(json.dumps({**meta, "tokenizer_hash": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_tokenizer_hash(bad)


def test_scaffold_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = GPT(small_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for p in model.parameters():  # fake one step so the optimizer has state
        p.grad = torch.randn_like(p)
    opt.step()
    tok_meta = {"version": "v0_byte", "tokenizer_type": "byte_level", "vocab_size": 257, "eos_id": 256, "tokenizer_hash": "abc"}
    path = save_checkpoint(tmp_path / "ck.pt", model=model, tokenizer_meta=tok_meta, step=12, val_loss=3.5, optimizer=opt)
    ckpt = load_checkpoint(path)
    for key in ("state_dict", "config", "tokenizer_meta", "step", "val_loss"):
        assert key in ckpt, f"thiếu key hợp đồng {key}"
    assert ckpt["step"] == 12 and ckpt["tokenizer_meta"]["tokenizer_hash"] == "abc"
    rebuilt = build_model_from_checkpoint(ckpt)
    for (n, a), b in zip(model.state_dict().items(), rebuilt.state_dict().values()):
        assert torch.equal(a, b), n
    assert rebuilt.lm_head.weight is rebuilt.tok_emb.weight, "weight tying phải còn sau khi load"
    assert isinstance(gpt_model_builder(ckpt["config"]), GPT)
    opt2 = torch.optim.AdamW(rebuilt.parameters(), lr=1e-3)
    opt2.load_state_dict(ckpt["optimizer"])
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_scaffold_checkpoint_loads_in_member3_adapter(tmp_path: Path) -> None:
    """Runs only after member 3's src/eval is merged into this branch."""
    adapter_mod = pytest.importorskip("src.eval.gpt_adapter")
    model = GPT(small_cfg())
    tok_meta = {"version": "v0_byte", "tokenizer_type": "byte_level", "vocab_size": 257, "eos_id": 256, "tokenizer_hash": "abc"}
    path = save_checkpoint(tmp_path / "ck.pt", model=model, tokenizer_meta=tok_meta, step=0, val_loss=None)
    adapter = adapter_mod.GPTAdapter.from_checkpoint(str(path), model_builder=gpt_model_builder, device="cpu")
    assert adapter.vocab_size == 257 and adapter.tokenizer_hash == "abc"
