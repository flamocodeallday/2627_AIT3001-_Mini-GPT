import os
import json
import math
import pytest
import torch
import torch.nn as nn
import numpy as np
from src.eval.base import DatasetBundle, evaluate_model
from src.eval.lstm import CustomLSTMCell, LSTMLM, LSTMConfig


def test_custom_lstm_cell_matches_pytorch():
    """Đối chiếu CustomLSTMCell tự viết với torch.nn.LSTMCell (sai số < 1e-5)."""
    torch.manual_seed(42)
    B, D_in, H = 4, 16, 32
    custom_cell = CustomLSTMCell(D_in, H)
    ref_cell = nn.LSTMCell(D_in, H)

    # Sao chép trọng số từ CustomLSTMCell sang torch.nn.LSTMCell
    with torch.no_grad():
        ref_cell.weight_ih.copy_(custom_cell.weight_ih.weight)
        ref_cell.bias_ih.copy_(custom_cell.weight_ih.bias)
        ref_cell.weight_hh.copy_(custom_cell.weight_hh.weight)
        ref_cell.bias_hh.zero_()

    x = torch.randn(B, D_in)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    h_custom, c_custom = custom_cell(x, (h0, c0))
    h_ref, c_ref = ref_cell(x, (h0, c0))

    assert torch.allclose(h_custom, h_ref, atol=1e-5)
    assert torch.allclose(c_custom, c_ref, atol=1e-5)


def test_lstm_lm_forward_and_eval_contract(tmp_path):
    vocab_size = 64
    rng = np.random.default_rng(7)
    val_arr = rng.integers(0, vocab_size, size=300, dtype=np.uint16)
    data_dir = tmp_path / "v0_byte"
    data_dir.mkdir(parents=True, exist_ok=True)
    val_arr.tofile(data_dir / "val.bin")
    meta = {
        "version": "v0_byte",
        "tokenizer_type": "byte",
        "vocab_size": vocab_size,
        "eos_id": 0,
        "seed": 42,
        "tokenizer_hash": "hash_lstm_test",
        "splits": {"val": {"n_tokens": 300, "n_chars": 300}},
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    ds = DatasetBundle(str(data_dir))
    cfg = LSTMConfig(vocab_size=ds.vocab_size, embed_dim=32, hidden_dim=32, num_layers=2)
    model = LSTMLM(cfg, tokenizer_hash=ds.tokenizer_hash, device="cpu")

    # 1. Kiểm tra loss khởi tạo ban đầu xấp xỉ ln(vocab_size)
    x = torch.randint(0, ds.vocab_size, (2, 32))
    y = torch.randint(0, ds.vocab_size, (2, 32))
    logits, loss, _ = model(x, y)
    assert logits.shape == (2, 32, ds.vocab_size)
    expected_init_loss = math.log(ds.vocab_size)
    assert abs(loss.item() - expected_init_loss) < 0.8

    # 2. Kiểm tra chạy qua hàm evaluate_model chuẩn
    res = evaluate_model(model, ds, split="val", ctx=64, max_tokens=None, save_dir=None)
    assert res.tag == "debug"
    assert res.perplexity > 0

    # 3. Kiểm tra lưu & đọc checkpoint kèm train_tokens và train_config
    ckpt_path = str(tmp_path / "lstm_test.pt")
    model.save_checkpoint(
        path=ckpt_path,
        tokenizer_meta={"version": ds.version, "tokenizer_hash": ds.tokenizer_hash, "vocab_size": ds.vocab_size},
        step=10,
        val_loss=res.avg_nll,
        train_tokens=12345,
        train_config={"lr": 3e-3, "seed": 42},
    )
    loaded = LSTMLM.from_checkpoint(ckpt_path, device="cpu")
    assert loaded.train_tokens == 12345
    assert loaded.train_config["lr"] == 3e-3


def test_lstm_on_v0_byte_if_present():
    v0_dir = "data/processed/v0_byte"
    if not os.path.exists(os.path.join(v0_dir, "meta.json")):
        pytest.skip("Thiếu data/processed/v0_byte trên clone sạch")

    ds = DatasetBundle(v0_dir)
    cfg = LSTMConfig(vocab_size=ds.vocab_size, embed_dim=64, hidden_dim=64, num_layers=2)
    model = LSTMLM(cfg, tokenizer_hash=ds.tokenizer_hash, device="cpu")
    res = evaluate_model(model, ds, split="val", ctx=64, max_tokens=2000, save_dir=None)
    assert res.tag == "debug"
    assert res.perplexity > 0
