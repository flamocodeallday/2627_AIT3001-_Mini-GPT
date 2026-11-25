import math
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


def test_lstm_lm_forward_and_eval_contract():
    ds = DatasetBundle("data/processed/v0_byte")
    cfg = LSTMConfig(vocab_size=ds.vocab_size, embed_dim=64, hidden_dim=64, num_layers=2)
    model = LSTMLM(cfg, tokenizer_hash=ds.tokenizer_hash, device="cpu")

    # 1. Kiểm tra loss khởi tạo ban đầu xấp xỉ ln(vocab_size)
    x = torch.randint(0, ds.vocab_size, (2, 32))
    y = torch.randint(0, ds.vocab_size, (2, 32))
    logits, loss, _ = model(x, y)
    assert logits.shape == (2, 32, ds.vocab_size)
    expected_init_loss = math.log(ds.vocab_size)
    assert abs(loss.item() - expected_init_loss) < 0.8

    # 2. Kiểm tra chạy qua hàm evaluate_model chuẩn
    res = evaluate_model(model, ds, split="val", ctx=64, max_tokens=2000, save_dir=None)
    assert res.tag == "debug"
    assert res.perplexity > 0