"""Auto-grader for member 2's model exercises (bài 1-8). Run one exercise at a time, e.g.

    pytest tests/test_gpt.py -k mlp
    pytest tests/test_gpt.py -k attention
    pytest tests/test_gpt.py -k block
    pytest tests/test_gpt.py -k forward
    pytest tests/test_gpt.py -k count_params
    pytest tests/test_gpt.py -k "top_k or top_p"
    pytest tests/test_gpt.py -k sample
    pytest tests/test_gpt.py -k generate

Tests named `test_scaffold_*` check the AI-written parts and must pass from day one.
`F.scaled_dot_product_attention` appears here ONLY as a reference to compare against.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.config import PRESETS, GPTConfig, estimate_params, make_config
from src.model.gpt import GPT, MLP, Block, CausalSelfAttention
from src.model.sampling import filter_top_k, filter_top_p, sample_next_token


def small_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=257, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# ----------------------------------------------------------------------------- scaffold
def test_scaffold_config_roundtrip_and_validation() -> None:
    cfg = make_config("main_30m", vocab_size=8192, block_size=512)
    assert GPTConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.head_dim == 64
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, n_embd=30, n_head=4)
    with pytest.raises(NotImplementedError):
        GPTConfig(vocab_size=10, pos_type="rope")


def test_scaffold_presets_have_head_dim_64_and_expected_main_size() -> None:
    for name, shape in PRESETS.items():
        assert shape["n_embd"] // shape["n_head"] == 64, name
    est = estimate_params(make_config("main_30m", vocab_size=8192, block_size=512))
    assert 24_000_000 < est["non_embedding"] < 26_000_000
    assert 29_000_000 < est["total"] < 30_500_000


def test_scaffold_weight_tying_and_init() -> None:
    torch.manual_seed(0)
    model = GPT(small_cfg())
    assert model.lm_head.weight is model.tok_emb.weight
    assert abs(model.tok_emb.weight.std().item() - 0.02) < 0.005
    untied = GPT(small_cfg(tie_weights=False))
    assert untied.lm_head.weight is not untied.tok_emb.weight


def test_scaffold_mask_not_in_state_dict() -> None:
    sd = GPT(small_cfg()).state_dict()
    assert not any("causal_mask" in k for k in sd)


# ----------------------------------------------------------------------------- bài 1: MLP
def test_mlp_shape_and_position_wise() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    mlp = MLP(cfg).eval()
    x = torch.randn(3, 7, cfg.n_embd)
    out = mlp(x)
    assert out.shape == (3, 7, cfg.n_embd), f"kỳ vọng (3, 7, {cfg.n_embd}), nhận {tuple(out.shape)}"
    # position-wise: changing position 0 must not change position 1
    x2 = x.clone()
    x2[:, 0] += 1.0
    assert torch.allclose(mlp(x2)[:, 1:], out[:, 1:]), "MLP phải xử lý từng vị trí độc lập"


def test_mlp_uses_both_linears_and_nonlinearity() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    mlp = MLP(cfg).eval()
    x = torch.randn(2, 5, cfg.n_embd)
    out = mlp(x)
    out.sum().backward()
    assert mlp.fc_in.weight.grad is not None and mlp.fc_out.weight.grad is not None
    # a purely linear map would satisfy f(2x) == 2 f(x)
    assert not torch.allclose(mlp(2 * x), 2 * out, atol=1e-4), "thiếu hàm kích hoạt phi tuyến?"


# ----------------------------------------------------------------------------- bài 2: attention
def _reference_attention(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    H = attn.n_head
    q, k, v = attn.qkv_proj(x).split(C, dim=-1)
    q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.out_proj(y.transpose(1, 2).reshape(B, T, C))


@pytest.mark.parametrize("T", [16, 5, 1])
def test_attention_matches_reference(T: int) -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, T, cfg.n_embd)
    out = attn(x)
    assert out.shape == (2, T, cfg.n_embd), f"kỳ vọng {(2, T, cfg.n_embd)}, nhận {tuple(out.shape)}"
    ref = _reference_attention(attn, x)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"lệch so với attention tham chiếu: max abs err = {err:.2e} (kiểm tra scale, mask, thứ tự q/k/v, cách tách head)"


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 10, cfg.n_embd)
    out = attn(x)
    t = 4
    x_future = x.clone()
    x_future[:, t + 1 :] = torch.randn_like(x_future[:, t + 1 :])
    out_future = attn(x_future)
    assert torch.allclose(out[:, : t + 1], out_future[:, : t + 1], atol=1e-6), "vị trí <= t đang nhìn thấy token tương lai"
    assert not torch.allclose(out[:, t + 1 :], out_future[:, t + 1 :]), "output tương lai phải đổi khi input tương lai đổi"


def test_attention_dropout_only_in_train_mode() -> None:
    torch.manual_seed(0)
    cfg = small_cfg(dropout=0.5)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    attn.eval()
    assert torch.allclose(attn(x), attn(x)), "eval mode phải tất định"
    attn.train()
    assert not torch.allclose(attn(x), attn(x)), "train mode với dropout > 0 phải ngẫu nhiên"


# ----------------------------------------------------------------------------- bài 3: block
def test_block_shape_and_gradients() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    block = Block(cfg)
    x = torch.randn(2, 8, cfg.n_embd, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    for name in ("norm_attn.weight", "attn.qkv_proj.weight", "norm_mlp.weight", "mlp.fc_in.weight"):
        assert dict(block.named_parameters())[name].grad is not None, f"{name} không nhận gradient: sub-layer chưa được dùng?"


def test_block_residual_is_identity_when_sublayers_output_zero() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    block = Block(cfg).eval()
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.mlp.fc_out.weight.zero_()
    x = torch.randn(2, 8, cfg.n_embd) * 3 + 1
    assert torch.allclose(block(x), x, atol=1e-6), "khi hai sub-layer trả 0, block phải trả đúng x (residual, pre-norm)"


# ----------------------------------------------------------------------------- bài 4: GPT.forward
def test_forward_shapes_and_loss_none() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 11))
    logits, loss = model(idx)
    assert logits.shape == (3, 11, cfg.vocab_size), f"kỳ vọng logits (3, 11, V), nhận {tuple(logits.shape)}"
    assert loss is None


@pytest.mark.parametrize("vocab", [257, 8192])
def test_forward_initial_loss_close_to_ln_vocab(vocab: int) -> None:
    torch.manual_seed(0)
    cfg = small_cfg(vocab_size=vocab, n_embd=64, n_head=4)
    model = GPT(cfg)
    idx = torch.randint(0, vocab, (4, 16))
    tgt = torch.randint(0, vocab, (4, 16))
    logits, loss = model(idx, tgt)
    assert loss is not None and loss.ndim == 0
    assert abs(loss.item() - math.log(vocab)) < 0.3, f"loss ban đầu {loss.item():.3f}, kỳ vọng ≈ ln({vocab}) = {math.log(vocab):.3f}"
    ref = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-5), "loss phải là cross-entropy trung bình trên mọi vị trí"


def test_forward_rejects_too_long_input() -> None:
    cfg = small_cfg()
    with pytest.raises(ValueError):
        GPT(cfg)(torch.zeros(1, cfg.block_size + 1, dtype=torch.long))


def test_forward_uses_positions() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    model = GPT(cfg).eval()
    idx = torch.full((1, 6), 42, dtype=torch.long)  # same token at every position
    logits, _ = model(idx)
    assert not torch.allclose(logits[0, 0], logits[0, 5]), "cùng một token ở các vị trí khác nhau phải cho logits khác nhau"


def test_forward_is_causal_end_to_end() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    idx2 = idx.clone()
    idx2[0, 7:] = (idx2[0, 7:] + 1) % cfg.vocab_size
    a, _ = model(idx)
    b, _ = model(idx2)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-5)


def test_forward_overfits_one_batch() -> None:
    torch.manual_seed(0)
    cfg = small_cfg(n_embd=64, n_head=4)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    tgt = torch.randint(0, cfg.vocab_size, (4, 16))
    for _ in range(200):
        _, loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5, f"không overfit nổi 1 batch: loss cuối = {loss.item():.3f}"


# ----------------------------------------------------------------------------- bài 5: count_params
@pytest.mark.parametrize("tie,bias", [(True, False), (False, False), (True, True)])
def test_count_params_matches_formula(tie: bool, bias: bool) -> None:
    cfg = small_cfg(tie_weights=tie, bias=bias)
    model = GPT(cfg)
    est = estimate_params(cfg)
    unique = {id(p): p for p in model.parameters() if p.requires_grad}
    assert model.count_params(exclude_embeddings=False) == est["total"] == sum(p.numel() for p in unique.values())
    assert model.count_params(exclude_embeddings=True) == est["non_embedding"]


# ----------------------------------------------------------------------------- bài 6: top-k / top-p
def test_top_k_keeps_exactly_k_largest() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0], [0.0, -1.0, 2.0, 1.0, 3.0]])
    out = filter_top_k(logits, 2)
    assert out.shape == logits.shape
    assert torch.equal(torch.isfinite(out), torch.tensor([[False, True, False, True, False], [False, False, True, False, True]]))
    assert torch.equal(out[torch.isfinite(out)], logits[torch.isfinite(out)]), "giá trị được giữ phải giữ nguyên"
    assert torch.equal(logits, torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0], [0.0, -1.0, 2.0, 1.0, 3.0]])), "không được sửa input in-place"


def test_top_k_larger_than_vocab_is_noop() -> None:
    logits = torch.randn(2, 6)
    assert torch.equal(filter_top_k(logits, 100), logits)


def test_top_p_nucleus() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    kept = torch.isfinite(filter_top_p(logits, 0.7))
    assert kept.tolist() == [[True, True, False, False]], f"p=0.7 phải giữ 2 token đầu, nhận {kept.tolist()}"
    kept = torch.isfinite(filter_top_p(logits, 0.85))
    assert kept.tolist() == [[True, True, True, False]]
    kept = torch.isfinite(filter_top_p(logits, 0.01))
    assert kept.tolist() == [[True, False, False, False]], "luôn giữ ít nhất token xác suất cao nhất"
    assert torch.isfinite(filter_top_p(logits, 1.0)).all()


def test_top_p_works_on_unsorted_rows() -> None:
    probs = torch.tensor([[0.05, 0.5, 0.15, 0.3], [0.25, 0.25, 0.25, 0.25]])
    kept = torch.isfinite(filter_top_p(probs.log(), 0.7))
    assert kept[0].tolist() == [False, True, False, True]
    assert kept[1].sum().item() == 3


# ----------------------------------------------------------------------------- bài 7: sample_next_token
def test_sample_greedy_and_shape() -> None:
    logits = torch.randn(5, 50)
    out = sample_next_token(logits, temperature=0.0)
    assert out.shape == (5, 1) and out.dtype == torch.long
    assert torch.equal(out.squeeze(1), logits.argmax(-1))
    assert torch.equal(sample_next_token(logits, temperature=1.0, top_k=1).squeeze(1), logits.argmax(-1))
    assert torch.equal(sample_next_token(logits, temperature=1.0, top_p=1e-6).squeeze(1), logits.argmax(-1))


def test_sample_respects_top_k() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 30)
    allowed = set(logits.topk(3).indices[0].tolist())
    gen = torch.Generator().manual_seed(1)
    draws = {int(sample_next_token(logits, 1.0, top_k=3, generator=gen)) for _ in range(300)}
    assert draws <= allowed and len(draws) > 1


def test_sample_distribution_matches_softmax_and_temperature() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    gen = torch.Generator().manual_seed(0)
    n = 6000
    counts = torch.zeros(3)
    for _ in range(n):
        counts[int(sample_next_token(logits, 1.0, generator=gen))] += 1
    assert torch.allclose(counts / n, F.softmax(logits, -1)[0], atol=0.03), "tần suất mẫu phải ≈ softmax(logits)"
    counts_hot = torch.zeros(3)
    for _ in range(n):
        counts_hot[int(sample_next_token(logits, 5.0, generator=gen))] += 1
    assert counts_hot[2] > counts[2] * 1.5, "temperature cao phải làm phân phối phẳng hơn"


def test_sample_is_reproducible_with_generator() -> None:
    logits = torch.randn(4, 100)
    a = sample_next_token(logits, 0.9, top_k=20, top_p=0.9, generator=torch.Generator().manual_seed(7))
    b = sample_next_token(logits, 0.9, top_k=20, top_p=0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------- bài 8: generate
def test_generate_shapes_and_range() -> None:
    torch.manual_seed(0)
    cfg = small_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (2, 3))
    out = model.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=10)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :3], prompt), "phần đầu phải là prompt"
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_size


def test_generate_prompt_longer_than_block_size() -> None:
    cfg = small_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 5))
    out = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, cfg.block_size + 9)


def test_generate_deterministic_and_restores_mode() -> None:
    cfg = small_cfg(dropout=0.3)
    model = GPT(cfg).train()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = model.generate(prompt, 6, temperature=0.8, generator=torch.Generator().manual_seed(3))
    b = model.generate(prompt, 6, temperature=0.8, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "cùng generator seed phải ra cùng kết quả (dropout phải tắt khi sinh)"
    assert model.training, "generate phải trả model về mode cũ"
