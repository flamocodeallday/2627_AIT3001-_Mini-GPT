import os
import json
import math
import pytest
import numpy as np
from src.eval.base import DatasetBundle, BaseLanguageModel, evaluate_model
from src.eval.ngram import NGramLM


class UniformLanguageModel(BaseLanguageModel):
    """Mô hình kiểm thử an toàn: dự đoán phân phối đều P(x) = 1 / vocab_size."""

    def next_token_probs(self) -> np.ndarray:
        return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)

    def nll(self, token_ids: np.ndarray, ctx: int):
        num_eval = len(token_ids) - 1
        total_nll = num_eval * math.log(self.vocab_size)
        return float(total_nll), int(num_eval)


def _create_synthetic_dataset(tmp_path, version="v1_synthetic", vocab_size=32, n_train=2000, n_val=400):
    rng = np.random.default_rng(42)
    train_arr = rng.integers(0, vocab_size, size=n_train, dtype=np.uint16)
    val_arr = rng.integers(0, vocab_size, size=n_val, dtype=np.uint16)

    data_dir = tmp_path / version
    data_dir.mkdir(parents=True, exist_ok=True)
    train_arr.tofile(data_dir / "train.bin")
    val_arr.tofile(data_dir / "val.bin")

    meta = {
        "version": version,
        "tokenizer_type": "synthetic",
        "vocab_size": vocab_size,
        "eos_id": 0,
        "seed": 42,
        "tokenizer_hash": "synthetic_hash_abcdef123456",
        "splits": {
            "train": {"n_tokens": n_train, "n_chars": n_train * 3},
            "val": {"n_tokens": n_val, "n_chars": n_val * 3},
        },
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return DatasetBundle(str(data_dir))


def test_synthetic_dataset_uniform_and_metadata(tmp_path):
    ds = _create_synthetic_dataset(tmp_path, version="v1_synthetic", vocab_size=32)
    model = UniformLanguageModel("uniform_check", ds.vocab_size, ds.tokenizer_hash)
    assert math.isclose(float(model.next_token_probs().sum()), 1.0, rel_tol=1e-9)

    # Full split -> official
    res_full = evaluate_model(model, ds, split="val", ctx=64, max_tokens=None, save_dir=None)
    assert res_full.tag == "official"
    assert res_full.is_full_split is True
    assert math.isclose(res_full.perplexity, 32.0, rel_tol=1e-5)
    assert isinstance(res_full.git_commit, str)
    assert isinstance(res_full.git_dirty, bool)
    assert isinstance(res_full.hardware, str)

    # Subset -> debug
    res_sub = evaluate_model(model, ds, split="val", ctx=64, max_tokens=100, save_dir=None)
    assert res_sub.tag == "debug"
    assert res_sub.is_full_split is False


def test_tokenizer_hash_mismatch_raises(tmp_path):
    ds = _create_synthetic_dataset(tmp_path)
    bad_model = UniformLanguageModel("bad_hash", ds.vocab_size, "wrong_hash_000")
    with pytest.raises(ValueError, match="Lệch tokenizer_hash"):
        evaluate_model(bad_model, ds, split="val", ctx=64, max_tokens=100, save_dir=None)


@pytest.mark.parametrize("order", [3, 4])
def test_ngram_kn_prob_sums_to_one_without_normalization(order):
    """Kiểm tra trực tiếp tổng _kn_prob trên toàn vocab ≈ 1.0 cho nhiều loại ngữ cảnh."""
    vocab_size = 24
    rng = np.random.default_rng(123)
    # Chỉ dùng token 0..19 trong train để token 20..23 là OOV chưa từng gặp
    train_tokens = rng.integers(0, 20, size=1500, dtype=np.uint16)

    ngram = NGramLM(order=order, vocab_size=vocab_size)
    ngram.fit(train_tokens)
    assert ngram.train_tokens == 1500

    # 1. Ngữ cảnh đủ dài đã xuất hiện trong train
    seen_full = tuple(train_tokens[10 : 10 + order - 1].tolist())
    # 2. Ngữ cảnh ngắn hơn bậc n (1 token và rỗng)
    short_1 = (int(train_tokens[10]),)
    empty_hist = ()
    # 3. Ngữ cảnh chưa từng gặp (chứa token OOV)
    unseen_full = tuple([21] * (order - 1))
    unseen_mixed = (int(train_tokens[10]), 22) if order >= 3 else (22,)

    for hist in [seen_full, short_1, empty_hist, unseen_full, unseen_mixed]:
        effective_level = min(len(hist) + 1, order)
        is_highest = effective_level == order
        probs = [
            ngram._kn_prob(hist, w, level=effective_level, is_highest=is_highest)
            for w in range(vocab_size)
        ]
        assert all(p > 0.0 for p in probs), f"Có xác suất <= 0 với hist={hist}"
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-6), (
            f"Tổng _kn_prob != 1.0 ({sum(probs)}) cho order={order}, hist={hist}"
        )


def test_v0_byte_dataset_and_ngram_if_present():
    v0_dir = "data/processed/v0_byte"
    if not os.path.exists(os.path.join(v0_dir, "meta.json")):
        pytest.skip("Thiếu data/processed/v0_byte trên clone sạch")

    ds = DatasetBundle(v0_dir)
    assert ds.version == "v0_byte"
    assert ds.vocab_size == 257

    train_tokens = ds.get_split_tokens("train")
    ngram = NGramLM(order=3, vocab_size=ds.vocab_size, tokenizer_hash=ds.tokenizer_hash)
    ngram.fit(train_tokens, max_train_tokens=200_000)

    probs = ngram.next_token_probs([72, 224])
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
    assert np.all(probs > 0.0)

    res = evaluate_model(ngram, ds, split="val", ctx=512, max_tokens=20_000, save_dir=None)
    assert res.tag == "debug"
    assert res.perplexity < 50.0
