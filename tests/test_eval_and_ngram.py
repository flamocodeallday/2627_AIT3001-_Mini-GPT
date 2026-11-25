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


def test_v0_byte_dataset_and_uniform_safety():
    ds = DatasetBundle("data/processed/v0_byte")
    assert ds.version == "v0_byte"
    assert ds.vocab_size == 257

    model = UniformLanguageModel("uniform_check", ds.vocab_size, ds.tokenizer_hash)
    assert math.isclose(float(model.next_token_probs().sum()), 1.0, rel_tol=1e-9)

    res = evaluate_model(model, ds, split="val", ctx=256, max_tokens=10000, save_dir=None)
    assert res.tag == "debug"
    assert math.isclose(res.perplexity, 257.0, rel_tol=1e-5)


def test_tokenizer_hash_mismatch_raises():
    ds = DatasetBundle("data/processed/v0_byte")
    bad_model = UniformLanguageModel("bad_hash", ds.vocab_size, "wrong_hash_000")
    with pytest.raises(ValueError, match="Lệch tokenizer_hash"):
        evaluate_model(bad_model, ds, split="val", ctx=256, max_tokens=1000, save_dir=None)


def test_ngram_kneser_ney_on_v0_byte():
    ds = DatasetBundle("data/processed/v0_byte")
    train_tokens = ds.get_split_tokens("train")

    # Huấn luyện 3-gram trên 200,000 token đầu để kiểm thử nhanh
    ngram = NGramLM(order=3, vocab_size=ds.vocab_size, tokenizer_hash=ds.tokenizer_hash)
    ngram.fit(train_tokens, max_train_tokens=200_000)

    # 1. Tổng xác suất phân phối tiếp theo phải bằng 1.0
    probs = ngram.next_token_probs([72, 224])  # Ví dụ vài byte tiếng Việt
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
    assert np.all(probs > 0.0)

    # 2. Đánh giá trên 20,000 token đầu tập val: PPL phải tốt hơn nhiều so với 257.0
    res = evaluate_model(ngram, ds, split="val", ctx=256, max_tokens=20_000, save_dir="results")
    print(f"\n[DEBUG N-gram n=3] PPL = {res.perplexity:.2f} | BPC = {res.bpc:.4f}")
    assert res.tag == "debug"
    assert res.perplexity < 50.0