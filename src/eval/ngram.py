import math
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional
import numpy as np
from .base import BaseLanguageModel


class NGramLM(BaseLanguageModel):
    """
    Mô hình ngôn ngữ N-gram tự cài đặt hoàn toàn với Interpolated Kneser-Ney Smoothing
    (theo Jurafsky & Martin, SLP3, Chương 3 & Phụ lục C).
    """

    def __init__(
        self,
        order: int = 3,
        vocab_size: int = 257,
        discount: float = 0.75,
        tokenizer_hash: Optional[str] = None,
        name: Optional[str] = None,
    ):
        if order < 1:
            raise ValueError("Bậc n-gram (order) phải >= 1")
        super().__init__(
            name=name or f"ngram_kn_n{order}",
            vocab_size=vocab_size,
            tokenizer_hash=tokenizer_hash,
        )
        self.order = order
        self.discount = discount

        # Lưu đếm cho từng bậc k = 1 .. order
        # ngram_counts[k][tuple_of_k_tokens] = count
        self.ngram_counts: Dict[int, Counter] = {k: Counter() for k in range(1, order + 1)}
        # context_totals[k][prefix_of_k_minus_1_tokens] = tổng số lần prefix xuất hiện
        self.context_totals: Dict[int, Counter] = {k: Counter() for k in range(2, order + 1)}
        # context_unique_next[k][prefix] = số loại token khác nhau đi sau prefix (cho trọng số lambda)
        self.context_unique_next: Dict[int, Counter] = {k: Counter() for k in range(2, order + 1)}

        # Các cấu trúc phục vụ Kneser-Ney Continuation Probability cho các bậc thấp (k < order)
        # unique_preceding[k][ngram_of_k] = số loại token đứng ngay trước ngram_of_k
        self.unique_preceding: Dict[int, Dict[Tuple[int, ...], set]] = {
            k: defaultdict(set) for k in range(1, order)
        }
        # unique_preceding_context[k][prefix_of_k_minus_1] = tổng số cặp (w_prev, ngram) duy nhất
        self.continuation_context_totals: Dict[int, Counter] = {
            k: Counter() for k in range(2, order)
        }
        self.total_bigram_types: int = 0
        self.total_unigram_tokens: int = 0
        self.is_fitted: bool = False

    def fit(self, train_tokens: np.ndarray, max_train_tokens: Optional[int] = None) -> "NGramLM":
        """Huấn luyện đếm tần suất n-gram từ mảng token_ids (uint16)."""
        tokens = train_tokens[:max_train_tokens] if max_train_tokens else train_tokens
        tokens_list = tokens.tolist()
        n_tok = len(tokens_list)
        if n_tok < self.order:
            raise ValueError("Chuỗi huấn luyện ngắn hơn bậc n-gram!")

        # 1. Đếm tần suất cho tất cả các bậc 1 .. order
        for k in range(1, self.order + 1):
            counts_k = self.ngram_counts[k]
            for i in range(n_tok - k + 1):
                ng = tuple(tokens_list[i : i + k])
                counts_k[ng] += 1

        self.total_unigram_tokens = sum(self.ngram_counts[1].values())

        # 2. Tính thống kê ngữ cảnh và tập tiền tố duy nhất cho Kneser-Ney
        for k in range(2, self.order + 1):
            ctx_tot = self.context_totals[k]
            ctx_uniq = self.context_unique_next[k]
            for ng, cnt in self.ngram_counts[k].items():
                prefix = ng[:-1]
                ctx_tot[prefix] += cnt
                ctx_uniq[prefix] += 1

                # Phục vụ xác suất continuation cho bậc (k - 1)
                suffix = ng[1:]
                prev_tok = ng[0]
                self.unique_preceding[k - 1][suffix].add(prev_tok)

        for k in range(2, self.order):
            for suffix, prev_set in self.unique_preceding[k].items():
                prefix = suffix[:-1]
                self.continuation_context_totals[k][prefix] += len(prev_set)

        if self.order >= 2:
            self.total_bigram_types = len(self.ngram_counts[2])

        self.is_fitted = True
        return self

    def _kn_prob(self, history: Tuple[int, ...], word: int, level: int, is_highest: bool) -> float:
        """Tính đệ quy xác suất Interpolated Kneser-Ney P_KN(word | history) ở bậc `level`."""
        # Bậc cơ sở (unigram, level == 1)
        if level == 1:
            if self.order == 1 or self.total_bigram_types == 0:
                # Add-1 smoothing nếu chỉ dùng mô hình 1-gram
                cnt = self.ngram_counts[1].get((word,), 0)
                return (cnt + 1.0) / (self.total_unigram_tokens + self.vocab_size)
            else:
                # P_continuation(w) = |{w_{i-1} : c(w_{i-1}, w) > 0}| / |{(w_{j-1}, w_j) : c > 0}|
                # Trộn với phân phối đều nhỏ (1e-5) để bảo đảm mọi token OOV đều có P > 0
                cont_cnt = len(self.unique_preceding[1].get((word,), ()))
                p_cont = cont_cnt / max(self.total_bigram_types, 1)
                return (1.0 - 1e-4) * p_cont + 1e-4 * (1.0 / self.vocab_size)

        prefix = history[-(level - 1) :]
        ngram = prefix + (word,)

        if is_highest:
            c_ngram = self.ngram_counts[level].get(ngram, 0)
            c_prefix = self.context_totals[level].get(prefix, 0)
            n_unique_next = self.context_unique_next[level].get(prefix, 0)
        else:
            # Với các bậc trung gian (1 < level < order), Kneser-Ney dùng continuation counts
            c_ngram = len(self.unique_preceding[level].get(ngram, ()))
            c_prefix = self.continuation_context_totals[level].get(prefix, 0)
            n_unique_next = self.context_unique_next[level].get(prefix, 0)

        lower_prob = self._kn_prob(history, word, level - 1, is_highest=False)

        if c_prefix == 0:
            return lower_prob

        discounted = max(c_ngram - self.discount, 0.0) / c_prefix
        lam = (self.discount * n_unique_next) / c_prefix
        return discounted + lam * lower_prob

    def next_token_probs(self, context_ids: List[int]) -> np.ndarray:
        """Trả về phân phối xác suất trên toàn bộ vocab_size cho token tiếp theo."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước!")

        hist = tuple(context_ids[-(self.order - 1) :]) if self.order > 1 else ()
        effective_level = min(len(hist) + 1, self.order)

        probs = np.zeros(self.vocab_size, dtype=np.float64)
        for w in range(self.vocab_size):
            probs[w] = self._kn_prob(hist, w, level=effective_level, is_highest=(effective_level == self.order))

        # Chuẩn hóa nhẹ tránh sai số dấu phẩy động
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs.fill(1.0 / self.vocab_size)
        return probs

    def nll(self, token_ids: np.ndarray, ctx: int = 256) -> Tuple[float, int]:
        """
        Tính tổng Negative Log-Likelihood (nats) trên chuỗi token_ids.
        Tham số ctx được giữ đồng nhất chữ ký giao diện (n-gram dùng tối đa order-1 token trước đó).
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước!")

        tokens = token_ids.tolist()
        n = len(tokens)
        if n <= 1:
            return 0.0, 0

        total_nll = 0.0
        # Cache xác suất cho các ngữ cảnh lặp lại để tăng tốc độ đánh giá trên tập val/test
        cache: Dict[Tuple[int, ...], float] = {}

        for i in range(1, n):
            start_idx = max(0, i - (self.order - 1)) if self.order > 1 else i
            hist = tuple(tokens[start_idx:i])
            target = tokens[i]
            key = hist + (target,)

            prob = cache.get(key)
            if prob is None:
                effective_level = len(hist) + 1
                prob = self._kn_prob(
                    hist, target, level=effective_level, is_highest=(effective_level == self.order)
                )
                prob = max(prob, 1e-12)
                if len(cache) < 500_000:
                    cache[key] = prob

            total_nll -= math.log(prob)

        return float(total_nll), int(n - 1)

    def count_params(self) -> Dict[str, int]:
        """Số tham số hiệu dụng = tổng số n-gram duy nhất lưu trong bảng đếm."""
        total_ngrams = sum(len(c) for c in self.ngram_counts.values())
        return {
            "total_params": int(total_ngrams),
            "non_embedding_params": int(total_ngrams),
        }

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> List[int]:
        """Sinh chuỗi token tiếp theo từ prompt_ids."""
        rng = np.random.default_rng(seed)
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            probs = self.next_token_probs(generated)
            if temperature <= 0.0:
                next_id = int(np.argmax(probs))
            else:
                logits = np.log(np.maximum(probs, 1e-12)) / temperature
                logits -= np.max(logits)
                exp_p = np.exp(logits)
                probs = exp_p / np.sum(exp_p)
                next_id = int(rng.choice(self.vocab_size, p=probs))
            generated.append(next_id)
        return generated