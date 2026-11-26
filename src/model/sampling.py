"""Next-token sampling strategies: greedy, temperature, top-k, top-p (nucleus).

All three functions are exercises for Todd (member 2). Graded by tests/test_gpt.py (-k "top_k or top_p or sample").
Shapes: logits is (B, V) = un-normalized scores for the NEXT token of each sequence.
"""

from __future__ import annotations

import torch


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits of each row, set every other entry to -inf.

    logits: (B, V). Returns a NEW tensor (B, V); do not modify the input in place.
    If k >= V the logits are returned unchanged. Ties at the k-th value may all be kept.
    """
    # take shape
    B, V = logits.shape
    # If k >= V the logits are returned unchanged
    if k >= V: 
        return logits.clone()
    values, _ = torch.topk(logits, k, dim=-1)   # values: (B, k), sắp GIẢM dần
    kth = values[:, -1:]                              # (B, 1): phần tử cuối = lớn thứ k
    return logits.masked_fill(logits < kth, float("-inf"))





def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probability mass reaches p.

    logits: (B, V), 0 < p <= 1. Returns a NEW tensor (B, V) with removed entries set to -inf.
    Rule: sort probabilities in descending order, keep a token if the cumulative mass of the
    tokens STRICTLY BEFORE it is < p. Hence the most likely token is always kept.
    p == 1.0 keeps everything.
    """
    if p >= 1.0:
        return logits.clone()

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)   # (B, V)
    probs = torch.softmax(sorted_logits, dim=-1)                                  # (B, V)
    cum = torch.cumsum(probs, dim=-1)                                         # (B, V)
    mass_before = cum - probs                                                 # (B, V)
    remove = mass_before >= p                                                 # (B, V) bool: True = bỏ
    filtered_sorted = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.empty_like(logits).scatter(-1, sorted_idx, filtered_sorted)   # (B, V) thứ tự gốc
    return out



def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one next token per row. Returns (B, 1) int64.

    Conventions:
      * temperature == 0 -> greedy (argmax), top_k / top_p are ignored.
      * otherwise: divide logits by temperature, then apply top-k (if given), then top-p (if given),
        then softmax and draw one sample with `torch.multinomial(..., generator=generator)`.
    """
    if temperature == 0: 
        return torch.argmax(logits, dim = -1, keepdim = True)

    scaled = logits/ temperature
    if top_k is not None:
        scaled = filter_top_k(scaled, top_k)
    if top_p is not None:
        scaled = filter_top_p(scaled, top_p)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
    
