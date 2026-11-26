"""GPT configuration, size presets, and a closed-form parameter counter.

Everything here is infrastructure (written with AI help, see docs/ai_usage/role2_gpt.md).
The model itself lives in `src/model/gpt.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

# Only these values are implemented today. The other names are reserved for the
# week-4 ablations (RoPE / RMSNorm / SwiGLU) so checkpoints stay forward-compatible.
_IMPLEMENTED = {
    "pos_type": {"learned"},
    "norm_type": {"layernorm"},
    "mlp_type": {"gelu"},
}


@dataclass
class GPTConfig:
    """Hyper-parameters of the decoder-only Transformer (interface contract v0)."""

    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    pos_type: str = "learned"
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.block_size <= 0:
            raise ValueError("vocab_size and block_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} is not divisible by n_head={self.n_head}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        for key, allowed in _IMPLEMENTED.items():
            value = getattr(self, key)
            if value not in allowed:
                raise NotImplementedError(f"{key}={value!r} is reserved for week-4 ablations, not implemented yet")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | "GPTConfig") -> "GPTConfig":
        if isinstance(payload, GPTConfig):
            return payload
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown GPTConfig keys: {sorted(unknown)}")
        return cls(**payload)


# Shapes only. vocab_size and block_size come from the data / yaml config.
# Head dim is 64 everywhere. Non-embedding counts assume bias=False (see estimate_params).
PRESETS: dict[str, dict[str, int]] = {
    "tiny_debug": {"n_layer": 2, "n_head": 2, "n_embd": 128},   # ~0.39M non-emb, CPU smoke tests
    # Scaling ladder (decision C: D = 20 x N_non_emb tokens per run)
    "scale_1m": {"n_layer": 3, "n_head": 3, "n_embd": 192},     # ~1.33M
    "scale_3m": {"n_layer": 4, "n_head": 4, "n_embd": 256},     # ~3.15M
    "scale_7m": {"n_layer": 6, "n_head": 5, "n_embd": 320},     # ~7.38M
    "scale_15m": {"n_layer": 8, "n_head": 6, "n_embd": 384},    # ~14.2M
    "main_30m": {"n_layer": 8, "n_head": 8, "n_embd": 512},     # ~25.2M non-emb, ~29.6M total (decision A)
    # Depth vs width ablation at a ~25-28M non-emb budget (week 4)
    "abl_wide_5x640": {"n_layer": 5, "n_head": 10, "n_embd": 640},  # ~24.6M
    "abl_deep_16x384": {"n_layer": 16, "n_head": 6, "n_embd": 384},  # ~28.3M
}

SCALING_LADDER: tuple[str, ...] = ("scale_1m", "scale_3m", "scale_7m", "scale_15m", "main_30m")


def make_config(preset: str, vocab_size: int, block_size: int = 512, **overrides: Any) -> GPTConfig:
    """Build a GPTConfig from a named preset plus data-dependent fields."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}. Available: {sorted(PRESETS)}")
    kwargs: dict[str, Any] = {**PRESETS[preset], "vocab_size": vocab_size, "block_size": block_size}
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def estimate_params(cfg: GPTConfig) -> dict[str, int]:
    """Count parameters from the formula, without building the model.

    Convention (matches docs/eval_protocol.md, Kaplan et al. 2020):
      * total         = every unique trainable tensor (a tied lm_head is counted once)
      * non_embedding = total - token embedding - position embedding
                        (- lm_head as well when it is NOT tied, since it is the unembedding)
    Used by tests to cross-check `GPT.count_params`.
    """
    C, V, T, L = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
    b = 1 if cfg.bias else 0

    norm = C + b * C                       # LayerNorm weight (+ bias)
    attn = (C * 3 * C + b * 3 * C) + (C * C + b * C)        # qkv projection + output projection
    mlp = (C * 4 * C + b * 4 * C) + (4 * C * C + b * C)     # expand to 4C + project back
    per_block = 2 * norm + attn + mlp

    tok_emb = V * C
    pos_emb = T * C
    lm_head = 0 if cfg.tie_weights else V * C

    non_embedding = L * per_block + norm   # + final norm
    total = non_embedding + tok_emb + pos_emb + lm_head
    return {"total": int(total), "non_embedding": int(non_embedding)}
