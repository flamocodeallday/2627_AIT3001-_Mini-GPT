"""Decoder-only Transformer (GPT) written from scratch for the mini-GPT project.

Who wrote what (also recorded in docs/ai_usage/role2_gpt.md):
  * Module wiring in every `__init__`, weight init and weight tying: scaffolding (AI-assisted).
  * Every method that raises `NotImplementedError("TODO(todd): ...")`: Todd (member 2)
    implements by hand. Specs, shapes and test commands are in docs/role2_gpt/gpt_todo.md.

Rules for the core: do NOT use `F.scaled_dot_product_attention`, `nn.MultiheadAttention`
or `nn.TransformerDecoderLayer` here. They are only allowed in tests, as a reference.

Shape legend: B = batch, T = sequence length (<= block_size), C = n_embd,
H = n_head, D = head_dim = C // H, V = vocab_size.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import GPTConfig
from src.model.sampling import sample_next_token


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> GELU -> C, then dropout."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C). Same transformation applied to every position independently."""
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x
        


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Contract used by the tests (keep it):
      * `qkv_proj(x)` has last dim 3C and is split IN THIS ORDER into q, k, v (each (B, T, C)).
      * Each of q, k, v is reshaped to (B, T, H, D) and transposed to (B, H, T, D).
      * Head outputs are merged back to (B, T, C) and passed through `out_proj`.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)    # on the attention weights
        self.resid_dropout = nn.Dropout(config.dropout)   # on the output, before the residual add
        # Lower-triangular boolean mask, shape (1, 1, block_size, block_size).
        # True = "query i may look at key j" (j <= i). persistent=False: it is derived from
        # block_size, so it is not stored in checkpoints (smaller files, no load mismatches).
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C).

        Steps: project to q, k, v -> split heads -> scores = q @ k^T / sqrt(D) -> mask the future
        -> softmax over keys -> attn_dropout -> weights @ v -> merge heads -> out_proj -> resid_dropout.
        Must work for any T <= block_size (slice the mask to T x T).
        """
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        qkv = self.qkv_proj(x)  # shape B, T, 3C
        q, k, v = qkv.split(C, dim = -1) # each shape B, T, C

        q = q.view(B, T, H, D).transpose(1, 2) # shape B, H, T, D
        k = k.view(B, T, H, D).transpose(1, 2) # shape B, H, T, D
        v = v.view(B, T, H, D).transpose(1, 2) # shape B, H, T, D

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = self.causal_mask[:, :, :T, :T]      # (1, 1, T, T)
        scores = scores.masked_fill(~mask, float("-inf"))   # điền -inf chỗ KHÔNG được nhìn
        att = scores.softmax(dim = -1)                      
        att = self.attn_dropout(att)                       # attn_dropout

        y = att @ v                         # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)                         # out_proj rồi resid_dropout
        y = self.resid_dropout(y)
        return y



class Block(nn.Module):
    """One pre-norm Transformer block: attention sub-layer then MLP sub-layer, each with a residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C). Pre-norm: normalize the INPUT of each sub-layer, add its output back to x."""
        # TODO(todd) bài 3
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x
        


class GPT(nn.Module):
    """Token + learned position embeddings -> n_layer Blocks -> final LayerNorm -> lm_head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # Weight tying (Press & Wolf 2017): the output projection reuses the token embedding
            # matrix (V, C). Saves V*C parameters (~4.2M for V=8192, C=512) and ties the "meaning"
            # of a token on the input side and the output side.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the projections that write INTO the residual stream:
        # each block adds 2 such outputs, so the residual variance grows with depth; dividing the
        # std by sqrt(2 * n_layer) keeps the stream's scale roughly constant at init.
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.fc_out.weight"):
                nn.init.normal_(param, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # N(0, 0.02) everywhere. With weight tying this matters a lot: N(0, 1) embeddings would make
        # the initial logits huge (std ~ sqrt(C)) and the first loss far above ln(V).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Interface contract v0 (GPTAdapter of member 3 relies on it).

        idx:     (B, T) int64 token ids, T <= block_size (raise ValueError otherwise).
        targets: (B, T) int64 next-token ids, or None.
        Returns: (logits, loss)
            logits: (B, T, V) for EVERY position (the evaluator needs the NLL of all tokens).
            loss:   scalar mean cross-entropy over all B*T positions if targets is given, else None.
        """
        B, T = idx.shape
        if T > self.config.block_size: 
            raise ValueError(f"Sequence length can not exceed block size, {T} > {self.config.block_size}")

        tok = self.tok_emb(idx)
        pos = torch.arange(T, device = idx.device)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        for block in self.blocks: 
            x = block(x)

        x = self.norm_f(x)                                    # (B, T, C)  norm_f
        logits = self.lm_head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))  # scalar
        return logits, loss



    def count_params(self, exclude_embeddings: bool = False) -> int:
        """Number of unique trainable parameters (interface contract v0).

        Convention (docs/eval_protocol.md, Kaplan et al. 2020):
          * a tied lm_head shares storage with tok_emb, so it is counted ONCE;
          * exclude_embeddings=True also removes tok_emb AND pos_emb
            (and lm_head too if it is not tied).
        Must equal `estimate_params(self.config)`.
        """
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.tok_emb.weight.numel()                   # tok_emb
            total -= self.pos_emb.weight.numel()                  # pos_emb
            if self.lm_head.weight is not self.tok_emb.weight:   # KHÔNG tie
                total -= self.lm_head.weight.numel()               # lm_head
        return total

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling (interface contract v0).

        idx: (B, T0) prompt ids (T0 may exceed block_size). Returns (B, T0 + max_new_tokens).
        Each step: keep only the last block_size tokens as context -> forward -> logits of the LAST
        position -> `sample_next_token(...)` -> append. Remember to put the model in eval mode
        and restore the previous mode afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.config.block_size:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :]
                    next_token = sample_next_token(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                    )
                    idx = torch.cat([idx, next_token], dim=1)
        finally:
            self.train(was_training)
        return idx
        


    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any] | GPTConfig) -> "GPT":
        """`model_builder` for GPTAdapter.from_checkpoint: rebuild the model from ckpt["config"]."""
        return cls(GPTConfig.from_dict(cfg))
