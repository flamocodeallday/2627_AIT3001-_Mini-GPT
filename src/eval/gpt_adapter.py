import inspect
from typing import Dict, Tuple, Optional, Callable, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLanguageModel


class GPTAdapter(BaseLanguageModel):
    """
    Adapter bọc mô hình GPT của Thành viên 2 vào giao diện BaseLanguageModel.
    Tuân thủ Hợp đồng giao diện v0:
      - forward(idx, targets=None) -> (logits, loss)
      - checkpoint: {state_dict, config, tokenizer_meta, step, val_loss}
    """

    def __init__(
        self,
        gpt_model: nn.Module,
        vocab_size: int,
        tokenizer_hash: Optional[str] = None,
        name: str = "mini_gpt",
        device: Optional[str] = None,
    ):
        super().__init__(name=name, vocab_size=vocab_size, tokenizer_hash=tokenizer_hash)
        self.model = gpt_model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    @torch.no_grad()
    def nll(self, token_ids: np.ndarray, ctx: int = 256, batch_size: int = 32) -> Tuple[float, int]:
        """Tính tổng Negative Log-Likelihood (nats) trên chuỗi token_ids."""
        self.model.eval()
        n = len(token_ids)
        if n <= 1:
            return 0.0, 0

        num_chunks = (n - 1) // ctx
        total_nll = 0.0
        total_tokens = 0

        if num_chunks > 0:
            usable = num_chunks * ctx
            x_all = torch.from_numpy(token_ids[:usable].astype(np.int64)).view(num_chunks, ctx)
            y_all = torch.from_numpy(token_ids[1 : usable + 1].astype(np.int64)).view(num_chunks, ctx)

            for start in range(0, num_chunks, batch_size):
                xb = x_all[start : start + batch_size].to(self.device)
                yb = y_all[start : start + batch_size].to(self.device)
                logits, _ = self.model(xb, targets=None)
                chunk_nll = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    yb.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(chunk_nll.item())
                total_tokens += int(yb.numel())

        rem_start = num_chunks * ctx
        if rem_start < n - 1:
            xr = torch.from_numpy(token_ids[rem_start : n - 1].astype(np.int64)).unsqueeze(0).to(self.device)
            yr = torch.from_numpy(token_ids[rem_start + 1 : n].astype(np.int64)).unsqueeze(0).to(self.device)
            logits, _ = self.model(xr, targets=None)
            rem_nll = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                yr.reshape(-1),
                reduction="sum",
            )
            total_nll += float(rem_nll.item())
            total_tokens += int(yr.numel())

        return total_nll, total_tokens

    def count_params(self) -> Dict[str, int]:
        """
        Gọi count_params(exclude_embeddings=...) của Thành viên 2 nếu có,
        hoặc tự động đếm theo chuẩn Kaplan et al. (2020).
        """
        if hasattr(self.model, "count_params") and callable(self.model.count_params):
            sig = inspect.signature(self.model.count_params)
            if "exclude_embeddings" in sig.parameters:
                total = int(self.model.count_params(exclude_embeddings=False))
                non_emb = int(self.model.count_params(exclude_embeddings=True))
                return {"total_params": total, "non_embedding_params": non_emb}

        unique_params = {id(p): p for p in self.model.parameters() if p.requires_grad}
        total = sum(p.numel() for p in unique_params.values())
        emb_params = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Embedding):
                emb_params += mod.weight.numel()
        return {
            "total_params": int(total),
            "non_embedding_params": int(max(0, total - emb_params)),
        }

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        model_builder: Callable[[Any], nn.Module],
        name: str = "mini_gpt",
        device: Optional[str] = None,
    ) -> "GPTAdapter":
        """Load checkpoint chuẩn của nhóm và trích xuất tokenizer_hash để kiểm tra an toàn."""
        ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
        cfg = ckpt["config"]
        tok_meta = ckpt.get("tokenizer_meta", {})
        tok_hash = tok_meta.get("tokenizer_hash")
        vocab_size = int(tok_meta.get("vocab_size") or getattr(cfg, "vocab_size", cfg.get("vocab_size")))

        gpt_model = model_builder(cfg)
        gpt_model.load_state_dict(ckpt["state_dict"])
        return cls(
            gpt_model=gpt_model,
            vocab_size=vocab_size,
            tokenizer_hash=tok_hash,
            name=name,
            device=device,
        )