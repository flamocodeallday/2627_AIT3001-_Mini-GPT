import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLanguageModel


@dataclass
class LSTMConfig:
    vocab_size: int = 257
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    tie_weights: bool = True


class CustomLSTMCell(nn.Module):
    """
    Tự cài đặt một bước thời gian của LSTM (Hochreiter & Schmidhuber, 1997; Olah, 2015).
    Không sử dụng torch.nn.LSTMCell hay torch.nn.LSTM.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gộp 4 phép chiếu tuyến tính (i, f, g, o) thành 2 ma trận để tối ưu tính toán trên GPU
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        # Khởi tạo bias cổng quên (forget gate) = 1.0 giúp truyền gradient tốt hơn ở bước đầu
        with torch.no_grad():
            self.weight_ih.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def forward(
        self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: (batch_size, input_size)
        state: (h_prev, c_prev), mỗi tensor có shape (batch_size, hidden_size)
        """
        h_prev, c_prev = state
        gates = self.weight_ih(x_t) + self.weight_hh(h_prev)

        # Tách 4 cổng theo chiều đặc trưng
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_gate)   # Input gate
        f_t = torch.sigmoid(f_gate)   # Forget gate
        g_t = torch.tanh(g_gate)      # Cell candidate
        o_t = torch.sigmoid(o_gate)   # Output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMLM(nn.Module, BaseLanguageModel):
    """
    Mô hình ngôn ngữ đa tầng sử dụng CustomLSTMCell tự viết,
    tuân thủ giao diện BaseLanguageModel của nhóm.
    """

    def __init__(
        self,
        config: LSTMConfig,
        tokenizer_hash: Optional[str] = None,
        name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        BaseLanguageModel.__init__(
            self,
            name=name or f"lstm_l{config.num_layers}_h{config.hidden_dim}",
            vocab_size=config.vocab_size,
            tokenizer_hash=tokenizer_hash,
        )
        self.config = config
        if config.tie_weights and config.embed_dim != config.hidden_dim:
            raise ValueError("Khi bật tie_weights=True, embed_dim phải bằng hidden_dim!")

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        self.emb_drop = nn.Dropout(config.dropout)

        self.cells = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            in_dim = config.embed_dim if layer_idx == 0 else config.hidden_dim
            self.cells.append(CustomLSTMCell(in_dim, config.hidden_dim))

        self.layer_drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for _ in range(self.config.num_layers):
            h = torch.zeros(batch_size, self.config.hidden_dim, device=self.device)
            c = torch.zeros(batch_size, self.config.hidden_dim, device=self.device)
            states.append((h, c))
        return states

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        idx: (B, T) token IDs
        targets: (B, T) token IDs dịch phải 1 bước
        """
        B, T = idx.shape
        if states is None:
            states = self.init_hidden(B)

        x = self.emb_drop(self.token_emb(idx))  # (B, T, embed_dim)
        outputs = []

        # Duyệt tuần tự theo chiều thời gian T qua từng tầng CustomLSTMCell
        for t in range(T):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, states[layer_idx])
                states[layer_idx] = (h_t, c_t)
                x_t = self.layer_drop(h_t) if layer_idx < self.config.num_layers - 1 else h_t
            outputs.append(x_t)

        hidden_seq = torch.stack(outputs, dim=1)  # (B, T, hidden_dim)
        logits = self.lm_head(self.ln_f(hidden_seq))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss, states

    @torch.no_grad()
    def nll(self, token_ids: np.ndarray, ctx: int = 256, batch_size: int = 64) -> Tuple[float, int]:
        """
        Tính tổng Negative Log-Likelihood (nats) trên chuỗi token_ids theo từng đoạn dài `ctx`.
        """
        self.eval()
        n = len(token_ids)
        if n <= 1:
            return 0.0, 0

        # Chia chuỗi thành các mẫu (x, y) không chồng lấp độ dài `ctx`
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
                logits, _, _ = self.forward(xb)
                chunk_nll = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    yb.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(chunk_nll.item())
                total_tokens += int(yb.numel())

        # Xử lý phần đuôi còn dư (< ctx token) để không bỏ sót token nào của tập val/test
        rem_start = num_chunks * ctx
        if rem_start < n - 1:
            xr = torch.from_numpy(token_ids[rem_start : n - 1].astype(np.int64)).unsqueeze(0).to(self.device)
            yr = torch.from_numpy(token_ids[rem_start + 1 : n].astype(np.int64)).unsqueeze(0).to(self.device)
            logits, _, _ = self.forward(xr)
            rem_nll = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                yr.reshape(-1),
                reduction="sum",
            )
            total_nll += float(rem_nll.item())
            total_tokens += int(yr.numel())

        return total_nll, total_tokens

    def count_params(self) -> Dict[str, int]:
        """Đếm tham số (tự động loại trùng lặp nếu bật weight tying)."""
        unique_params = {id(p): p for p in self.parameters() if p.requires_grad}
        total = sum(p.numel() for p in unique_params.values())
        emb_params = self.token_emb.weight.numel()
        non_emb = total - emb_params
        return {
            "total_params": int(total),
            "non_embedding_params": int(non_emb),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[int]:
        """Sinh văn bản tiếp nối từ prompt_ids, duy trì trạng thái ẩn (h, c) từng bước."""
        self.eval()
        generated = list(prompt_ids)
        idx = torch.tensor([generated], dtype=torch.long, device=self.device)

        # Đẩy toàn bộ prompt qua mô hình để lấy state cuối cùng
        logits, _, states = self.forward(idx)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_id = int(torch.argmax(next_logits, dim=-1).item())
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled[indices_to_remove] = -float("inf")

                probs = F.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_id)
            step_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            next_logits, _, states = self.forward(step_input, states=states)
            next_logits = next_logits[:, -1, :]

        return generated

    def save_checkpoint(self, path: str, tokenizer_meta: Dict, step: int, val_loss: float):
        """Lưu checkpoint theo đúng Hợp đồng giao diện v0 của nhóm."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": asdict(self.config),
                "tokenizer_meta": tokenizer_meta,
                "step": step,
                "val_loss": val_loss,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "LSTMLM":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = LSTMConfig(**ckpt["config"])
        tok_hash = ckpt.get("tokenizer_meta", {}).get("tokenizer_hash")
        model = cls(config=cfg, tokenizer_hash=tok_hash, device=device)
        model.load_state_dict(ckpt["state_dict"])
        return model