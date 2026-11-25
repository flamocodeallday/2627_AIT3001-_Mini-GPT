import os
import json
import math
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np


class DatasetBundle:
    """Đọc dữ liệu nhị phân uint16 (.bin) và metadata chuẩn theo Hợp đồng v0."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Không tìm thấy meta.json tại: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.version: str = self.meta["version"]
        self.tokenizer_type: str = self.meta["tokenizer_type"]
        self.vocab_size: int = int(self.meta["vocab_size"])
        self.eos_id: int = int(self.meta["eos_id"])
        self.seed: int = int(self.meta["seed"])
        self.tokenizer_hash: str = self.meta["tokenizer_hash"]
        self.splits_meta: Dict = self.meta["splits"]

    def get_split_tokens(self, split: str) -> np.ndarray:
        """Trả về mảng memmap uint16 cho split ('train', 'val', 'test')."""
        if split not in self.splits_meta:
            raise ValueError(f"Split không hợp lệ: {split}. Chỉ chấp nhận {list(self.splits_meta.keys())}")
        bin_path = os.path.join(self.data_dir, f"{split}.bin")
        arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        expected_tokens = int(self.splits_meta[split]["n_tokens"])
        if len(arr) != expected_tokens:
            raise ValueError(
                f"Số token trong {split}.bin ({len(arr)}) không khớp meta.json ({expected_tokens})!"
            )
        return arr

    def get_split_n_chars(self, split: str) -> int:
        return int(self.splits_meta[split]["n_chars"])


class BaseLanguageModel(ABC):
    """Giao diện chuẩn bắt buộc cho N-gram, LSTM và GPTAdapter."""

    def __init__(self, name: str, vocab_size: int, tokenizer_hash: Optional[str] = None):
        self.name = name
        self.vocab_size = vocab_size
        self.tokenizer_hash = tokenizer_hash

    @abstractmethod
    def nll(self, token_ids: np.ndarray, ctx: int) -> Tuple[float, int]:
        """
        Tính tổng Negative Log-Likelihood (nats) trên chuỗi token_ids.
        Trả về: (total_nll, num_tokens_evaluated)
        """
        raise NotImplementedError

    def count_params(self) -> Dict[str, int]:
        """Trả về thông tin số tham số của mô hình."""
        return {"total_params": 0, "non_embedding_params": 0}


@dataclass
class EvalMetrics:
    model_name: str
    data_version: str
    split: str
    tag: str                    # 'debug' nếu v0_byte, 'official' nếu bản chính
    ctx: int
    total_nll: float
    num_tokens: int
    n_chars: int
    avg_nll: float
    perplexity: float
    bpc: float
    params: Dict[str, int]
    timestamp: str
    git_commit: str


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def evaluate_model(
    model: BaseLanguageModel,
    dataset: DatasetBundle,
    split: str = "val",
    ctx: int = 256,
    max_tokens: Optional[int] = None,
    save_dir: Optional[str] = "results",
) -> EvalMetrics:
    """
    Hàm đánh giá thống nhất theo Hợp đồng giao diện v0.
    Tự động kiểm tra tokenizer_hash và gắn nhãn 'debug' cho v0_byte.
    """
    if model.vocab_size != dataset.vocab_size:
        raise ValueError(
            f"Lệch vocab_size: model={model.vocab_size} vs dataset={dataset.vocab_size}"
        )

    if model.tokenizer_hash is not None and model.tokenizer_hash != dataset.tokenizer_hash:
        raise ValueError(
            f"Lệch tokenizer_hash! Model ({model.tokenizer_hash[:12]}...) != "
            f"Dataset ({dataset.tokenizer_hash[:12]}...). Dừng đánh giá để tránh sai lệch!"
        )

    tokens = dataset.get_split_tokens(split)
    full_n_chars = dataset.get_split_n_chars(split)

    if max_tokens is not None and max_tokens < len(tokens):
        ratio = max_tokens / len(tokens)
        tokens = tokens[:max_tokens]
        n_chars = max(int(full_n_chars * ratio), 1)
    else:
        n_chars = full_n_chars

    total_nll, eval_tokens = model.nll(tokens, ctx=ctx)
    if eval_tokens <= 0:
        raise ValueError("Số token được đánh giá bằng 0!")

    avg_nll = total_nll / eval_tokens
    perplexity = math.exp(min(avg_nll, 100.0))
    bpc = total_nll / (n_chars * math.log(2.0))

    tag = "debug" if dataset.version == "v0_byte" else "official"

    result = EvalMetrics(
        model_name=model.name,
        data_version=dataset.version,
        split=split,
        tag=tag,
        ctx=ctx,
        total_nll=round(total_nll, 4),
        num_tokens=int(eval_tokens),
        n_chars=int(n_chars),
        avg_nll=round(avg_nll, 6),
        perplexity=round(perplexity, 4),
        bpc=round(bpc, 6),
        params=model.count_params(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_commit=_get_git_commit(),
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = f"{model.name}_{dataset.version}_{split}.json"
        with open(os.path.join(save_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    return result