import os
import json
import math
import time
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import numpy as np


def _load_uint16_memmap(bin_path: str) -> np.ndarray:
    """
    Đọc mảng token uint16 từ file .bin.
    Dùng chung MemmapDataset của TV1 (src.data.pack) theo quy ước repo,
    fallback về np.memmap nếu file quá ngắn (<= 1 token) hoặc thiếu module.
    """
    try:
        from src.data.pack import MemmapDataset  # type: ignore

        ds = MemmapDataset(bin_path, block_size=1)
        if hasattr(ds, "data"):
            return ds.data
    except Exception:
        pass
    return np.memmap(bin_path, dtype="<u2", mode="r")


class DatasetBundle:
    """Đọc dữ liệu nhị phân uint16 (.bin) và kiểm tra metadata/hash chuẩn theo Hợp đồng v0."""

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
        arr = _load_uint16_memmap(bin_path)
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
        self.train_tokens: Optional[int] = None
        self.train_config: Dict[str, Any] = {}

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
    tag: str                    # 'debug' nếu v0_byte hoặc đánh giá tập con, 'official' nếu toàn bộ val/test bản chính
    ctx: int
    is_full_split: bool
    total_nll: float
    num_tokens: int
    n_chars: int
    avg_nll: float
    perplexity: float
    bpc: float
    params: Dict[str, int]
    train_tokens: Optional[int] = None
    train_config: Dict[str, Any] = field(default_factory=dict)
    hardware: str = "unknown"
    eval_time_sec: float = 0.0
    timestamp: str = ""
    git_commit: str = "unknown"
    git_dirty: bool = False


def _get_git_info() -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit, bool(status)
    except Exception:
        return "unknown", False


def _get_hardware_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    proc = platform.processor() or platform.machine() or "cpu"
    return f"cpu:{proc}"


def evaluate_model(
    model: BaseLanguageModel,
    dataset: DatasetBundle,
    split: str = "val",
    ctx: int = 512,
    max_tokens: Optional[int] = None,
    save_dir: Optional[str] = "results",
    train_tokens: Optional[int] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> EvalMetrics:
    """
    Hàm đánh giá thống nhất theo Hợp đồng giao diện v0.
    Tự động kiểm tra tokenizer_hash, gắn nhãn 'debug' cho v0_byte hoặc tập con,
    và ghi đầy đủ metadata tái lập (hyperparams, train_tokens, hardware, git_commit + dirty).
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
    is_full_split = max_tokens is None or max_tokens >= len(tokens)

    if not is_full_split and max_tokens is not None:
        ratio = max_tokens / len(tokens)
        tokens = tokens[:max_tokens]
        n_chars = max(int(full_n_chars * ratio), 1)
    else:
        n_chars = full_n_chars

    t0 = time.time()
    total_nll, eval_tokens = model.nll(tokens, ctx=ctx)
    eval_time_sec = round(time.time() - t0, 2)

    if eval_tokens <= 0:
        raise ValueError("Số token được đánh giá bằng 0!")

    avg_nll = total_nll / eval_tokens
    perplexity = math.exp(min(avg_nll, 100.0))
    bpc = total_nll / (n_chars * math.log(2.0))

    tag = "debug" if (dataset.version == "v0_byte" or not is_full_split) else "official"
    git_commit, git_dirty = _get_git_info()

    resolved_train_tokens = train_tokens if train_tokens is not None else getattr(model, "train_tokens", None)
    resolved_train_config = (
        dict(train_config)
        if train_config is not None
        else dict(getattr(model, "train_config", {}) or {})
    )

    result = EvalMetrics(
        model_name=model.name,
        data_version=dataset.version,
        split=split,
        tag=tag,
        ctx=ctx,
        is_full_split=is_full_split,
        total_nll=round(total_nll, 4),
        num_tokens=int(eval_tokens),
        n_chars=int(n_chars),
        avg_nll=round(avg_nll, 6),
        perplexity=round(perplexity, 4),
        bpc=round(bpc, 6),
        params=model.count_params(),
        train_tokens=int(resolved_train_tokens) if resolved_train_tokens is not None else None,
        train_config=resolved_train_config,
        hardware=_get_hardware_info(),
        eval_time_sec=eval_time_sec,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = f"{model.name}_{dataset.version}_{split}.json"
        with open(os.path.join(save_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    return result