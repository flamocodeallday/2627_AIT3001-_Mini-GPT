# 2627_AIT3001-_Mini-GPT

Mini-GPT tiếng Việt từ đầu (Đề tài 9 — Học sâu AIT3001).

## Cài đặt môi trường Local (Python 3.13.5)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Chạy kiểm thử đơn vị (local fixtures ~500 KB):

```powershell
pytest
```

## Pipeline dữ liệu (Thành viên 1)

### Gói 1: Làm sạch, khử trùng lặp và chia split
- Chạy thử trên bộ giả lập local:
  ```powershell
  python -m scripts.prepare_data --config configs/data.yaml --source fixtures --inspect-n 5
  ```
- Chạy trên Google Colab Pro (stream toàn bộ `wikimedia/wikipedia` `20231101.vi`, xuất `.jsonl.zst` lên Google Drive):
  ```bash
  python -m scripts.prepare_data \
    --config configs/data.yaml \
    --source hf \
    --output-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
    --compress \
    --inspect-n 50 \
    --inspect-out /content/drive/MyDrive/minigpt/interim/sample_50_inspect.txt
  ```

### Gói 2: Đóng gói `v0_byte` (`train.bin`, `val.bin`, `test.bin`, `meta.json`)
- Chạy trên Google Colab Pro từ split đã làm sạch (lát cắt ~10 MB train, ~1 MB val, ~1 MB test):
  ```bash
  python -m scripts.pack_data \
    --config configs/data.yaml \
    --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
    --output-dir /content/drive/MyDrive/minigpt/processed/v0_byte \
    --sanity-check
  ```

### Gói 4: Khảo sát kích thước từ vựng (`4096`, `8192`, `16384`) & Đối chiếu `tokenizers`
- Chạy test nhanh trên fixtures local:
  ```powershell
  python -m scripts.benchmark_vocab --source fixtures --train-mb 0.5 --vocab-sizes 300 350 --out-dir results/vocab_fixture
  ```
- Chạy thực chiến trên Google Colab Pro (trích 40 MB từ `train`, đánh giá trên toàn bộ `val`, xuất `custom_bpe*.json`, `vocab_benchmark.json` và `vocab_benchmark.md`):
  ```bash
  python -m scripts.benchmark_vocab \
    --config configs/data.yaml \
    --source interim \
    --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
    --train-mb 40 \
    --vocab-sizes 4096 8192 16384 \
    --out-dir /content/drive/MyDrive/minigpt/vocab_benchmark
  ```

### Gói 5: Encode song song toàn bộ corpus thành `v1_bpe8192`
- Nạp trực tiếp `custom_bpe8192.json` đã chốt từ Gói 4, chạy `ProcessPoolExecutor` trên toàn bộ `train`/`val`/`test` (không giới hạn `target_mb`) và kiểm tra chéo `.bin`:
  ```bash
  python -m scripts.pack_data \
    --config configs/data.yaml \
    --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
    --tokenizer-path /content/drive/MyDrive/minigpt/vocab_benchmark/custom_bpe8192.json \
    --output-dir /content/drive/MyDrive/minigpt/processed/v1_bpe8192 \
    --batch-size 500 \
    --sanity-check
  ```

### Tích hợp đọc `.bin` bằng PyTorch cho Thành viên 2 (`MemmapDataset`)

```python
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapTextDataset(Dataset):
    """Đọc mảng token uint16 qua memmap với bộ nhớ gần như bằng 0."""

    def __init__(self, bin_path: str, block_size: int) -> None:
        self.data = np.memmap(bin_path, dtype="<u2", mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]

# Đọc metadata: meta = json.load(open("data/processed/v0_byte/meta.json"))
# vocab_size, eos_id = meta["vocab_size"], meta["eos_id"]
```
