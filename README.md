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
