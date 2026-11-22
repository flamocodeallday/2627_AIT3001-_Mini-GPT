# 2627_AIT3001-_Mini-GPT

Mini-GPT tiếng Việt từ đầu (Đề tài 9 — Học sâu AIT3001).

## Cài đặt môi trường Local (Python 3.13.5)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Chạy toàn bộ unit test (trên bộ giả lập ~500 KB `fixtures/sample_vi_wiki.jsonl`):

```powershell
pytest -v
```

---

## Tái lập dữ liệu từ đầu đến cuối (Thành viên 1)

Pipeline dữ liệu tuân thủ đúng thứ tự bất biến:
`Làm sạch (NFC + filter) -> Khử trùng lặp (SHA-256) -> Chia split (98/1/1, seed=42) -> Train BPE (chỉ trên train) -> Đóng gói uint16 (.bin)`.

### Bước 1: Tải, làm sạch & chia split (`prepare_data.py`)
- **Test local (fixtures):**
  ```powershell
  python -m scripts.prepare_data --config configs/data.yaml --source fixtures --inspect-n 5
  ```
- **Chạy thực chiến trên Google Colab Pro (`wikimedia/wikipedia`, `20231101.vi`):**
  ```bash
  python -m scripts.prepare_data \
    --config configs/data.yaml \
    --source hf \
    --output-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
    --compress \
    --inspect-n 50 \
    --inspect-out /content/drive/MyDrive/minigpt/interim/sample_50_inspect.txt
  ```

### Bước 2: Phát hành bản debug `v0_byte` (10 MB train / 1 MB val / 1 MB test)
```bash
python -m scripts.pack_data \
  --config configs/data.yaml \
  --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
  --output-dir /content/drive/MyDrive/minigpt/processed/v0_byte \
  --sanity-check
```

### Bước 3: Khảo sát cỡ từ vựng (`4096`, `8192`, `16384`) & đối chiếu `tokenizers`
```bash
python -m scripts.benchmark_vocab \
  --config configs/data.yaml \
  --source interim \
  --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
  --train-mb 40 \
  --vocab-sizes 4096 8192 16384 \
  --out-dir /content/drive/MyDrive/minigpt/vocab_benchmark
```
Nhóm đã thống nhất chốt `vocab_size = 8192` (`custom_bpe8192.json`).

### Bước 4: Đóng gói song song bản chính thức `v1_bpe8192`
```bash
python -m scripts.pack_data \
  --config configs/data.yaml \
  --interim-dir /content/drive/MyDrive/minigpt/interim/cleaned_splits \
  --tokenizer-path /content/drive/MyDrive/minigpt/vocab_benchmark/custom_bpe8192.json \
  --output-dir /content/drive/MyDrive/minigpt/processed/v1_bpe8192 \
  --batch-size 500 \
  --sanity-check
```

### Bước 5: Notebook khám phá & kiểm định (`notebooks/01_corpus_tokenizer.ipynb`)
- Mở `notebooks/01_corpus_tokenizer.ipynb`: giữ `MODE = "fixtures"` để chạy kiểm thử nhanh trên máy cá nhân, hoặc đổi `MODE = "colab"` để đọc trực tiếp dữ liệu thật từ Google Drive và chạy `Restart & Run All`.

---

## Hợp đồng giao diện dữ liệu (cho Thành viên 2 & Thành viên 3)

Mỗi thư mục phiên bản (`data/processed/<version>/`) gồm: `train.bin`, `val.bin`, `test.bin`, `tokenizer.json`, `meta.json`.

- **Định dạng nhị phân:** Mảng 1 chiều little-endian `uint16` (`dtype="<u2"` hoặc `np.uint16`), chèn **đúng 1 `eos_id`** giữa hai tài liệu liên tiếp.
- **Quy tắc bắt buộc khi tích hợp:**
  1. **Không hard-code `vocab_size` hay `eos_id`:** Luôn đọc động từ `meta.json` (`meta["vocab_size"]`, `meta["eos_id"]`).
  2. **Kiểm tra chéo `tokenizer_hash`:** So khớp `meta["tokenizer_hash"]` với SHA-256 của `tokenizer.json` (và lưu vào checkpoint GPT / kiểm tra trong adapter đánh giá).
  3. **Đóng băng `test.bin`:** Tuyệt đối không dùng `test.bin` để huấn luyện BPE hay tinh chỉnh mô hình; BPC dùng `meta["splits"][split]["n_chars"]`.

```python
import json
import numpy as np
import torch
from torch.utils.data import Dataset

with open("data/processed/v1_bpe8192/meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
vocab_size, eos_id = meta["vocab_size"], meta["eos_id"]

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
```

---

## Link tải dữ liệu đã đóng gói (External Storage — không đưa `.bin` vào Git)

| Phiên bản | Tokenizer | `vocab_size` | `eos_id` | Số token (`train` / `val` / `test`) | `tokenizer_hash` (SHA-256) | Link tải |
|---|---|---:|---:|---|---|---|
| `v0_byte` | `byte_level` | 257 | 256 | Lát cắt ~10 / 1 / 1 MiB | *(cập nhật từ `meta.json`)* | `[Điền link Google Drive v0_byte tại đây]` |
| `v1_bpe8192` | `custom_regex_bpe` | 8192 | 256 | `263,990,187` / `2,775,101` / `2,777,879` | `951cb42b935d6a837c568a57610749a6ce1be4ce6c0d323c77ee697ae4957f68` | `[Điền link Google Drive v1_bpe8192 tại đây]` |
