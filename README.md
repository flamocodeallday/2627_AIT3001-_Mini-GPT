# 2627_AIT3001-_Mini-GPT

Bài tập lớn môn Học sâu (Đề tài 9): Tự cài đặt mini-GPT tiếng Việt từ đầu, so sánh với baseline N-gram (Interpolated Kneser-Ney) và LSTM tự cài đặt, khảo sát Model Scaling Laws.

Tài liệu của nhóm (giao thức đánh giá, nhật ký dùng AI theo từng thành viên, tài liệu riêng từng vai trò): [`docs/README.md`](docs/README.md).

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

---

## Mô hình GPT (Thành viên 2)

Phân công (phần AI dựng, phần tự cài) và những gì rút ra khi làm: `docs/ai_usage/role2_gpt.md`.

```powershell
pytest -q tests/test_gpt.py tests/test_train.py        # chấm từng phần: thêm -k attention, -k train_step, ...
python -m scripts.smoke_test                            # train → checkpoint → load → eval → generate trên v0_byte (CPU)
python -m src.train.train --config configs/gpt/tiny_debug.yaml
```

Trên Colab (GPU), model chính và thang scaling (quyết định A–C: 8×512, `block_size` 512, D = 20 × N_non-emb):

```bash
python -m src.train.train --config configs/gpt/main_30m.yaml --device cuda
python -m src.train.train --config configs/gpt/scaling.yaml --preset scale_3m --device cuda
python -m src.train.train --config configs/gpt/main_30m.yaml --device cuda --resume checkpoints/<run_id>/last.pt
```

- Checkpoint: `checkpoints/<run_id>/{best,last}.pt` (bị gitignore), theo hợp đồng `{state_dict, config, tokenizer_meta, step, val_loss}`. TV3 load bằng `GPTAdapter.from_checkpoint(path, model_builder=src.train.checkpoint.gpt_model_builder)`.
- Log TensorBoard: `logs/tb/<run_id>`. Tóm tắt mỗi run (config, số tham số, số token, token/tham số, commit, phần cứng): `results/train_<run_id>.json`.

---

## Phần Thành viên 3 (Quang): Baseline, Đánh giá & Scaling (`src/eval/`, `scripts/`)

### Cấu trúc mã nguồn
- `src/eval/base.py`: `DatasetBundle` (kiểm tra `meta.json` & `tokenizer_hash`), giao diện `BaseLanguageModel`, hàm `evaluate_model()` tính PPL/BPC và ghi đầy đủ metadata tái lập.
- `src/eval/ngram.py`: `NGramLM` tự cài đặt Interpolated Kneser-Ney (tối ưu đếm bằng NumPy).
- `src/eval/lstm.py`: `CustomLSTMCell` và `LSTMLM` tự cài đặt từ đầu (không dùng `nn.LSTM`/`nn.LSTMCell`).
- `src/eval/gpt_adapter.py`: `GPTAdapter` bọc mô hình GPT của TV2 vào `BaseLanguageModel`.
- `docs/eval_protocol.md`: Quy chuẩn đánh giá, quy ước đếm tham số và luật ngân sách scaling.

### Cách chạy & File đầu ra

1. **Chạy unit test (chạy được trên clone sạch không cần `data/`):**
   ```bash
   pytest -q
   ```

2. **Huấn luyện Baseline LSTM (`ctx=512`):**
   - LSTM nhỏ (`lstm_l2_h256`, ~1.05M non-emb):
     ```bash
     python -m scripts.train_lstm --data_dir data/processed/v1_bpe8192 --preset small --ctx 512 --batch_size 16 --max_steps 3000 --eval_interval 500
     ```
   - LSTM lớn cùng cỡ GPT chính (`lstm_l3_h1024`, ~25.2M non-emb, chạy trên Colab GPU):
     ```bash
     python -m scripts.train_lstm --data_dir data/processed/v1_bpe8192 --preset large --ctx 512 --batch_size 16 --grad_accum 2 --lr 1e-3 --max_steps 3000 --eval_interval 500
     ```
     *(Có thể thêm `--resume results/checkpoints/<model_name>_<version>_best.pt` để train nối tiếp từ checkpoint đã lưu.)*
   - **Đầu ra:** Checkpoint tốt nhất tại `results/checkpoints/<model_name>_<version>_best.pt` và log TensorBoard tại `runs/`.

3. **Đánh giá tổng hợp trên toàn bộ tập `val` (`ctx=512`, official):**
   ```bash
   python -m scripts.run_eval_all \
     --data_dir data/processed/v1_bpe8192 \
     --split val --ctx 512 \
     --ngram_train_tokens 10000000 \
     --lstm_ckpts results/checkpoints/lstm_l2_h256_v1_bpe8192_best.pt results/checkpoints/lstm_l3_h1024_v1_bpe8192_best.pt \
     --gpt_ckpt <path_to_gpt_ckpt>
   ```
   - **Đầu ra:**
     - `results/<model_name>_v1_bpe8192_val.json` (metadata đầy đủ từng mô hình)
     - `results/summary_v1_bpe8192_val.md` và `results/summary_v1_bpe8192_val.csv`

4. **Khảo sát Model Scaling Laws ($D = 20 \times N_{\text{non-emb}}$, `ctx=512`):**
   - GPT (chính, lấy preset từ `src/model/config.py`):
     ```bash
     python -m scripts.run_scaling --arch gpt --data_dir data/processed/v1_bpe8192 --ctx 512 --token_multiplier 20
     ```
   - LSTM (đường so sánh phụ):
     ```bash
     python -m scripts.run_scaling --arch lstm --data_dir data/processed/v1_bpe8192 --ctx 512 --token_multiplier 20
     ```
   - **Đầu ra:** `results/scaling_<arch>_<version>.json` và biểu đồ `results/scaling_law_<arch>_<version>.png`.

5. **Pipeline đa nền tảng (Windows / Linux / Colab):**
   ```bash
   python -m scripts.run_pipeline --data_dir data/processed/v1_bpe8192 --ctx 512 --run_train_lstm_small --run_eval
   ```

### Kết quả Official trên toàn bộ `val` (`v1_bpe8192`, `ctx=512`)
Xem chi tiết tại [results/summary_v1_bpe8192_val.md](results/summary_v1_bpe8192_val.md):

| Model | Non-Emb Params | Train Tokens | Eval Tokens | Avg NLL | Perplexity | BPC |
|---|---|---|---|---|---|---|
| `ngram_kn_n3` | 5,687,280 | 10,000,000 | 2,775,100 | 4.8865 | 132.49 | 2.0493 |
| `ngram_kn_n4` | 13,056,157 | 10,000,000 | 2,775,100 | 4.8603 | 129.06 | 2.0383 |
| `lstm_l2_h256` | 1,051,136 | 24,576,000 | 2,775,100 | 4.2938 | 73.25 | 1.8007 |
| `lstm_l3_h1024` | 25,180,160 | 49,152,000 | 2,775,100 | 3.8694 | 47.91 | 1.6227 |
