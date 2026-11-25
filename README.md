# 2627_AIT3001-_Mini-GPT

Bài tập lớn môn Học sâu (Đề tài 9): Tự cài đặt mini-GPT tiếng Việt từ đầu, so sánh với baseline N-gram (Interpolated Kneser-Ney) và LSTM tự cài đặt, khảo sát Model Scaling Laws.

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
     python -m scripts.train_lstm --data_dir data/processed/v1_bpe8192 --preset large --ctx 512 --batch_size 8 --grad_accum 4 --lr 1e-3 --max_steps 6000 --eval_interval 500
     ```
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
