# Quy chuẩn Đánh giá Mô hình Ngôn ngữ (Evaluation Protocol v1)

## 1. Dữ liệu và Tính toàn vẹn
- **Nguồn dữ liệu:** Đọc trực tiếp từ thư mục phiên bản (`data/processed/<version>/`), gồm `train.bin`, `val.bin`, `test.bin` (mảng `uint16`) và `meta.json`. Không hard-code `vocab_size` hay `eos_id`.
- **Kiểm tra phiên bản (Hash Check):** Mọi mô hình/adapter trước khi đánh giá phải đối chiếu `tokenizer_hash` và `vocab_size` với `meta.json`. Nếu sai lệch, hệ thống báo lỗi ngay lập tức.
- **Đóng băng tập Test:** Tập `test.bin` chỉ chạy **một lần duy nhất** ở cuối dự án để báo cáo kết quả cuối cùng; mọi tinh chỉnh siêu tham số thực hiện trên `val.bin`.
- **Chính sách dùng tập con & Nhãn dữ liệu:**
  - Mọi kết quả **official** bắt buộc đánh giá trên **toàn bộ tập `val.bin`** (khi đó `n_chars` lấy chính xác 100% từ `meta["splits"][split]["n_chars"]`).
  - Mọi kết quả chạy trên `v0_byte` hoặc trên tập con (`max_tokens < len(split)`) tự động gắn cờ `"tag": "debug"` và **không đưa vào bảng kết quả chính**.

## 2. Giao diện Mô hình & Quy ước Chia đoạn
Mọi mô hình (N-gram, LSTM, GPT) cài đặt phương thức:
`nll(token_ids: np.ndarray, ctx: int) -> tuple[float, int]`
- **Chia đoạn không chồng lấn (Non-overlapping Chunks):** Chuỗi `token_ids` được cắt thành các đoạn liên tiếp không chồng lấn độ dài `ctx` (phần đuôi `< ctx` được đánh giá riêng để không bỏ sót token). Trạng thái/ngữ cảnh được **reset ở đầu mỗi đoạn**, áp dụng thống nhất cho cả **LSTM** và **GPT**.
- **Độ dài ngữ cảnh chuẩn (`ctx = 512`, Quyết định B):**
  - `ctx = 512` bằng đúng `block_size` của mô hình GPT chính, dùng chung cho cả **GPT** và **LSTM**.
  - Mọi kết quả cũ chạy ở `ctx = 256` phải được chạy lại ở `ctx = 512`.
  - Với **N-gram**, mô hình dùng tối đa $n - 1$ token liền trước; bảng kết quả phải ghi rõ bậc $n$ và **số token train đã fit**.
- **Đầu ra:** `(total_nll, num_tokens)` trong đó `total_nll` là tổng Negative Log-Likelihood theo cơ số tự nhiên $e$ (đơn vị *nats*).

## 3. Chỉ số Đánh giá
1. **Perplexity theo token (Chính — khi cùng tokenizer & cùng split):**
   $$\text{PPL} = \exp\left(\frac{\text{total\_nll}}{\text{num\_tokens}}\right)$$
2. **Bits-Per-Character — BPC (Phụ — dùng để đối chiếu chéo khi đổi tokenizer):**
   $$\text{BPC} = \frac{\text{total\_nll}}{n\_chars \times \ln(2)}$$

## 4. Quy ước Đếm Tham số
- **GPT (Kaplan et al., 2020):** Báo cáo cả `total_params` và `non_embedding_params` (loại trừ cả **Token Embedding** và **Positional Embedding**, có tính weight tying).
- **LSTM:** Báo cáo cả `total_params` và `non_embedding_params` (loại trừ Token Embedding, có tính weight tying). Baseline gồm LSTM nhỏ (`lstm_l2_h256`, ~1.05M non-emb) và LSTM lớn (`lstm_l3_h1024`, ~25.2M non-emb, tương đương GPT chính).
- **N-gram:** Báo cáo "số tham số hiệu dụng" bằng tổng số n-gram duy nhất được lưu trong bảng đếm.

## 5. Luật Ngân sách Model Scaling (Quyết định C)
- **Kiến trúc chính:** Khảo sát scaling trên **GPT** (lấy preset từ `src/model/config.py`: `PRESETS`, `GPTConfig`).
- **Luật ngân sách:** Mỗi cỡ mô hình được huấn luyện với số token:
  $$D = 20 \times N_{\text{non-emb}}$$
  tại `ctx = 512`.
- **Lịch Learning Rate:** Mỗi run có warmup tuyến tính (5%), sau đó cosine decay giảm về `min_lr` **đúng tại cuối ngân sách của run đó**.
- **Điểm lớn nhất:** Mô hình GPT chính (~25M non-emb, ~500M token) là điểm lớn nhất trên đường scaling.
- **Đường so sánh phụ:** Scaling trên **LSTM** (`--arch lstm`) được giữ lại làm đường đối chiếu phụ và phải ghi nhãn rõ.

## 6. Yêu cầu Tái lập (Reproducibility)
Mỗi file JSON kết quả bắt buộc ghi đầy đủ: toàn bộ siêu tham số huấn luyện (`train_config`), số token đã train (`train_tokens`), số token val đã đánh giá (`num_tokens`, `is_full_split`), phần cứng (`hardware`), thời gian chạy (`eval_time_sec`, `timestamp`), và `git_commit` kèm cờ `git_dirty`. Kết quả official chỉ sinh từ commit sạch bằng script, không chỉnh sửa tay.
