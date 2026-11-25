# Quy chuẩn Đánh giá Mô hình Ngôn ngữ (Evaluation Protocol v0)

## 1. Dữ liệu và Tính toàn vẹn
- **Nguồn dữ liệu:** Đọc trực tiếp từ thư mục phiên bản (`data/processed/<version>/`), gồm `train.bin`, `val.bin`, `test.bin` (mảng `uint16`) và `meta.json`.
- **Kiểm tra phiên bản (Hash Check):** Mọi mô hình/adapter trước khi đánh giá phải đối chiếu `tokenizer_hash` và `vocab_size` với `meta.json`. Nếu sai lệch, hệ thống báo lỗi ngay lập tức.
- **Đóng băng tập Test:** Tập `test.bin` chỉ dùng để báo cáo kết quả cuối cùng; mọi tinh chỉnh siêu tham số thực hiện trên `val.bin`.
- **Nhãn dữ liệu thử nghiệm:** Mọi kết quả chạy trên `v0_byte` bắt buộc gắn cờ `"tag": "debug"` và không đưa vào bảng kết quả chính của báo cáo.

## 2. Giao diện Mô hình Chung
Mọi mô hình (N-gram, LSTM, GPT) phải cài đặt phương thức:
`nll(token_ids: np.ndarray, ctx: int) -> tuple[float, int]`
- **Đầu vào:** Mảng 1 chiều `token_ids` và độ dài ngữ cảnh `ctx`.
- **Đầu ra:** `(total_nll, num_tokens)` trong đó `total_nll` là tổng Negative Log-Likelihood tính theo cơ số tự nhiên $e$ (đơn vị *nats*), và `num_tokens` là tổng số token thực sự được tính loss.

## 3. Chỉ số Đánh giá
1. **Perplexity theo token (Chính — khi cùng tokenizer & cùng test set):**
   $$\text{PPL} = \exp\left(\frac{\text{total\_nll}}{\text{num\_tokens}}\right)$$
2. **Bits-Per-Character — BPC (Phụ — dùng để đối chiếu chéo khi đổi tokenizer):**
   $$\text{BPC} = \frac{\text{total\_nll}}{n\_chars \times \ln(2)}$$
   Trong đó `n_chars` lấy trực tiếp từ `meta["splits"][split]["n_chars"]`.

## 4. Quy ước Đếm Tham số
- **GPT:** Báo cáo cả `total_params` và `non_embedding_params` (loại trừ Token Embedding và Positional Embedding theo Kaplan et al., 2020; có tính weight tying).
- **LSTM:** Đếm toàn bộ tham số khả huấn của Embedding, các cổng LSTM và Linear head (có tính weight tying nếu áp dụng).
- **N-gram:** Báo cáo "số tham số hiệu dụng" bằng tổng số n-gram duy nhất được lưu trong bảng đếm.