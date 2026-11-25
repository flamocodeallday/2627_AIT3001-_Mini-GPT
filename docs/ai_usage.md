# Nhật Ký Sử Dụng Công Cụ AI (AI Usage Log) — Thành Viên 3

- **Thành viên thực hiện:** Quang (Thành viên 3 — Baseline, Evaluation Harness & Model Scaling)
- **Đề tài:** Đề tài 9 — Xây dựng mô hình ngôn ngữ Mini-GPT tiếng Việt (`mini-gpt-vi`)
- **Công cụ AI sử dụng:** Gemini (Google)
- **Phạm vi ghi nhận:** Toàn bộ quá trình thiết kế kiến trúc đánh giá, cài đặt mô hình đối chứng (N-gram, Custom LSTM), kiểm thử tự động và chạy thực nghiệm trên `v0_byte` (debug) & `v1_bpe8192` (official).

---

## 1. Bảng Tổng Hợp Chi Tiết Các Lần Sử Dụng AI

| Giai đoạn / Tệp liên quan | Mục đích hỗ trợ & Nội dung yêu cầu AI | Phần chỉnh sửa, kiểm chứng & Trách nhiệm của sinh viên |
| :--- | :--- | :--- |
| **1. Hạ tầng đánh giá chuẩn**<br>`docs/eval_protocol.md`<br>`src/eval/base.py` | Thiết kế giao diện trừu tượng `BaseLanguageModel`, lớp đọc dữ liệu `DatasetBundle` (`np.memmap`) và hàm `evaluate_model` tính NLL, Perplexity, BPC kèm cơ chế kiểm tra `tokenizer_hash`. | Tự đối chiếu công thức quy đổi BPC theo tỷ lệ `total_tokens / total_chars` trong `meta.json`; chạy kiểm thử sanity check với `UniformLM` xác nhận $\text{PPL} = V$ (`257.0` trên `v0_byte` và `8192.0` trên `v1_bpe8192`) và tự động gắn nhãn `debug` / `official`. |
| **2. Baseline N-gram**<br>`src/eval/ngram.py` | Gợi ý cấu trúc lưu trữ bảng đếm ngữ cảnh và cài đặt thuật toán làm trơn **Interpolated Kneser-Ney Smoothing** đa bậc từ đầu bằng Python/NumPy. | Kiểm tra công thức chiết khấu $D$, trọng số backoff $\lambda$ và xác suất tiếp nối (continuation probability); viết unit test trong `tests/test_eval_and_ngram.py` kiểm chứng tổng phân phối xác suất $\sum_w P(w \mid h) = 1.0$. |
| **3. Baseline Custom LSTM**<br>`src/eval/lstm.py`<br>`scripts/train_lstm.py` | Hỗ trợ cấu trúc lớp `CustomLSTMCell` (tách 4 cổng $i, f, g, o$ thủ công bằng `nn.Linear`, không dùng `nn.LSTM`), khởi tạo Weight Tying và vòng lặp huấn luyện với Warmup + Cosine LR Decay. | Viết test đối chiếu sai số đầu ra giữa `CustomLSTMCell` và `torch.nn.LSTMCell` ($< 10^{-5}$); phát hiện và sửa lỗi bùng nổ logits ban đầu ($\approx 25.33$) bằng khởi tạo ma trận Embedding $\mathcal{N}(0, 0.02)$; tự điều chỉnh siêu tham số và chạy thực nghiệm. |
| **4. Cầu nối GPT & Tổng hợp**<br>`src/eval/gpt_adapter.py`<br>`scripts/run_eval_all.py` | Thiết kế lớp bọc `GPTAdapter` chuẩn hóa mô hình Transformer của Thành viên 2 vào `BaseLanguageModel`, đếm tham số tách biệt embedding/non-embedding và xuất bảng `.md`/`.csv`. | Chạy kiểm thử tích hợp với mô hình giả lập (`DummyGPT`), xác minh cơ chế chặn lệch `tokenizer_hash` hoạt động chính xác trước khi tích hợp checkpoint chính thức. |
| **5. Khảo sát Scaling Laws & Hạ tầng GPU**<br>`scripts/run_scaling.py` | Thiết kế pipeline huấn luyện 4 cấu hình tăng dần quy mô (`scale_xs` đến `scale_l`), ước tính FLOPs ($C \approx 6 N_{\text{non-emb}} D$), khớp hồi quy log-log và tối ưu bộ nhớ GPU. | Phát hiện hiện tượng Windows TDR / quá nhiệt trên GPU laptop (RTX 3050) khi chạy chuỗi dài; chủ động bổ sung Mixed Precision (`float16`), dọn cache VRAM định kỳ, lưu checkpoint từng phần và chuyển sang Google Colab (Tesla T4) để hoàn tất thực nghiệm chính thức `v1_bpe8192`. |
| **6. Tổng hợp & Phân tích kết quả**<br>`results/*`<br>Báo cáo tiến độ | Hỗ trợ tổng hợp bảng đối chiếu số liệu từ hai giai đoạn thực nghiệm và kiểm tra chéo các chỉ số NLL, Perplexity, BPC. | Tự đối chiếu toàn bộ số liệu từ file JSON/CSV gốc; tự phân tích nguyên nhân hiện tượng thưa thớt dữ liệu trên 4-gram BPE và hiệu ứng giới hạn bước huấn luyện cố định (Fixed-Step Compute Bottleneck) trên `scale_l`. |

---

## 2. Cam Kết Trung Thực Học Thuật & Tự Kiểm Chứng

1. **Hiểu và làm chủ toàn bộ mã nguồn:** Mọi thành phần cốt lõi trong `src/eval/` và `scripts/` đều được kiểm tra từng dòng, tuân thủ quy định tự cài đặt thuật toán (không gọi thư viện có sẵn cho Kneser-Ney N-gram và LSTM cell) và vượt qua toàn bộ bộ kiểm thử tự động (`pytest`).
2. **Tính xác thực của số liệu thực nghiệm:** Toàn bộ kết quả trong thư mục `results/` (`summary_v0_byte_val.*`, `summary_v1_bpe8192_val.*`, `scaling_v1_bpe8192.*`) đều được sinh trực tiếp từ quá trình chạy mã nguồn trên tập dữ liệu thực tế của nhóm với hạt giống ngẫu nhiên cố định (`seed = 42`), có thể tái tạo hoàn toàn qua `scripts/run_eval_all.py` và `scripts/run_scaling.py`.