# Nhật ký sử dụng công cụ AI (`docs/ai_usage.md`)

> Theo quy định của đề bài, mọi công cụ AI được sử dụng trong quá trình làm dự án đều phải được khai báo rõ mục đích, phạm vi và cách kiểm chứng.

## Tuần 2 — Thành viên 1 (Corpus & Tokenizer)

### Gói 1: Khung Repo & Pipeline làm sạch/chia dữ liệu (`16/11/2026`)
- **Công cụ sử dụng:** Trợ lý AI lập trình (Antigravity).
- **Mục đích:**
  - Thiết kế cấu trúc repo ban đầu (`pyproject.toml`, `requirements.txt`, `configs/data.yaml`, `.gitignore`).
  - Viết module streaming `src/data/clean.py`, `src/data/split.py`, `src/data/config.py`, script CLI `scripts/prepare_data.py`, bộ dữ liệu giả lập `fixtures/sample_vi_wiki.jsonl`, và unit test `tests/test_clean.py`, `tests/test_split.py`.
- **Kiểm chứng & nắm bắt kiến thức:**
  - Kiểm tra toàn bộ logic NFC, loại ký tự điều khiển `Cc`/`Cf`, quy tắc cắt tiêu đề Wikipedia tiếng Việt ở đầu dòng, dedup trước split để tránh leakage, và cơ chế streaming $O(1)$ RAM.
  - Chạy `pytest` trên bộ fixture ~500 KB và đối chiếu kết quả thực tế (11 passed).

### Gói 2: Giao diện Tokenizer chung, `ByteTokenizer` & Đóng gói `.bin` (`16/11/2026`)
- **Công cụ sử dụng:** Trợ lý AI lập trình (Antigravity).
- **Mục đích:**
  - Thiết kế `src/data/tokenizer_base.py` với descriptor hỗ trợ đồng thời `Tokenizer.load(path)` và `tok.load(path)`, hàm băm SHA-256 trên 1 file JSON duy nhất.
  - Cài đặt `src/data/pack.py` (`ByteTokenizer` `vocab_size=257`, `eos_id=256`, mặc định `allow_special=False`, hàm `build_bins` ghi đệm `uint16`, `MemmapDataset`), CLI `scripts/pack_data.py`, snippet PyTorch trong `README.md` và `tests/test_pack.py`.
- **Kiểm chứng & nắm bắt kiến thức:**
  - Xác nhận cơ chế mã hóa byte thô không đụng `eos_id` khi gặp chuỗi literal `<|endoftext|>`, chèn đúng 1 `eos_id` giữa các bài, đếm `n_chars` theo đúng text sau chuẩn hóa, và chạy `pytest` + CLI `--sanity-check` trên fixtures (15 passed).

### Gói 3: Tự cài đặt Regex-BPE (`src/data/bpe.py`, `18/11/2026`)
- **Công cụ sử dụng:** Trợ lý AI lập trình (Antigravity).
- **Mục đích:**
  - Thiết kế lớp `CustomBPE` kế thừa `BaseTokenizer` với regex pre-tokenization riêng cho tiếng Việt (`[\p{L}\p{M}]+` giữ liền dấu tổ hợp), train theo tần suất cụm pre-token duy nhất kết hợp chỉ mục đảo `pair_to_chunks` + max-heap cập nhật gia tăng, encode có memoization cache (giới hạn dung lượng để không tràn RAM trên 1,6 GB) và `save`/`load` 1 file JSON mã hóa Base64.
  - Viết `tests/test_bpe.py` kiểm chứng đối chiếu với vòng lặp BPE tham chiếu, ca biên tiếng Việt NFC/NFD, emoji, ký tự điều khiển, `<|endoftext|>`, và tích hợp `build_bins`.
- **Kiểm chứng & nắm bắt kiến thức:**
  - Hiểu rõ bất biến cập nhật delta trong `pair_counts` và lý do phải đẩy lại cặp vào heap khi tần suất giảm (lazy deletion heap).
  - Chạy `pytest -v` toàn bộ repo (19 passed).

### Gói 4 (Bước 1): Khảo sát cỡ từ vựng & Đối chiếu thư viện chuẩn (`18/11/2026`)
- **Công cụ sử dụng:** Trợ lý AI lập trình (Antigravity).
- **Mục đích:**
  - Viết module `src/data/vocab_eval.py` và CLI `scripts/benchmark_vocab.py` để train & lưu `custom_bpe{4096,8192,16384}.json`, tính Compression Ratio (`n_bytes/n_tokens`, `n_chars/n_tokens`), Fertility (`n_tokens/n_words`), tốc độ train/encode và chi phí tham số Embedding ($V \times d_{\text{model}}$ với $d_{\text{model}} \in \{256, 384, 512\}$, weight tying) trên ngân sách 10M và 30M.
  - Đối chiếu trực tiếp trên cùng mẫu `train` và `val` với Byte-level BPE của HuggingFace `tokenizers` (chỉ dùng benchmark theo Quyết định #5).
  - Viết `tests/test_vocab_benchmark.py` test trên `fixtures` với cỡ vocab `[300, 350]`.
- **Kiểm chứng & nắm bắt kiến thức:**
  - Hiểu rõ ý nghĩa của weight tying ($V \times d_{\text{model}}$ dùng chung cho embedding và LM head) và lý do so sánh BPC/compression ratio thay vì perplexity giữa các tokenizer khác cỡ vocab.
  - Chạy `pytest -v` toàn bộ repo (21 passed) và chạy thử CLI `benchmark_vocab` trên `fixtures`.
