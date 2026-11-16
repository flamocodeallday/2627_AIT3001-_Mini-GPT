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
