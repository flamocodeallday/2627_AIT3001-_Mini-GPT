# Nhật ký Sử dụng AI (AI Usage Log)

---

## Thành viên 3 (Quang) — Baseline, Đánh giá & Scaling (`src/eval/`, `scripts/`, `tests/`)

- **Công cụ sử dụng:** Antigravity / Coding Agent.
- **Phạm vi hỗ trợ:**
  - Thiết kế và rà soát cấu trúc module đánh giá (`BaseLanguageModel`, `DatasetBundle`, `evaluate_model`) nhằm bảo đảm đầy đủ metadata tái lập (`train_config`, `train_tokens`, `hardware`, `git_commit`, `git_dirty`).
  - Tối ưu hóa đếm N-gram bằng NumPy bit-packing (`uint64`) và kiểm tra tính chuẩn hóa xác suất của Interpolated Kneser-Ney (`_kn_prob`).
  - Viết unit test chạy độc lập trên clone sạch (`tests/test_eval_and_ngram.py`, `tests/test_lstm.py`, `tests/test_gpt_adapter.py` với `DummyGPT`).
  - Chuẩn hóa script huấn luyện LSTM (`lstm_l2_h256`, `lstm_l3_h1024`), script đánh giá tổng hợp (`run_eval_all.py`), script scaling GPT theo luật ngân sách $D = 20 \times N_{\text{non-emb}}$ (`run_scaling.py`), và pipeline đa nền tảng (`run_pipeline.py`).
  - Soạn thảo tài liệu `docs/eval_protocol.md` và phần hướng dẫn trong `README.md`.
- **Kiểm chứng của sinh viên:** Toàn bộ mã nguồn được sinh viên rà soát, kiểm thử bằng `pytest -q`, và chạy thực nghiệm trên tập dữ liệu `v1_bpe8192`.
