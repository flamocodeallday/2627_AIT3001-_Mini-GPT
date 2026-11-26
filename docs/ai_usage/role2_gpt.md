# AI usage — Thành viên 2 (Todd): Mô hình GPT & vòng huấn luyện (`src/model/`, `src/train/`)

> Mục lục và quy ước chung: [`README.md`](README.md). Đề bài và danh sách bài tự cài: [`../role2_gpt/gpt_todo.md`](../role2_gpt/gpt_todo.md).

## Tuần 3

### Gói 1: Khung code GPT và bộ test chấm bài (`25/11/2026`)
- **Công cụ:** Claude (Anthropic).
- **AI viết (phần hạ tầng):**
  - `src/model/config.py`: `GPTConfig`, các preset và công thức `estimate_params`.
  - Trong `src/model/gpt.py`: phần `__init__`, khởi tạo trọng số và weight tying.
  - `src/train/data.py`: bọc `MemmapDataset` của TV1.
  - `src/train/checkpoint.py`.
  - `build_optimizer`.
  - Khung `src/train/train.py`: CLI, AMP, ngân sách token theo quyết định C, eval, log, resume, ghi `results/`.
  - `configs/gpt/*.yaml`.
  - `scripts/smoke_test.py`.
  - `tests/test_gpt.py` và `tests/test_train.py`.
  - `docs/role2_gpt/gpt_todo.md`.
- **Phần để trống cho sinh viên:**
  - Trong `src/model/gpt.py` và `src/model/sampling.py`: `MLP.forward`, `CausalSelfAttention.forward`, `Block.forward`, `GPT.forward`, `GPT.count_params`, `GPT.generate`, `filter_top_k`, `filter_top_p`, `sample_next_token`.
  - Trong `src/train/`: `get_lr`, `split_decay_params`, `train_step`.
  - Tất cả đều có `raise NotImplementedError` và test chấm riêng.
- **Ràng buộc:** không dùng `F.scaled_dot_product_attention` hay `nn.MultiheadAttention` trong code lõi; hai hàm này chỉ xuất hiện trong test để đối chiếu. Không chép code nanoGPT.

### Gói 2: Tự cài 12 bài, AI đóng vai trò gia sư (`26/11/2026`)
- **Công cụ:** Claude (Anthropic), dùng trong khung chat.
- **Cách dùng AI:**
  - Giải thích khái niệm: ý nghĩa của B/T/C, multi-head (B,H,T,D), causal mask và vì sao dùng −∞, thứ tự softmax rồi mới nhân V, LayerNorm, pre-LN và post-LN (Xiong et al. 2020), residual như "đường cao tốc" cho gradient, warmup, top-k và top-p.
  - Đưa khung code có ô trống để sinh viên tự điền.
  - Đọc code sinh viên viết và chỉ ra lỗi. Các lỗi đã được chỉ ra:
    - Gán module thay vì gọi module (`att = self.attn_dropout`).
    - Tra `pos_emb` bằng `idx` thay vì bằng position id.
    - Dùng `dim=0` thay cho `dim=-1` khi lọc theo hàng.
    - Dùng list comprehension để lọc top-k thay vì `masked_fill`.
- **Mức độ AI đưa sẵn code (ghi trung thực):**
  - **Bài 1 (`MLP.forward`):** AI đưa đáp án 4 dòng, kèm giải thích cách gọi module.
  - **Bài 3 (`Block.forward`):** AI đưa dòng residual cho attention; sinh viên tự viết dòng cho MLP.
  - **Bài 2, 4, 5, 6:** AI đưa khung có ô trống và tên API cần dùng. Sinh viên tự điền code, AI review.
  - **Bài 7–11** (`sample_next_token`, `generate`, `get_lr`, `split_decay_params`, `train_step`): sinh viên tự viết dựa trên gợi ý trong `gpt_todo.md`.
- **Kiểm chứng:**
  - `pytest -q tests/test_gpt.py tests/test_train.py` pass toàn bộ. Bộ test gồm: attention khớp `F.scaled_dot_product_attention` tới sai số 1e-5, mask nhân quả, loss ban đầu ≈ ln V, overfit 1 batch, grad accumulation tương đương full batch, `count_params` khớp `estimate_params`.
  - `python -m scripts.smoke_test` in `SMOKE TEST PASSED`. Smoke test gồm train 60 step, kiểm tra khóa checkpoint, load lại model, eval và sinh văn bản.
  - `configs/gpt/tiny_debug.yaml` (v0_byte, CPU, 200 step):
    - Loss bắt đầu 5,58 ≈ ln 257 và giảm xuống 1,91.
    - val 1,958 (ppl ≈ 7,1).
    - LR warmup 20 step lên 3e-3, rồi giảm theo cosine về đúng `min_lr` = 3e-4 ở step cuối.
  - Phần kiểm tra `GPTAdapter` trong smoke test chưa chạy, vì nhánh này chưa có `src/eval` của TV3. Sẽ kiểm tra khi ghép nhánh.

### Gói 3: Sắp xếp tài liệu, commit và merge (`26/11/2026`)
- **Công cụ:** Claude (Anthropic), thao tác trực tiếp trên repo.
- **AI làm:**
  - Tách `docs/ai_usage.md` thành mỗi thành viên một file trong `docs/ai_usage/`.
  - Chuyển `gpt_todo.md` vào `docs/role2_gpt/` và cập nhật các đường dẫn tham chiếu.
  - Commit nhánh `feat/gpt-model`, merge vào `main`.
  - Chuẩn bị nhánh ghép với `feat/tv3-eval`.
- **Kiểm chứng:** sinh viên rà lại diff và chạy lại `pytest` cùng smoke test trên máy Windows.
