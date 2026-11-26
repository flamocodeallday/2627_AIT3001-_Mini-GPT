# AI usage — Thành viên 2 (Todd): Mô hình GPT & vòng huấn luyện (`src/model/`, `src/train/`)

> Mục lục và quy ước chung: [`README.md`](README.md).
> Công cụ: Claude (Anthropic). Thời gian: 25–26/11/2026 (tuần 3).

## 1. AI dựng gì (phần hạ tầng)

AI viết toàn bộ phần không phải lõi của mô hình, để sinh viên tập trung vào attention, block và vòng train theo đúng luật của môn:

- **Cấu hình:** `GPTConfig`, các preset (thang scaling, `main_30m`, 2 biến thể ablation), công thức `estimate_params`.
- **Khung `GPT`:** các `__init__`, khởi tạo trọng số N(0; 0,02), init thu nhỏ cho các phép chiếu ghi vào residual stream, weight tying.
- **Dữ liệu:** `TokenBatcher` bọc `MemmapDataset` của TV1, kiểm tra `tokenizer_hash`.
- **Huấn luyện:** `build_optimizer`, checkpoint ghi atomic theo hợp đồng với `GPTAdapter`, khung `train.py` (CLI, AMP, ngân sách D = 20 × N_non-emb, eval, log, resume, ghi `results/`).
- **Cấu hình chạy:** `configs/gpt/*.yaml`.
- **Kiểm thử:** `scripts/smoke_test.py`, `tests/test_gpt.py`, `tests/test_train.py`. Hai file test này là bộ chấm cho các hàm sinh viên tự cài.
- **Repo:** sắp xếp `docs/` theo từng thành viên; commit; merge vào `main`; gộp xung đột khi ghép nhánh `feat/tv3-eval`.

Ràng buộc: trong code lõi không dùng `F.scaled_dot_product_attention` hay `nn.MultiheadAttention`. Hai hàm này chỉ xuất hiện trong test để đối chiếu.

## 2. Sinh viên tự code gì

12 hàm lõi, mỗi hàm có test chấm riêng:

| Nhóm | Hàm |
|---|---|
| Mô hình | `MLP.forward`, `CausalSelfAttention.forward`, `Block.forward`, `GPT.forward`, `GPT.count_params`, `GPT.generate` |
| Sinh văn bản | `filter_top_k`, `filter_top_p`, `sample_next_token` |
| Huấn luyện | `get_lr`, `split_decay_params`, `train_step` |

AI đóng vai trò gia sư. Nó giải thích khái niệm, đưa khung code có ô trống, đọc code sinh viên viết và chỉ ra lỗi. Mức độ AI đưa sẵn code, ghi trung thực:

- **`MLP.forward`:** AI đưa đáp án (4 dòng).
- **`Block.forward`:** AI đưa dòng residual của attention; sinh viên tự viết dòng của MLP.
- **Attention, `GPT.forward`, `count_params`, top-k, top-p:** sinh viên tự điền vào khung có ô trống, sau đó AI review.
- **`sample_next_token`, `generate`, `get_lr`, `split_decay_params`, `train_step`:** sinh viên tự viết.

## 3. Những gì đã hiểu và insight rút ra

- **Ba chiều B, T, C.** Mọi tensor trong mô hình đều có dạng (B, T, C).
  - B là số chuỗi trong batch.
  - T là số vị trí trong một chuỗi, tối đa `block_size`.
  - C là số chiều biểu diễn của mỗi token.
  - Thói quen ghi shape ở cuối mỗi dòng code là cách nhanh nhất để không bị lạc.
- **MLP là "position-wise".** `nn.Linear` chỉ tác động lên chiều cuối, nên mọi vị trí đi qua cùng một ma trận và không trao đổi thông tin với nhau. Trộn thông tin giữa các token là việc của attention. MLP chiếm khoảng 2/3 số tham số của một block: 2 × 512 × 2048 × 8 layer ≈ 16,8M trên tổng 25,2M.
- **Multi-head.** C được chia thành H head, mỗi head D = C/H chiều. Phải `view` thành (B,T,H,D) rồi mới `transpose` sang (B,H,T,D). Lý do: phép `@` coi mọi chiều trước hai chiều cuối là chiều batch, nên mỗi cặp (b, h) cho ra một ma trận điểm (T,T) riêng. Khi gộp head lại cần `contiguous()` trước `view`.
- **Causal mask dùng −∞ chứ không dùng 0.** softmax tính exp: exp(0) = 1 vẫn cho trọng số, còn exp(−∞) = 0 mới loại hẳn token tương lai. Mask là buffer `persistent=False`: nó đi theo model sang GPU nhưng không được lưu vào checkpoint, vì có thể tạo lại từ `block_size`.
- **Pre-LN và residual.**
  - Với pre-LN, đường x chạy thẳng từ đầu vào tới đầu ra: x_cuối = x_đầu + Σ fᵢ.
  - Đạo hàm của x + f(x) là I + f′. Thành phần I giúp gradient về tới các lớp đầu mà không bị triệt tiêu.
  - Post-LN đặt norm ngay trên đường chính, nên lúc khởi tạo gradient bất ổn và bắt buộc phải có warmup (Xiong et al. 2020).
- **Warmup.** Lúc đầu trọng số là ngẫu nhiên và Adam chưa ước lượng được độ lớn của gradient, nên LR cần tăng dần từ gần 0.
- **Loss ban đầu ≈ ln V.** Init N(0; 0,02) cho logits gần bằng 0, tức phân phối gần đều trên V token. Đã kiểm chứng: v0_byte bắt đầu ở 5,58, so với ln 257 = 5,55. Đây là phép kiểm tra nhanh cho thấy `forward`, init và weight tying đều đúng.
- **Weight tying** tiết kiệm V × C ≈ 4,2M tham số (với V = 8192, C = 512). `self.parameters()` tự bỏ tham số dùng chung, nên ma trận tie chỉ được đếm một lần. Muốn biết lm_head có đang tie không thì kiểm tra bằng `is`.
- **Top-k và top-p.**
  - Top-k giữ số lượng token cố định.
  - Top-p giữ số token thay đổi theo hình dạng phân phối: phân phối nhọn thì giữ ít, phân phối phẳng thì giữ nhiều.
  - Top-p được tính hoàn toàn bằng tensor: sort, rồi cumsum, rồi tính "khối lượng xác suất đứng trước" bằng cumsum trừ xác suất của chính token đó, rồi tạo mask. Cuối cùng dùng `scatter` để đưa về thứ tự token gốc.
- **Cosine schedule kết thúc đúng ở cuối ngân sách.** Đã kiểm chứng: `tiny_debug` chạm `min_lr` = 3e-4 đúng ở step cuối.

## 4. Những ý hiểu chưa đúng đã được sửa

| Hiểu ban đầu | Hiểu đúng |
|---|---|
| T là "token" | T là **số vị trí** (độ dài chuỗi); mỗi vị trí chứa một token |
| Tách q, k, v bằng `x[B,T,1]`, `x[B,T,2]`, `x[B,T,3]` | Cắt chiều cuối dài 3C thành 3 khúc, mỗi khúc dài C: `qkv.split(C, dim=-1)` |
| Nhân V trước rồi mới softmax | **softmax(QKᵀ/√D) trước** để ra trọng số, rồi mới nhân V để ra trung bình có trọng số |
| LayerNorm "đưa về phân phối chuẩn" | Với từng token: trừ trung bình, chia độ lệch chuẩn trên C chiều, rồi nhân γ, cộng β. Dữ liệu không trở thành phân phối Gaussian |
| Post-LN là norm **trước** khi cộng residual | Post-LN: `x = LN(x + f(x))`, tức norm **sau** khi cộng. Pre-LN: `x = x + f(LN(x))` |
| Residual "kéo nghĩa từ về ngữ cảnh câu" | Trộn ngữ cảnh là việc của **attention**. Residual giữ lại thông tin cũ (sub-layer chỉ cộng thêm một phần cập nhật) và tạo đường đi cho gradient |

Các lỗi code đã gặp và rút kinh nghiệm:

| Lỗi | Cách đúng |
|---|---|
| `att = self.attn_dropout` (gán module) | Phải gọi module trên tensor: `self.attn_dropout(att)`. Nếu thấy lỗi có chữ `'Dropout'` hay `'Linear'` thì gần như chắc chắn là quên `(...)` |
| `x += ...` trong `Block` | Phép sửa tại chỗ phá autograd, vì LayerNorm đã lưu x để tính backward. Dùng `x = x + ...` |
| `pos_emb(idx)` | Phải tra bằng position id `arange(T)`. Tra bằng id token sẽ báo `index out of range`; trên GPU lỗi này hiện ra dưới dạng `device-side assert`. Đặt hai tên biến khác nhau (`pos_ids`) để tránh gõ nhầm |
| `topk(..., dim=0)` | Thao tác trên từng hàng thì dùng `dim=-1`; `dim=0` là so sánh giữa các mẫu trong batch |
| Dùng list comprehension / vòng lặp Python trên tensor | Dùng phép toán trên cả tensor (`masked_fill`). Ngưỡng của top-k lấy bằng `values[:, -1:]`, giữ chiều để broadcast được |

## 5. Kiểm chứng

- **Test chấm:** `pytest -q tests/test_gpt.py tests/test_train.py` pass toàn bộ. Các phép kiểm tra gồm:
  - Attention khớp `F.scaled_dot_product_attention` tới sai số 1e-5.
  - Mask nhân quả.
  - Loss ban đầu ≈ ln V.
  - Overfit được 1 batch.
  - Grad accumulation tương đương full batch.
  - `count_params` khớp `estimate_params`.
- **Smoke test:** `python -m scripts.smoke_test` in `SMOKE TEST PASSED` (train, checkpoint, load lại, eval, sinh văn bản).
- **`tiny_debug`** (v0_byte, CPU, 200 step): loss giảm từ 5,58 xuống 1,91; val 1,958; lịch LR đúng thiết kế.
- **Ghép với TV3:** kiểm tra `GPTAdapter` sẽ được thực hiện trong smoke test sau khi ghép nhánh `feat/tv3-eval`.
