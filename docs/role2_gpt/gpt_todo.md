# Thành viên 2: GPT, bối cảnh và bài tập (nhánh `feat/gpt-model`)

> Đọc mục 1–3 trước (khoảng 10 phút), rồi làm các bài ở mục 4 theo đúng thứ tự. Mỗi bài có lệnh test riêng.
> Gợi ý được giấu trong `<details>` và chia 3 cấp. Chỉ mở khi bí, và mở từ cấp 1.

## 1. Bối cảnh

- **Nhánh:** `feat/gpt-model` tách từ `feat/tv1-data-pipeline`, nên có sẵn `src/data/` của TV1. Khi xong: push nhánh này (TV1 + TV2) → merge vào `main` → merge nhánh làm lại của TV3 vào sau.
- **Dữ liệu:** `data/processed/v0_byte/` (vocab 257, chỉ dùng để debug) và `data/processed/v1_bpe8192/` (vocab 8192, `eos_id` 256, train 264M token). Code luôn đọc `meta.json`, không hard-code.
- **Quyết định đã chốt (25/11):**
  - **A:** model chính `main_30m` = 8 layer × 512, 8 head, weight tying → khoảng 25,2M non-emb và 29,6M tổng.
  - **B:** `block_size` = `ctx` đánh giá = 512.
  - **C:** ngân sách D = 20 × N_non-emb token cho mọi cỡ (model chính khoảng 504M token, khoảng 1,9 epoch). Mỗi run warmup rồi cosine về `min_lr` đúng ở cuối ngân sách của nó. Model chính là điểm lớn nhất của đường scaling.
- **Hợp đồng với TV3 (`GPTAdapter`):**
  - `forward(idx, targets=None) -> (logits, loss)`, trong đó `logits` có shape (B, T, V) cho **mọi** vị trí.
  - Checkpoint gồm `{state_dict, config, tokenizer_meta, step, val_loss}`.
  - `count_params(exclude_embeddings)`, với non-emb loại cả `tok_emb` và `pos_emb`.
  - `GPT.from_config_dict` / `gpt_model_builder` dùng làm `model_builder`.
- **Luật của môn:** tự viết attention, block và vòng train. Không dùng `F.scaled_dot_product_attention` hay `nn.MultiheadAttention` trong code lõi; chúng chỉ xuất hiện trong test để đối chiếu. Không chép code nanoGPT. Ghi việc dùng AI vào `docs/ai_usage/role2_gpt.md`.

## 2. Bản đồ file

| File | Ai viết | Nội dung |
|---|---|---|
| `src/model/config.py` | AI | `GPTConfig`, `PRESETS` (thang scaling, `main_30m`, 2 biến thể ablation), `estimate_params` (công thức) |
| `src/model/gpt.py` | AI khung, **Todd viết lõi** | `MLP`, `CausalSelfAttention`, `Block`, `GPT`. Các `__init__`, khởi tạo trọng số và weight tying đã viết sẵn |
| `src/model/sampling.py` | **Todd** | `filter_top_k`, `filter_top_p`, `sample_next_token` |
| `src/train/data.py` | AI | `load_meta`, kiểm tra hash, `TokenBatcher` (bọc `MemmapDataset` của TV1) |
| `src/train/schedule.py` | **Todd** | `get_lr` |
| `src/train/optim.py` | AI + **Todd** | `build_optimizer` (AI); `split_decay_params` (Todd) |
| `src/train/checkpoint.py` | AI | save (ghi atomic) / load / build model / builder cho adapter |
| `src/train/train.py` | AI + **Todd** | CLI, AMP, ngân sách, eval, log, resume, ghi `results/` (AI); `train_step` (Todd) |
| `configs/gpt/*.yaml` | AI | `tiny_debug` (v0, CPU), `main_30m`, `scaling` (cùng thiết lập train với `main_30m`) |
| `scripts/smoke_test.py` | AI | train → checkpoint → load → đánh giá (kèm `GPTAdapter` nếu có) → sinh văn bản |
| `tests/test_gpt.py`, `tests/test_train.py` | AI | chấm bài. Các test `test_scaffold_*` phải qua ngay từ đầu |

## 3. Luồng dữ liệu cần nắm (đáp án cho câu "tensor đi qua những shape nào")

```
idx (B,T) int64
 ├─ tok_emb ─► (B,T,C)
 └─ pos_emb(arange(T)) ─► (T,C)  ── cộng broadcast ─► x (B,T,C) ─► dropout
x ─► [Block] × n_layer:
        x = x + Attn(LN(x))      Attn: qkv_proj (B,T,3C) → q,k,v (B,H,T,D) → scores (B,H,T,T)
                                       → mask tương lai → softmax → @v (B,H,T,D) → gộp (B,T,C) → out_proj
        x = x + MLP(LN(x))       MLP: C → 4C → GELU → C
x ─► norm_f ─► lm_head (dùng chung ma trận với tok_emb) ─► logits (B,T,V) ─► cross-entropy với targets (B,T)
```

Vòng train cho mỗi optimizer step: đặt LR theo `get_lr`, gọi `train_step`, gồm `grad_accum_steps` micro-batch, mỗi micro-batch (32, 512).
- Tokens/step = 32 × 4 × 512 = 65.536.
- Model chính chạy khoảng 7.690 step.

## 4. Bài tập (làm theo thứ tự)

Lệnh kiểm tra tổng: `pytest -q tests/test_gpt.py tests/test_train.py`. Lúc bắt đầu: 8 passed, 1 skipped, 37 failed, và tất cả các test failed đều do `NotImplementedError`.

### Bài 1: `MLP.forward` (khởi động, khoảng 5 phút). `pytest tests/test_gpt.py -k mlp`
Mục tiêu: C → 4C → GELU → C → dropout.
**Vấn đáp:** Vì sao gọi là "position-wise"? Vì sao mở rộng ra 4C?
<details><summary>Gợi ý 1</summary>Mọi module cần dùng đã được tạo sẵn trong <code>__init__</code>, chỉ cần gọi đúng thứ tự.</details>

### Bài 2: `CausalSelfAttention.forward` (bài quan trọng nhất). `pytest tests/test_gpt.py -k attention`
Mục tiêu: multi-head masked self-attention, khớp với bản tham chiếu tới sai số 1e-5. Thứ tự tách q, k, v và cách tách head được ghi trong docstring của lớp.
**Vấn đáp:** Mask đi vào đâu trong công thức và vì sao dùng −∞? Vì sao chia cho √D? Bỏ mask thì loss khi train sẽ ra sao?
<details><summary>Gợi ý 1</summary>Viết shape cạnh mỗi dòng code. Các shape lần lượt phải là (B,T,3C), rồi 3×(B,T,C), rồi 3×(B,H,T,D), rồi (B,H,T,T), rồi (B,H,T,D), rồi (B,T,C).</details>
<details><summary>Gợi ý 2</summary>Các hàm cần dùng: <code>split</code> theo chiều cuối, <code>view</code> + <code>transpose(1, 2)</code>, <code>@</code> với <code>k.transpose(-2, -1)</code>, <code>masked_fill</code> với mask đã cắt <code>[:, :, :T, :T]</code> (mask đang là True ở chỗ ĐƯỢC nhìn), <code>F.softmax(dim=-1)</code>. Trước khi <code>view</code> lại về (B,T,C) cần <code>contiguous()</code>.</details>
<details><summary>Gợi ý 3</summary>Các bước bằng lời: (1) chiếu x thành qkv rồi cắt thành 3 phần, mỗi phần rộng C; (2) đổi mỗi phần sang (B,H,T,D); (3) điểm = q nhân k chuyển vị, chia căn D; (4) chỗ nào mask là False thì gán −∞; (5) softmax theo chiều key, dropout; (6) nhân với v; (7) đưa chiều head về cạnh D rồi gộp lại thành C; (8) out_proj, rồi dropout.</details>

### Bài 3: `Block.forward`. `pytest tests/test_gpt.py -k block`
Mục tiêu: pre-norm, hai nhánh residual.
**Vấn đáp:** Residual giúp gradient đi qua mạng sâu như thế nào? Pre-LN khác post-LN ở điểm nào (Xiong et al. 2020)?
<details><summary>Gợi ý 1</summary>Hàm chỉ có 2 dòng. Ở mỗi dòng, x được cộng thêm (+=) đầu ra của một sub-layer, và sub-layer đó nhận đầu vào là x đã chuẩn hóa.</details>

### Bài 4: `GPT.forward`. `pytest tests/test_gpt.py -k forward`
Mục tiêu: embedding + position → các block → `norm_f` → `lm_head`, có tính loss. T > `block_size` thì raise `ValueError`.
**Vấn đáp:** Transformer không có RNN, vậy làm sao biết thứ tự từ? Vì sao loss ban đầu ≈ ln V? Overfit được 1 batch chứng minh điều gì?
<details><summary>Gợi ý 1</summary>Position id là <code>torch.arange(T)</code>, nhớ đặt đúng device. Phép cộng (B,T,C) + (T,C) tự broadcast.</details>
<details><summary>Gợi ý 2</summary><code>F.cross_entropy</code> cần logits dạng (N, V) và targets dạng (N,), nên phải gộp B và T lại.</details>

### Bài 5: `GPT.count_params`. `pytest tests/test_gpt.py -k count_params`
**Vấn đáp:** Weight tying tiết kiệm bao nhiêu tham số với 8×512 và V=8192? Vì sao Kaplan đếm non-embedding?
<details><summary>Gợi ý 1</summary><code>self.parameters()</code> đã tự bỏ trùng tham số dùng chung. Nếu tự gom bằng <code>id(p)</code> thì càng chắc chắn. Muốn biết lm_head có bị tie không, kiểm tra <code>is</code> giữa hai tensor weight.</details>

### Bài 6: `filter_top_k` (6a) và `filter_top_p` (6b). `pytest tests/test_gpt.py -k "top_k or top_p"`
location: src/model/sampling.py
**Vấn đáp:** top-k khác top-p ở điểm nào khi phân phối nhọn và khi phân phối phẳng?
<details><summary>Gợi ý 1</summary>top-k: tìm giá trị lớn thứ k của từng hàng (<code>torch.topk</code>), mọi phần tử nhỏ hơn giá trị đó gán −∞. Không sửa input tại chỗ.</details>
<details><summary>Gợi ý 2</summary>top-p: sắp xếp giảm dần (<code>torch.sort</code>, giữ lại chỉ số), tính softmax rồi <code>cumsum</code>. "Khối lượng các token đứng trước" bằng cumsum trừ xác suất của chính nó. Quyết định giữ hay bỏ trên thứ tự đã sắp, rồi đưa về thứ tự gốc bằng <code>scatter</code>.</details>

### Bài 7: `sample_next_token`. `pytest tests/test_gpt.py -k sample`
location: src/model/sampling.py
<details><summary>Gợi ý 1</summary>Thứ tự: temperature == 0 thì lấy argmax (giữ chiều, ra (B,1)); nếu không thì chia cho temperature, áp top-k, áp top-p, softmax, rồi <code>torch.multinomial(probs, 1, generator=generator)</code>.</details>

### Bài 8: `GPT.generate`. `pytest tests/test_gpt.py -k generate`
location: src/model/gpt.py
**Vấn đáp:** Vì sao phải cắt ngữ cảnh về `block_size`? Vì sao in ra từng token có thể thấy "�" (gợi ý: UTF-8, xem `HUONG_DAN_TV1.md`)?
<details><summary>Gợi ý 1</summary>Mỗi vòng lặp: lấy <code>idx[:, -block_size:]</code>, forward, lấy logits của vị trí cuối <code>[:, -1, :]</code>, sample, rồi <code>torch.cat</code> vào idx. Trước vòng lặp lưu lại <code>self.training</code>, gọi <code>eval()</code>; sau vòng lặp trả lại mode cũ.</details>

### Bài 9: `get_lr`. `pytest tests/test_train.py -k get_lr`
location: src/train/schedule.py
**Vấn đáp:** Vì sao cần warmup với Adam? Vì sao cosine phải kết thúc đúng ở cuối ngân sách (liên hệ lỗi `scale_l` của TV3)?
<details><summary>Gợi ý 1</summary>Chia 3 trường hợp: đang warmup, đã quá total_steps, còn lại là cosine. Tiến độ r = (step − warmup) / (total − warmup) đi từ 0 đến 1. Hệ số cosine 0,5·(1 + cos(π·r)) đi từ 1 về 0.</details>

### Bài 10: `split_decay_params`. `pytest tests/test_train.py -k split_decay`
location: src/train/optim.py
**Vấn đáp:** Vì sao không weight decay bias và LayerNorm?
<details><summary>Gợi ý 1</summary>Chỉ cần dựa vào <code>p.dim()</code>. Duyệt qua <code>model.parameters()</code> (đã tự bỏ trùng tham số tie) và chỉ lấy những tham số <code>requires_grad</code>.</details>

### Bài 11: `train_step`. `pytest tests/test_train.py -k train_step`
location: src/train/train.py
Mục tiêu: gradient accumulation đúng, clip sau khi unscale, trả về loss trung bình và grad norm trước khi clip.
**Vấn đáp:** Vì sao phải chia loss cho `grad_accum_steps`? Vì sao phải `unscale_` trước khi clip khi dùng fp16?
<details><summary>Gợi ý 1</summary>Docstring đã liệt kê đúng 4 bước. <code>clip_grad_norm_</code> trả về norm trước khi clip. Loss để ghi log lấy bằng <code>.item()</code> trước khi chia.</details>

### Bài 12: chạy đầu-cuối
```powershell
pytest -q                                   # toàn repo
python -m scripts.smoke_test                # khoảng 1 phút trên CPU, phải in SMOKE TEST PASSED
python -m src.train.train --config configs/gpt/tiny_debug.yaml   # xem loss giảm, tensorboard --logdir logs/tb
```
Sau đó lên Colab: `python -m src.train.train --config configs/gpt/main_30m.yaml --device cuda`. Khi bị ngắt phiên: thêm `--resume checkpoints/<run_id>/last.pt`.

## 5. Theo dõi tiến độ

| Bài | Trạng thái | Ngày xong |
|---|---|---|
| 1 MLP | ✅ | 26/11 |
| 2 Attention | ✅ | 26/11 |
| 3 Block | ✅ | 26/11 |
| 4 GPT.forward | ✅ | 26/11 |
| 5 count_params | ✅ | 26/11 |
| 6 top-k / top-p | ✅ | 26/11 |
| 7 sample_next_token | ✅ | 26/11 |
| 8 generate | ✅ | 26/11 |
| 9 get_lr | ✅ | 26/11 |
| 10 split_decay_params | ✅ | 26/11 |
| 11 train_step | ✅ | 26/11 |
| 12 smoke test + train tiny | ✅ | 26/11 |
