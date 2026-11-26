# Nhật ký sử dụng công cụ AI

> Theo quy định của đề bài, mọi công cụ AI được dùng trong dự án đều phải khai báo rõ: công cụ gì, dùng vào việc gì, phần nào do AI viết, phần nào sinh viên tự viết, và sinh viên kiểm chứng ra sao.

## Mỗi thành viên một file

| Thành viên | Phạm vi | File |
|---|---|---|
| TV1 | Corpus & tokenizer (`src/data/`) | [`role1_data.md`](role1_data.md) |
| TV2 (Todd) | Mô hình GPT & vòng huấn luyện (`src/model/`, `src/train/`) | [`role2_gpt.md`](role2_gpt.md) |
| TV3 (Quang) | Baseline, đánh giá & scaling (`src/eval/`) | [`role3_eval.md`](role3_eval.md) |

## Quy ước ghi

- Mỗi đợt làm việc có AI hỗ trợ là một mục `### Gói N: <nội dung> (dd/mm/yyyy)`.
- Mỗi gói có đủ 4 dòng:
  - **Công cụ**
  - **AI viết / AI hỗ trợ**
  - **Sinh viên tự làm**
  - **Kiểm chứng**
- Chỉ sửa file của mình. Như vậy khi merge nhánh sẽ không bị xung đột ở file này.
