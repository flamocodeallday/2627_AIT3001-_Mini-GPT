# Tài liệu dự án

```
docs/
├── README.md            # file này: mục lục
├── eval_protocol.md     # (chung) quy chuẩn đánh giá, đếm tham số, ngân sách scaling. Do TV3 soạn
└── ai_usage/            # nhật ký dùng AI, mỗi thành viên một file
    ├── README.md
    ├── role1_data.md
    ├── role2_gpt.md
    └── role3_eval.md
```

Quy ước:
- Tài liệu **dùng chung** cho cả nhóm (giao thức, quyết định chung) đặt ngay trong `docs/`.
- Tài liệu **riêng của một vai trò**, khi cần, đặt trong `docs/roleN_<tên>/`.
- Nhật ký AI luôn nằm trong `docs/ai_usage/roleN_<tên>.md`.
