# Bảng Tổng Hợp Kết Quả Đánh Giá (v1_bpe8192 - val)

| Model | Data | Split | Tag | Ctx | Total Params | Non-Emb Params | Train Tokens | Eval Tokens | Avg NLL (nats) | Perplexity | BPC | Git Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ngram_kn_n3 | v1_bpe8192 | val | official | 512 | 5,687,280 | 5,687,280 | 10,000,000 | 2,775,100 | 4.8865 | 132.49 | 2.0493 | b6aa483 |
| ngram_kn_n4 | v1_bpe8192 | val | official | 512 | 13,056,157 | 13,056,157 | 10,000,000 | 2,775,100 | 4.8603 | 129.06 | 2.0383 | b6aa483 |
| lstm_l2_h256 | v1_bpe8192 | val | official | 512 | 3,148,288 | 1,051,136 | 24,576,000 | 2,775,100 | 4.2938 | 73.25 | 1.8007 | b6aa483 |
| lstm_l3_h1024 | v1_bpe8192 | val | official | 512 | 33,568,768 | 25,180,160 | 49,152,000 | 2,775,100 | 3.8694 | 47.91 | 1.6227 | b6aa483 |
