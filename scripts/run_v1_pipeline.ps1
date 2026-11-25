$ErrorActionPreference = 'Stop'
python -m scripts.train_lstm --data_dir data/processed/v1_bpe8192 --hidden_dim 256 --num_layers 2 --ctx 256 --batch_size 16 --max_steps 3000 --eval_interval 500 --eval_max_tokens 200000
python -m scripts.run_eval_all --data_dir data/processed/v1_bpe8192 --split val --ctx 256 --ngram_train_tokens 2000000 --eval_max_tokens 200000 --lstm_ckpt results/checkpoints/lstm_l2_h256_v1_bpe8192_best.pt
python -m scripts.run_scaling --data_dir data/processed/v1_bpe8192 --steps_per_model 1000 --batch_size 16 --ctx 256 --eval_max_tokens 100000
