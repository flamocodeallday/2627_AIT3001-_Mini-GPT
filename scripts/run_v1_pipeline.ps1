$ErrorActionPreference = 'Stop'
python -m scripts.run_pipeline --data_dir data/processed/v1_bpe8192 --ctx 512 --run_train_lstm_small --run_eval
