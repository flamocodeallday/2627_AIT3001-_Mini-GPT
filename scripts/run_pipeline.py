import argparse
import subprocess
import sys


def _run(cmd):
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline đa nền tảng (Windows/Linux/Colab) cho Thành viên 3: baseline, eval, scaling"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed/v1_bpe8192")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--ngram_train_tokens", type=int, default=10_000_000)
    parser.add_argument("--run_train_lstm_small", action="store_true")
    parser.add_argument("--run_train_lstm_large", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--run_scaling_gpt", action="store_true")
    parser.add_argument("--gpt_ckpt", type=str, default=None)
    args = parser.parse_args()

    py = sys.executable

    if args.run_train_lstm_small:
        _run([
            py, "-m", "scripts.train_lstm",
            "--data_dir", args.data_dir,
            "--preset", "small",
            "--ctx", str(args.ctx),
            "--batch_size", "16",
            "--max_steps", "3000",
            "--eval_interval", "500",
        ])

    if args.run_train_lstm_large:
        _run([
            py, "-m", "scripts.train_lstm",
            "--data_dir", args.data_dir,
            "--preset", "large",
            "--ctx", str(args.ctx),
            "--batch_size", "8",
            "--grad_accum", "4",
            "--lr", "1e-3",
            "--max_steps", "6000",
            "--eval_interval", "500",
        ])

    if args.run_eval:
        eval_cmd = [
            py, "-m", "scripts.run_eval_all",
            "--data_dir", args.data_dir,
            "--split", "val",
            "--ctx", str(args.ctx),
            "--ngram_train_tokens", str(args.ngram_train_tokens),
            "--lstm_ckpts",
            f"results/checkpoints/lstm_l2_h256_{args.data_dir.split('/')[-1]}_best.pt",
            f"results/checkpoints/lstm_l3_h1024_{args.data_dir.split('/')[-1]}_best.pt",
        ]
        if args.gpt_ckpt:
            eval_cmd.extend(["--gpt_ckpt", args.gpt_ckpt])
        _run(eval_cmd)

    if args.run_scaling_gpt:
        _run([
            py, "-m", "scripts.run_scaling",
            "--arch", "gpt",
            "--data_dir", args.data_dir,
            "--ctx", str(args.ctx),
            "--token_multiplier", "20",
        ])


if __name__ == "__main__":
    main()
