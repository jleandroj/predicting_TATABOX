"""Run the full tokenizer (k=1..6) x technique (sft/dpo-sigmoid/dpo-ipo/dpo-lora)
ablation grid and write ``results/tokenizer_technique_ablation.csv``.

For each k-mer size, one SFT-trained model is the shared starting point for
all four post-training techniques (see :mod:`predicting_tatabox.ablation`).
At Week 8's pace (~12s per SFT+DPO cycle at ``n_embd=32`` on CPU), the default
24-cell grid at ``n_embd=48`` should take on the order of minutes.

Usage:
    uv run python scripts/run_ablation.py                      # requires the [ml] extra
    uv run python scripts/run_ablation.py --ks 1 3 6 --n-sft 50 --sft-epochs 5 --dpo-epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from predicting_tatabox.ablation import (
    DEFAULT_KS,
    DEFAULT_TECHNIQUES,
    run_grid,
    write_csv,
)
from predicting_tatabox.dpo import RunConfig

OUTPUT = Path("results/tokenizer_technique_ablation.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenizer x technique ablation grid.")
    parser.add_argument(
        "--ks", type=int, nargs="+", default=list(DEFAULT_KS), help="k-mer sizes to sweep."
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=list(DEFAULT_TECHNIQUES),
        choices=["sft", "dpo-sigmoid", "dpo-ipo", "dpo-lora"],
        help="Post-training techniques to run for each k.",
    )
    parser.add_argument("--n-sft", type=int, default=200, help="SFT corpus size.")
    parser.add_argument("--n-pref", type=int, default=100, help="Preference pairs for DPO.")
    parser.add_argument("--n-eval", type=int, default=50, help="Sequences sampled for motif_rate.")
    parser.add_argument("--sft-epochs", type=float, default=60.0, help="SFT training epochs.")
    parser.add_argument("--dpo-epochs", type=float, default=20.0, help="DPO training epochs.")
    parser.add_argument("--beta", type=float, default=0.05, help="DPO beta (KL penalty strength).")
    parser.add_argument(
        "--n-embd", type=int, default=48, help="GPT-2 embedding size (fixed across k)."
    )
    parser.add_argument("--n-layer", type=int, default=2, help="GPT-2 layer count.")
    parser.add_argument("--n-head", type=int, default=2, help="GPT-2 attention head count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no-track", action="store_true", help="Disable run logging to runs/.")
    args = parser.parse_args(argv)

    config = RunConfig(
        n_sft=args.n_sft,
        n_pref=args.n_pref,
        n_eval=args.n_eval,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
        beta=args.beta,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        seed=args.seed,
        track=not args.no_track,
    )

    print(
        f"Running {len(args.ks)} x {len(args.techniques)} grid "
        f"(ks={args.ks}, techniques={args.techniques}, "
        f"n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}) ..."
    )
    rows = run_grid(ks=args.ks, techniques=args.techniques, config=config)
    path = write_csv(rows, OUTPUT)
    print(f"\nWrote {len(rows)} rows to {path}\n")
    for row in rows:
        print(
            f"  k={row['k']}  {row['technique']:<12} vocab={row['vocab_size']:>5}  "
            f"params={row['total_params']:>7} (train {row['trainable_pct']:>6.2f}%)  "
            f"motif_rate={row['motif_rate']:.3f} ({row['improvement_over_baseline']:+.3f})  "
            f"time={row['train_seconds']:>6.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
