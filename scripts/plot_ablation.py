"""Plot the tokenizer x technique ablation results.

Reads ``results/tokenizer_technique_ablation.csv`` (from
``scripts/run_ablation.py``) and writes two figures:

- ``results/motif_rate_by_k.png``: TATA-box motif rate vs k-mer size, one
  line per post-training technique, with the SFT corpus's baseline motif
  rate (0.5, by construction) drawn as a reference line.
- ``results/total_params_by_k.png``: total model parameters vs k-mer size
  (from the "sft" rows), annotated with vocab size -- the empirical
  ``4**k`` vocab-table growth that motivated this grid (see the Week 3
  Transformers note).

Usage:
    uv run python scripts/plot_ablation.py        # requires the [viz] extra
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/tokenizer_technique_ablation.csv")
MOTIF_OUT = Path("results/motif_rate_by_k.png")
PARAMS_OUT = Path("results/total_params_by_k.png")


def _read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_motif_rate(rows: list[dict[str, str]]) -> Path:
    by_technique: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_technique[row["technique"]].append((int(row["k"]), float(row["motif_rate"])))
    baseline = float(rows[0]["baseline_motif_rate"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for technique, points in by_technique.items():
        points.sort()
        ks = [p[0] for p in points]
        rates = [p[1] for p in points]
        ax.plot(ks, rates, marker="o", label=technique)
    ax.axhline(baseline, color="gray", linestyle=":", label=f"SFT corpus baseline ({baseline:.2f})")
    ax.set_xlabel("k-mer size")
    ax.set_ylabel("motif rate (generated sequences)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(sorted({int(r["k"]) for r in rows}))
    ax.set_title("TATA-box motif rate vs tokenizer (k) and technique")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    MOTIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MOTIF_OUT, dpi=150)
    return MOTIF_OUT


def plot_total_params(rows: list[dict[str, str]]) -> Path:
    sft_rows = sorted((r for r in rows if r["technique"] == "sft"), key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in sft_rows]
    params = [int(r["total_params"]) for r in sft_rows]
    vocabs = [int(r["vocab_size"]) for r in sft_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar([str(k) for k in ks], params, color="tab:blue")
    for bar, vocab in zip(bars, vocabs, strict=True):
        ax.annotate(
            f"vocab={vocab}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("k-mer size")
    ax.set_ylabel("total parameters")
    ax.set_title("Total parameters vs k-mer size (vocab table grows ~4^k)")

    fig.tight_layout()
    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PARAMS_OUT, dpi=150)
    return PARAMS_OUT


def main() -> int:
    rows = _read_rows()
    if not rows:
        raise SystemExit(f"No data in {CSV_PATH}")
    motif_path = plot_motif_rate(rows)
    params_path = plot_total_params(rows)
    print(f"Wrote {motif_path}")
    print(f"Wrote {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
