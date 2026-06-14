"""Publication-quality figures for predicting_TATABOX.

Follows the Nature/Science visual language:
  - Background #FBFAF7, ink #1A1A1A, secondary grey #8A8A8A
  - NPG qualitative palette with consistent per-technique color mapping
  - Donut charts for composition, horizontal bars for rankings, line chart for
    performance trajectories, heatmap for the complete ablation grid

Writes four PDFs to results/:
  fig1_performance.pdf  -- TATA-box motif rate vs k-mer resolution (line chart)
  fig2_param_scale.pdf  -- Parameter composition per k (donut series)
  fig3_ranking.pdf      -- All 24 cells ranked by motif rate (horizontal bars)
  fig4_heatmap.pdf      -- Complete ablation grid (k x technique heatmap)

Usage:
    uv run python scripts/plot_publication.py    # requires the [viz] extra
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ── Design tokens ──────────────────────────────────────────────────────────────
BG = "#FBFAF7"
INK = "#1A1A1A"
GREY = "#8A8A8A"
RULE = "#E6E4DE"
SHADOW_C = "#3A3A48"

NPG = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]

TECHNIQUES = ["sft", "dpo-sigmoid", "dpo-ipo", "dpo-lora"]
TECH_COLOR = {t: NPG[i] for i, t in enumerate(TECHNIQUES)}
TECH_LABEL = {
    "sft": "SFT",
    "dpo-sigmoid": "DPO-sigmoid",
    "dpo-ipo": "DPO-IPO",
    "dpo-lora": "DPO-LoRA",
}
ACCENT = TECH_COLOR["dpo-sigmoid"]   # saturates to 1.0 at k >= 5
CALM = "#C8CACE"
BACKBONE = 60_480                    # non-embedding parameters, fixed across all k

RESULTS = Path("results/tokenizer_technique_ablation.csv")
OUT = Path("results")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_csv() -> list[dict[str, str]]:
    with RESULTS.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pivot(rows: list[dict[str, str]], field: str) -> dict[str, list[float]]:
    """Returns {technique: [value_k1, ..., value_k6]}."""
    ks = sorted({int(r["k"]) for r in rows})
    return {
        t: [
            float(
                next(r for r in rows if int(r["k"]) == k and r["technique"] == t)[field]
            )
            for k in ks
        ]
        for t in TECHNIQUES
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────
def strip(ax: Any, *, left: bool = False) -> None:
    """Remove all spines except optionally bottom and left."""
    for name, spine in ax.spines.items():
        on = name == "bottom" or (left and name == "left")
        spine.set_visible(on)
        if on:
            spine.set_color(RULE)
            spine.set_linewidth(0.5)
    ax.tick_params(length=0)
    ax.set_facecolor(BG)


def fig_header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=8, color=GREY)


def soft_shadow(ax: Any) -> None:
    """13-layer blur-like drop shadow; call before setting final ax limits."""
    for i in range(13):
        ax.add_patch(mpatches.Circle(
            (0.025, -0.045),
            1.0 + i * 0.017,
            color=SHADOW_C,
            alpha=0.005,
            zorder=1,
        ))


# ── Figure 1: Performance line chart ──────────────────────────────────────────
def fig1_performance(rows: list[dict[str, str]]) -> Path:
    ks = list(range(1, 7))
    mr = pivot(rows, "motif_rate")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)

    # Regime shading (coarse k=1-2, fine k=5-6)
    ax.axvspan(0.5, 2.5, color=RULE, alpha=0.70, zorder=0)
    ax.axvspan(4.5, 6.5, color=RULE, alpha=0.55, zorder=0)

    # Baseline
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.text(6.12, 0.5, "baseline  0.50", va="center", ha="left",
            fontsize=6.5, color=GREY)

    # Lines: one per technique; DPO-sigmoid highlighted
    for t in TECHNIQUES:
        hero = t == "dpo-sigmoid"
        ax.plot(
            ks, mr[t],
            color=TECH_COLOR[t],
            lw=2.5 if hero else 1.4,
            zorder=4 if hero else 2,
            marker="o",
            markersize=5.5 if hero else 3.5,
            markeredgewidth=0,
        )

    # Regime labels
    for xc, txt in [(1.5, "coarse\n(k = 1–2)"), (5.5, "fine\n(k = 5–6)")]:
        ax.text(xc, 1.12, txt, ha="center", va="bottom",
                fontsize=7.5, color=GREY, style="italic")

    # k=6 callout: single-token effect
    ax.annotate(
        "k = 6: TATAAA\n= one token",
        xy=(6, mr["sft"][5]),
        xytext=(4.85, 0.67),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.7},
        fontsize=7, color=GREY, ha="center",
    )

    # k=3-4 LoRA callout
    ax.annotate(
        "DPO collapses;\nLoRA best here",
        xy=(3, mr["dpo-sigmoid"][2]),
        xytext=(2.15, 0.22),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.7},
        fontsize=7, color=GREY, ha="center",
    )

    handles = [mpatches.Patch(color=TECH_COLOR[t], label=TECH_LABEL[t]) for t in TECHNIQUES]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK,
              handlelength=1.2, loc="upper left", ncol=2)

    ax.set_xlim(0.5, 7.0)
    ax.set_ylim(-0.07, 1.25)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k = {k}" for k in ks], color=INK, fontsize=8.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=GREY, fontsize=8)
    strip(ax, left=True)
    ax.yaxis.grid(True, color=RULE, lw=0.4, zorder=0)
    ax.set_xlabel("k-mer size (tokenizer resolution)", fontsize=9, color=INK, labelpad=8)
    ax.set_ylabel("TATA-box motif rate  (n = 50)", fontsize=9, color=INK, labelpad=8)

    fig_header(
        fig,
        "Tokenizer resolution dominates post-training alignment",
        "TATA-box motif rate across 24 ablation cells"
        "  ·  DGX Spark GB10  ·  sft_epochs = dpo_epochs = 100",
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.11, right=0.86)
    out = OUT / "fig1_performance.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ── Figure 2: Parameter composition donuts ────────────────────────────────────
def _draw_donut(ax: Any, k: int, total: int) -> None:
    emb = total - BACKBONE
    p_emb = emb / total
    p_bk = BACKBONE / total

    # Draw pie first, then set wedge zorders
    wedges, _ = ax.pie(
        [emb, BACKBONE],
        colors=[NPG[0], NPG[1]],
        startangle=90,
        wedgeprops={"width": 0.50, "edgecolor": BG, "linewidth": 1.8},
        radius=1.0,
    )
    for w in wedges:
        w.set_zorder(2)

    # Drop shadow behind wedges
    soft_shadow(ax)

    # Cover the donut hole to hide shadow that would show through
    ax.add_patch(mpatches.Circle((0, 0), 0.49, color=BG, zorder=3))

    # Center labels
    ax.text(0, 0.15, f"k = {k}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=INK, zorder=4)
    ax.text(0, -0.18, f"{total:,}", ha="center", va="center",
            fontsize=7, color=GREY, zorder=4)

    # Percent labels inside slices (only if slice is wide enough to fit)
    # Matplotlib pie: startangle=90, counterclockwise (increasing math angle).
    # First slice: embedding, starts at 90 deg.
    for frac, start_deg in [(p_emb, 90.0), (p_bk, 90.0 + p_emb * 360.0)]:
        if frac < 0.10:
            continue
        mid = math.radians(start_deg + frac * 180.0)
        ax.text(
            0.74 * math.cos(mid),
            0.74 * math.sin(mid),
            f"{frac:.0%}",
            ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=5,
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def fig2_param_scale(rows: list[dict[str, str]]) -> Path:
    ks = sorted({int(r["k"]) for r in rows})
    sft = {int(r["k"]): r for r in rows if r["technique"] == "sft"}

    fig, axes = plt.subplots(1, 6, figsize=(12, 3.8), facecolor=BG)
    for ax, k in zip(axes, ks):
        _draw_donut(ax, k, int(sft[k]["total_params"]))

    legend_handles = [
        mpatches.Patch(color=NPG[0], label="Embedding table  (vocab_size × 48 params)"),
        mpatches.Patch(color=NPG[1], label=f"Transformer backbone  ({BACKBONE:,} params, fixed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.04))

    fig_header(
        fig,
        "Embedding table grows as 4ᵏ · backbone stays fixed",
        "Parameter composition per tokenizer resolution"
        "  ·  n_embd = 48, n_layer = 2, n_head = 2",
    )
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.01, right=0.99, wspace=0.06)
    out = OUT / "fig2_param_scale.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ── Figure 3: Horizontal bar ranking ──────────────────────────────────────────
def fig3_ranking(rows: list[dict[str, str]]) -> Path:
    items = sorted(
        [
            {
                "label": f"k = {r['k']}  ·  {TECH_LABEL[r['technique']]}",
                "value": float(r["motif_rate"]),
            }
            for r in rows
        ],
        key=lambda d: d["value"],
    )   # ascending -> top of chart = best

    n = len(items)
    ys = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 8.5), facecolor=BG)
    ax.set_facecolor(BG)

    for i, d in enumerate(items):
        best = i == n - 1
        ax.barh(ys[i], d["value"], color=ACCENT if best else CALM,
                height=0.62, zorder=2)
        ax.text(
            d["value"] + 0.013, ys[i], f"{d['value']:.2f}",
            va="center", ha="left",
            fontsize=7.5,
            color=INK if best else GREY,
            fontweight="bold" if best else "normal",
        )

    # Thin row separators
    for y in ys[:-1]:
        ax.axhline(y + 0.5, color=RULE, lw=0.35, zorder=0)

    # Baseline marker
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.text(0.5, -1.8, "baseline  0.50",
            ha="center", va="top", fontsize=7, color=GREY)

    ax.set_yticks(ys)
    ax.set_yticklabels([d["label"] for d in items], fontsize=7.8, color=INK)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=GREY)
    strip(ax)
    ax.tick_params(axis="y", length=0, labelsize=7.8, labelcolor=INK)

    fig_header(
        fig,
        "All 24 ablation cells ranked by TATA-box motif rate",
        "Sorted ascending (top = best)  ·  dashed = 0.50 baseline  ·  top bar highlighted",
    )
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.35, right=0.93)
    out = OUT / "fig3_ranking.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ── Figure 4: Heatmap (k x technique) ─────────────────────────────────────────
def fig4_heatmap(rows: list[dict[str, str]]) -> Path:
    ks = sorted({int(r["k"]) for r in rows})
    mr = pivot(rows, "motif_rate")

    # 2D grid: rows = techniques (top to bottom), cols = k (left to right)
    grid = [[mr[t][ki] for ki in range(len(ks))] for t in TECHNIQUES]

    # Custom diverging colormap: NPG-blue at 0 → warm white at baseline → NPG-red at 1
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tatabox_div",
        ["#4DBBD5", "#F0EEE8", "#E64B35"],
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=BG)
    ax.set_facecolor(BG)

    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Value annotations in each cell
    for ri, t in enumerate(TECHNIQUES):
        for ci in range(len(ks)):
            val = mr[t][ci]
            fg = "white" if (val < 0.18 or val > 0.82) else INK
            ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=fg)

    # White grid lines between cells
    for x in range(len(ks) - 1):
        ax.axvline(x + 0.5, color=BG, lw=2.0, zorder=2)
    for y in range(len(TECHNIQUES) - 1):
        ax.axhline(y + 0.5, color=BG, lw=2.0, zorder=2)

    ax.set_xticks(list(range(len(ks))))
    ax.set_xticklabels([f"k = {k}" for k in ks], color=INK, fontsize=9)
    ax.set_yticks(list(range(len(TECHNIQUES))))
    ax.set_yticklabels([TECH_LABEL[t] for t in TECHNIQUES], color=INK, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("motif rate", fontsize=8, color=GREY)
    cbar.ax.tick_params(labelsize=7, labelcolor=GREY, length=0)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0%", "50%\nbaseline", "100%"])

    fig_header(
        fig,
        "TATA-box motif rate across the full k × technique grid",
        "Blue = below baseline · Red = above baseline · White center = 0.50",
    )
    fig.subplots_adjust(top=0.75, bottom=0.04, left=0.14, right=0.90)
    out = OUT / "fig4_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    rows = load_csv()
    paths = [
        fig1_performance(rows),
        fig2_param_scale(rows),
        fig3_ranking(rows),
        fig4_heatmap(rows),
    ]
    for p in paths:
        subprocess.Popen(["xdg-open", str(p)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
