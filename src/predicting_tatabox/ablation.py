"""The 6 (tokenizer) x 4 (technique) ablation grid.

For each k-mer size ``k`` in ``1..6`` we train one shared SFT starting point
(:func:`train_sft_for_k`), then branch off four post-training techniques from
*that same checkpoint* (:func:`run_cell`):

- ``"sft"``        -- baseline, no further training.
- ``"dpo-sigmoid"`` -- classic DPO (:mod:`trl.DPOTrainer`, ``loss_type="sigmoid"``).
- ``"dpo-ipo"``     -- same trainer, ``loss_type="ipo"``.
- ``"dpo-lora"``    -- DPO with LoRA adapters on the causal LM
  (:func:`predicting_tatabox.dpo.apply_lora`).

Holding the SFT starting point fixed per ``k`` isolates the effect of the
post-training technique; holding the architecture (``n_embd``, ``n_layer``,
``n_head``) fixed across ``k`` isolates the effect of the tokenizer's vocab
size on ``total_params``.
"""

from __future__ import annotations

import copy
import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from predicting_tatabox.data import TATA_BOX, to_kmers
from predicting_tatabox.dpo import (
    BASELINE_MOTIF_RATE,
    RunConfig,
    apply_lora,
    build_causal_model,
    count_parameters,
    generate_sequences,
    motif_rate,
    run_dpo,
    train_sft,
)
from predicting_tatabox.tokenizer import build_causal_tokenizer
from predicting_tatabox.tracking import log_run

#: Columns written to the ablation CSV, in order.
ABLATION_FIELDS: tuple[str, ...] = (
    "k",
    "technique",
    "vocab_size",
    "total_params",
    "trainable_params",
    "trainable_pct",
    "motif_rate",
    "baseline_motif_rate",
    "improvement_over_baseline",
    "train_seconds",
)

#: The four post-training techniques in the grid, and the DPO loss variant
#: (if any) each one trains with.
TECHNIQUE_LOSS_TYPE: dict[str, str] = {
    "sft": "sigmoid",  # unused: "sft" runs no DPO step, kept for config completeness
    "dpo-sigmoid": "sigmoid",
    "dpo-ipo": "ipo",
    "dpo-lora": "sigmoid",
}

#: Default technique axis, in grid order.
DEFAULT_TECHNIQUES: tuple[str, ...] = ("sft", "dpo-sigmoid", "dpo-ipo", "dpo-lora")

#: Default tokenizer axis: k-mer sizes 1 through 6.
DEFAULT_KS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def train_sft_for_k(k: int, config: RunConfig | None = None) -> tuple[Any, Any, RunConfig, float]:
    """Train the shared SFT starting point for k-mer size ``k``.

    Returns ``(model, tokenizer, config_with_k, sft_seconds)``. Every
    technique for this ``k`` branches off a :func:`copy.deepcopy` of
    ``model`` in :func:`run_cell`, so this SFT run is the single shared
    starting point for the whole row of the grid.
    """
    cfg = replace(config or RunConfig(), k=k)
    tokenizer = build_causal_tokenizer(cfg.k)
    model = build_causal_model(tokenizer, cfg)
    sft_seconds = train_sft(model, tokenizer, cfg)
    return model, tokenizer, cfg, sft_seconds


def run_cell(
    sft_model: Any,
    tokenizer: Any,
    config: RunConfig,
    technique: str,
    *,
    sft_seconds: float,
) -> dict[str, Any]:
    """Run one ``technique`` branch from ``sft_model`` and return a result row.

    ``sft_model`` is deep-copied first, so this is safe to call multiple
    times with different techniques on the same starting point.
    """
    if technique not in TECHNIQUE_LOSS_TYPE:
        raise ValueError(f"Unknown technique: {technique!r}")

    cfg = replace(config, loss_type=TECHNIQUE_LOSS_TYPE[technique])
    model = copy.deepcopy(sft_model)

    technique_seconds = 0.0
    if technique == "dpo-lora":
        model = apply_lora(model, cfg)
    if technique != "sft":
        technique_seconds = run_dpo(model, tokenizer, cfg)

    trainable_params, total_params = count_parameters(model)
    motif_kmers = to_kmers(TATA_BOX, cfg.k)
    texts = generate_sequences(model, tokenizer, cfg, seed=cfg.seed)
    rate = motif_rate(texts, motif_kmers)
    train_seconds = sft_seconds if technique == "sft" else technique_seconds

    row: dict[str, Any] = {
        "k": cfg.k,
        "technique": technique,
        "vocab_size": tokenizer.vocab_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(100 * trainable_params / total_params, 4),
        "motif_rate": round(rate, 4),
        "baseline_motif_rate": BASELINE_MOTIF_RATE,
        "improvement_over_baseline": round(rate - BASELINE_MOTIF_RATE, 4),
        "train_seconds": round(train_seconds, 3),
    }
    if cfg.track:
        log_run(name=f"ablation-k{cfg.k}-{technique}", params=asdict(cfg), metrics=row)
    return row


def run_grid(
    ks: Iterable[int] = DEFAULT_KS,
    techniques: Sequence[str] = DEFAULT_TECHNIQUES,
    config: RunConfig | None = None,
) -> list[dict[str, Any]]:
    """Run the full tokenizer x technique grid and return one row per cell."""
    base_config = config or RunConfig()
    rows: list[dict[str, Any]] = []
    for k in ks:
        sft_model, tokenizer, cfg, sft_seconds = train_sft_for_k(k, base_config)
        for technique in techniques:
            rows.append(run_cell(sft_model, tokenizer, cfg, technique, sft_seconds=sft_seconds))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write the ablation grid results to a CSV (created/overwritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ABLATION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in ABLATION_FIELDS})
    return path
