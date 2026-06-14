"""End-to-end smoke tests for the tokenizer x technique ablation grid.

Needs the optional ``[ml]`` stack and is skipped otherwise. Runs a tiny
2 (k) x 4 (technique) grid -- using k=1 and k=6, the tokenizer extremes --
on CPU in seconds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("peft")

from predicting_tatabox.ablation import ABLATION_FIELDS, run_grid, write_csv  # noqa: E402
from predicting_tatabox.dpo import RunConfig  # noqa: E402


def _tiny_config(**overrides: object) -> RunConfig:
    defaults: dict[str, object] = dict(
        seq_len=24,
        prompt_len=12,
        n_sft=16,
        n_pref=8,
        n_eval=4,
        sft_epochs=1.0,
        dpo_epochs=1.0,
        batch_size=4,
        max_length=32,
        max_new_tokens=10,
        n_embd=8,
        n_layer=1,
        n_head=1,
        lora_r=4,
        lora_alpha=8,
        seed=0,
        track=False,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def test_run_grid_tiny_two_ks_all_techniques(tmp_path: Path) -> None:
    cfg = _tiny_config(output_dir=str(tmp_path))
    rows = run_grid(ks=[1, 6], techniques=["sft", "dpo-sigmoid", "dpo-ipo", "dpo-lora"], config=cfg)

    assert len(rows) == 2 * 4
    for row in rows:
        assert set(ABLATION_FIELDS) <= set(row)
        assert 0.0 <= row["motif_rate"] <= 1.0
        assert row["baseline_motif_rate"] == 0.5
        assert row["train_seconds"] >= 0

    vocab_by_k = {row["k"]: row["vocab_size"] for row in rows}
    assert vocab_by_k[1] == 4**1 + 5
    assert vocab_by_k[6] == 4**6 + 5

    params_by_k = {row["k"]: row["total_params"] for row in rows if row["technique"] == "sft"}
    assert params_by_k[6] > params_by_k[1]

    lora_rows = [row for row in rows if row["technique"] == "dpo-lora"]
    for row in lora_rows:
        assert 0 < row["trainable_pct"] < 100


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "k": 1,
            "technique": "sft",
            "vocab_size": 9,
            "total_params": 1234,
            "trainable_params": 1234,
            "trainable_pct": 100.0,
            "motif_rate": 0.4,
            "baseline_motif_rate": 0.5,
            "improvement_over_baseline": -0.1,
            "train_seconds": 1.2,
        },
        {
            "k": 6,
            "technique": "dpo-lora",
            "vocab_size": 4101,
            "total_params": 5678,
            "trainable_params": 100,
            "trainable_pct": 1.76,
            "motif_rate": 0.7,
            "baseline_motif_rate": 0.5,
            "improvement_over_baseline": 0.2,
            "train_seconds": 2.3,
        },
    ]
    out = write_csv(rows, tmp_path / "ablation.csv")
    with out.open(encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert list(read[0].keys()) == list(ABLATION_FIELDS)
    assert read[1]["technique"] == "dpo-lora"
    assert read[1]["motif_rate"] == "0.7"
