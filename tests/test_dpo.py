"""Tests for the from-scratch causal LM: SFT, DPO loss variants, and LoRA.

Needs the optional ``[ml]`` stack (torch, transformers, trl, datasets, peft)
and is skipped otherwise. Everything runs on a tiny from-scratch causal LM
with a handful of examples, with no network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("peft")

from predicting_tatabox.data import TATA_BOX, to_kmers  # noqa: E402
from predicting_tatabox.dpo import (  # noqa: E402
    RunConfig,
    apply_lora,
    build_causal_model,
    count_parameters,
    generate_sequences,
    make_preference_pairs,
    motif_rate,
    run_dpo,
    train_sft,
)
from predicting_tatabox.tokenizer import build_causal_tokenizer  # noqa: E402


def _tiny_config(**overrides: object) -> RunConfig:
    defaults: dict[str, object] = dict(
        k=3,
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


def test_motif_rate() -> None:
    motif_kmers = to_kmers(TATA_BOX, k=3)
    texts = [f"x {motif_kmers} y", "no motif here", motif_kmers]
    assert motif_rate(texts, motif_kmers) == pytest.approx(2 / 3)


@pytest.mark.parametrize("k", [1, 6])
def test_make_preference_pairs_chosen_has_motif_rejected_does_not(k: int) -> None:
    cfg = _tiny_config(k=k)
    pairs = make_preference_pairs(cfg, seed=1)
    motif_kmers = to_kmers(TATA_BOX, cfg.k)

    assert len(pairs["prompt"]) == cfg.n_pref
    for chosen, rejected in zip(pairs["chosen"], pairs["rejected"], strict=True):
        assert motif_kmers in chosen
        assert motif_kmers not in rejected


@pytest.mark.parametrize("k", [1, 6])
def test_causal_tokenizer_prompt_is_prefix_of_prompt_plus_completion(k: int) -> None:
    """trl's DPO data prep requires tokenize(prompt) to be a prefix of
    tokenize(prompt + chosen) / tokenize(prompt + rejected)."""
    cfg = _tiny_config(k=k)
    tok = build_causal_tokenizer(cfg.k)
    pairs = make_preference_pairs(cfg, seed=1)
    prompt, chosen, rejected = pairs["prompt"][0], pairs["chosen"][0], pairs["rejected"][0]

    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_chosen = tok(prompt + chosen, add_special_tokens=False)["input_ids"]
    ids_rejected = tok(prompt + rejected, add_special_tokens=False)["input_ids"]

    assert ids_chosen[: len(ids_prompt)] == ids_prompt
    assert ids_rejected[: len(ids_prompt)] == ids_prompt


def test_train_sft_then_generate(tmp_path: Path) -> None:
    cfg = _tiny_config(output_dir=str(tmp_path))
    tok = build_causal_tokenizer(cfg.k)
    model = build_causal_model(tok, cfg)

    seconds = train_sft(model, tok, cfg)
    assert seconds > 0

    texts = generate_sequences(model, tok, cfg, seed=cfg.seed)
    assert len(texts) == cfg.n_eval
    rate = motif_rate(texts, to_kmers(TATA_BOX, cfg.k))
    assert 0.0 <= rate <= 1.0


@pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
def test_run_dpo_loss_variants(tmp_path: Path, loss_type: str) -> None:
    cfg = _tiny_config(output_dir=str(tmp_path), loss_type=loss_type)
    tok = build_causal_tokenizer(cfg.k)
    model = build_causal_model(tok, cfg)
    train_sft(model, tok, cfg)

    seconds = run_dpo(model, tok, cfg)
    assert seconds > 0


def test_apply_lora_reduces_trainable_params(tmp_path: Path) -> None:
    cfg = _tiny_config(output_dir=str(tmp_path))
    tok = build_causal_tokenizer(cfg.k)
    model = build_causal_model(tok, cfg)
    _, full_total = count_parameters(model)

    lora_model = apply_lora(model, cfg)
    trainable, total = count_parameters(lora_model)

    assert 0 < trainable < total
    assert total >= full_total


def test_run_dpo_with_lora(tmp_path: Path) -> None:
    cfg = _tiny_config(output_dir=str(tmp_path))
    tok = build_causal_tokenizer(cfg.k)
    model = build_causal_model(tok, cfg)
    train_sft(model, tok, cfg)

    lora_model = apply_lora(model, cfg)
    seconds = run_dpo(lora_model, tok, cfg)
    assert seconds > 0
