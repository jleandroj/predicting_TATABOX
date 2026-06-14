# predicting_TATABOX

Standalone 6×4 ablation: k-mer tokenizer (k=1..6) × post-training technique
(sft / dpo-sigmoid / dpo-ipo / dpo-lora) on a from-scratch GPT-2 causal LM.
Full 24-cell grid run on NVIDIA DGX Spark (GB10), 2026-06-14 — results committed.

Sibling to `bio-ml-lab` but **fully independent** (no shared imports, no shared venv).

## Key commands

```bash
# Full grid run + terminal log capture
uv run python scripts/run_ablation.py 2>&1 | tee logs/$(date +%Y%m%d-%H%M%S)-run_ablation.log

# Subset run (fast, for testing changes to the code)
uv run python scripts/run_ablation.py --ks 3 6 --sft-epochs 5 --dpo-epochs 2

# Generate PNG plots from existing CSV
uv run python scripts/plot_ablation.py

# Quality gate
.venv/bin/pre-commit run --all-files
.venv/bin/pytest
```

## Project layout

```
src/predicting_tatabox/
  data.py        — BASES, TATA_BOX, to_kmers(seq, k), make_examples(n, ...)
  tokenizer.py   — all_kmers(k), vocab_size(k)=4^k+5, build_causal_tokenizer(k)
  dpo.py         — RunConfig (all knobs), build_causal_model, train_sft, run_dpo,
                   apply_lora, make_preference_pairs, generate_sequences, motif_rate
  ablation.py    — run_cell, run_grid, write_csv; TECHNIQUE_LOSS_TYPE dict
  tracking.py    — vendored log_run → runs/YYYYMMDDTHHMMSS.json (gitignored)
scripts/
  run_ablation.py  — CLI → results/tokenizer_technique_ablation.csv
  plot_ablation.py — CSV → results/motif_rate_by_k.png + total_params_by_k.png
results/           — committed: CSV + PNGs (real DGX run)
runs/              — gitignored JSON logs (only .gitkeep tracked; use runs/*)
logs/              — gitignored terminal captures (use tee convention above)
tests/             — 39 tests; tiny configs: n_embd=8, n_layer=1, sft_epochs=1.0
```

## Key findings (to understand the code)

- **k drives performance** more than technique. k≤3: all techniques fail. k=3–4: `dpo-lora`
  (only 3,072 trainable params ~4%) is the only above-baseline technique. k≥5: full DPO
  saturates to 1.000 motif_rate.
- **DPO collapse at low k:** full DPO (sigmoid/ipo) shifts distribution toward a template,
  not the underlying motif → motif_rate drops to 0.00–0.06 at k=1–4.
- **k=6 single-token effect:** `TATAAA` is one vocab token → SFT alone reaches 0.78.
- **Vocab-table dominance:** `total_params = 60,480 + vocab_size(k) * 48` matches all 24 cells.
- **Gotchas:** `ref_model = copy.deepcopy(model)` always (empty `_name_or_path` crashes trl).
  `DPOConfig(loss_type=[config.loss_type])` — list, not string, in trl==1.6.0.

## Hard separation rule

`/home/leandro/bio-ml-lab` is a **separate** repo. Do not touch it from this session.
