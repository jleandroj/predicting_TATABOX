# Model card: `predicting_TATABOX` tokenizer x technique ablation

## Task

A from-scratch GPT-2-style causal LM learns to generate synthetic DNA
sequences (60 bases, A/C/G/T) such that, after training, freely generated
sequences contain a **TATA-box** promoter motif (`TATAAA`) more often than
the SFT corpus's baseline rate of 0.5 (the corpus is exactly 50%
motif-containing by construction). `motif_rate` is measured by checking
whether the motif's k-mers appear as a substring of the generated, decoded
token sequence.

## Architecture (fixed across the grid)

```
GPT2Config(
    vocab_size = 4**k + 5,   # k-mer vocab + 5 special tokens; the ONLY thing that varies with k
    n_positions = 80,
    n_embd  = 48,
    n_layer = 2,
    n_head  = 2,
)
```

`n_embd=48, n_layer=2, n_head=2` is ~2x [bio-ml-lab](https://github.com/jleandroj/bio-ml-lab)'s
Week 8 from-scratch causal LM (`n_embd=32`, ~30K params).

## Tokenizer axis: k-mer size

The same `WordLevel` tokenizer (`predicting_tatabox.tokenizer.build_causal_tokenizer`,
no post-processor -- see "Gotchas" below) is rebuilt for each `k`, with a
vocabulary of all `4**k` k-mers plus 5 special tokens
(`[PAD] [UNK] [CLS] [SEP] [MASK]`). Holding the architecture fixed, the only
thing that changes with `k` is the size of the token embedding table
(`wte`/`lm_head`, tied), so `total_params` grows almost entirely from vocab
size:

| k | vocab_size = 4^k + 5 | total_params (analytic: 60,480 + vocab x 48) |
|---|----------------------|-----------------------------------------------|
| 1 |    9                 |  60,912 |
| 2 |   21                 |  61,488 |
| 3 |   69                 |  63,792 |
| 4 |  261                 |  73,008 |
| 5 | 1,029                | 109,872 |
| 6 | 4,101                | 257,328 |

`k=3` is the "~64K, ~2x Week 8" reference point. At `k=6`, the 6-base TATA-box
motif (`TATAAA`) collapses into a **single vocabulary token** -- a
qualitatively different regime than `k<6`, where the model must generate the
motif as a sequence of multiple tokens.

The analytic column above is `60,480 + vocab_size(k) * 48`, where 60,480 is
the (k-independent) parameter count of everything except the tied
embedding/output table (2 transformer blocks + position embeddings + layer
norms). It was derived from, and matches, two measured points (k=1: 60,912;
k=6: 257,328) from a smoke run of `scripts/run_ablation.py`. The full grid run
below records the *measured* `total_params` per cell directly.

## Technique axis

For each `k`, one SFT run (`predicting_tatabox.dpo.train_sft`) is the shared
starting checkpoint for four branches (`predicting_tatabox.ablation.run_cell`):

| technique     | what happens |
|---------------|--------------|
| `sft`         | baseline -- no further training, report the SFT checkpoint's `motif_rate`. |
| `dpo-sigmoid` | `trl.DPOTrainer` on `(prompt, chosen, rejected)` triples, `loss_type="sigmoid"` (classic [DPO](https://arxiv.org/abs/2305.18290)). |
| `dpo-ipo`     | same trainer, `loss_type="ipo"` ([IPO](https://arxiv.org/abs/2310.12036)). |
| `dpo-lora`    | `dpo-sigmoid`, but the SFT checkpoint is wrapped with [LoRA](https://arxiv.org/abs/2106.09685) adapters (`peft.LoraConfig(task_type=CAUSAL_LM, target_modules=["c_attn"], r=8, lora_alpha=16)`) before DPO -- only the adapters are trainable. |

`chosen` completions have the TATA-box motif inserted; `rejected` completions
are guaranteed not to contain it. All four branches share `seed`, so
`generate_sequences` for `sft` vs the three DPO variants is a controlled
before/after comparison (same starting weights, same sampling seed, only the
post-training step differs).

## Gotchas (generalized from bio-ml-lab Week 8 to k=1..6 and to LoRA)

- **Tokenizer must have no post-processor.** `trl.DPOTrainer` requires
  `tokenize(prompt)` to be an exact prefix of `tokenize(prompt + chosen)` /
  `tokenize(prompt + rejected)`. `build_causal_tokenizer` places `[CLS]`/`[SEP]`
  as ordinary vocab tokens in the literal text instead of via a
  `TemplateProcessing` post-processor, and `make_preference_pairs` splits a
  single `to_kmers(...)` call at a k-mer boundary
  (`n_prompt_kmers = prompt_len - k + 1`) so prompt and completion share a
  consistent tokenization. Verified for `k=1` and `k=6` (the tokenizer
  extremes) in `tests/test_dpo.py`.
- **`ref_model` must be an explicit `copy.deepcopy`, even for LoRA.** For a
  from-scratch model (`config._name_or_path == ""`), `ref_model=None` makes
  `trl` try to *reload* the model and crash. `run_dpo` always passes
  `ref_model=copy.deepcopy(model)`. For the `dpo-lora` cell, `model` is a
  `PeftModel`; its deepcopy is used purely under `torch.no_grad()` for
  reference log-probs and is never registered with the optimizer, so its
  adapter weights being nominally "trainable" is harmless. Verified in
  `tests/test_dpo.py::test_run_dpo_with_lora`.
- **LoRA + GPT-2's `Conv1D`.** GPT-2's `c_attn` projection is a
  `transformers` `Conv1D`, not `nn.Linear`. `peft.get_peft_model` detects this
  automatically and sets `fan_in_fan_out=True` (logged as a `UserWarning`,
  expected and harmless).
- **`DPOConfig.loss_type` is a list.** In `trl==1.6.0`, `DPOConfig.loss_type`
  is `list[str]`; `run_dpo` passes `loss_type=[config.loss_type]`.

## Reproducing

```bash
uv pip install -e ".[ml]"
uv run python scripts/run_ablation.py        # -> results/tokenizer_technique_ablation.csv
uv pip install -e ".[viz]"
uv run python scripts/plot_ablation.py       # -> results/*.png
```

Each cell also logs a JSON record (params + metrics + git SHA + timestamp) to
`runs/` via `predicting_tatabox.tracking.log_run`.

## Results

_Pending a full run of `scripts/run_ablation.py` (24 cells, default config:
`n_sft=200, n_pref=100, n_eval=50, sft_epochs=60, dpo_epochs=20`) on the DGX
Spark. This section will report, per cell: `vocab_size`, `total_params`,
`trainable_params`/`trainable_pct`, `motif_rate` vs `baseline_motif_rate=0.5`,
and `train_seconds`, plus the two plots from `scripts/plot_ablation.py`
(`motif_rate_by_k.png`, `total_params_by_k.png`)._
