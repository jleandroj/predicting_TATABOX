"""From-scratch causal LM, SFT, DPO, and LoRA — the building blocks of one
ablation-grid cell.

This module is a generalization of bio-ml-lab's Week 8 ``finetune/dpo.py``:
the same SFT-then-(DPO|LoRA) recipe, but every knob that used to be fixed
(``k``, the DPO loss variant, whether LoRA adapters are used) is now a field
on :class:`RunConfig`, so :mod:`predicting_tatabox.ablation` can sweep over
them.

**Two stages, same as Week 8:**

1. **SFT** (:func:`train_sft`): the model learns to generate plausible k-mer
   "sentences" from the synthetic corpus (50% TATA-box motif, 50% not — see
   :func:`predicting_tatabox.data.make_examples`).
2. **Post-training** (:func:`run_dpo`, optionally behind :func:`apply_lora`):
   :class:`trl.DPOTrainer` further trains on ``(prompt, chosen, rejected)``
   triples, where ``chosen`` completions contain the TATA-box motif and
   ``rejected`` ones don't.

We measure :func:`motif_rate`: the fraction of freely generated sequences
containing the TATA-box k-mers. ``BASELINE_MOTIF_RATE = 0.5`` is the SFT
corpus's motif fraction by construction.

**Gotchas inherited from Week 8 (still apply, generalized to any k):**

- :class:`trl.DPOTrainer` checks that ``tokenize(prompt)`` is an exact prefix
  of ``tokenize(prompt + chosen)`` / ``tokenize(prompt + rejected)``.
  :func:`predicting_tatabox.tokenizer.build_causal_tokenizer` has **no
  post-processor** for this reason, and :func:`make_preference_pairs` splits a
  single :func:`~predicting_tatabox.data.to_kmers` call at a k-mer boundary.
- For a from-scratch model (empty ``config._name_or_path``), passing
  ``ref_model=None`` makes trl try to *reload* the model and crash. This
  module always passes an explicit frozen copy: ``ref_model =
  copy.deepcopy(model)``. This applies uniformly to the LoRA cell too (the
  ref model is then a deep copy of the ``PeftModel``; since it is only ever
  used under ``torch.no_grad()`` for reference log-probs and is never handed
  to the optimizer, its adapter weights being technically "trainable" is
  harmless).

This repo is purely generative (no classification head), so there is
deliberately no QLoRA/bitsandbytes here — see :func:`apply_lora` for the
LoRA-on-a-generative-model axis instead.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

from predicting_tatabox.data import BASES, TATA_BOX, make_examples, to_kmers


@dataclass
class RunConfig:
    """All knobs for one cell of the tokenizer x technique ablation grid.

    Attributes:
        k: k-mer size for the DNA tokenization (the tokenizer axis, swept
            1..6 by :mod:`predicting_tatabox.ablation`).
        seq_len: length of each synthetic DNA sequence.
        prompt_len: number of bases (from the start of each sequence) used as
            the DPO ``prompt``; the remaining ``seq_len - prompt_len`` bases
            are the ``chosen``/``rejected`` completion.
        n_sft: number of sequences in the SFT corpus.
        n_pref: number of (prompt, chosen, rejected) preference triples.
        n_eval: number of sequences sampled when measuring ``motif_rate``.
        sft_epochs, dpo_epochs: training epochs for each stage.
        batch_size: per-device batch size for both stages.
        sft_learning_rate, dpo_learning_rate: learning rates for each stage.
        beta: DPO temperature (controls divergence from the reference policy).
        max_length: max token length for both stages (also sets the model's
            position embedding size).
        max_new_tokens: tokens to sample when measuring ``motif_rate``.
        n_embd, n_layer, n_head: GPT-2 architecture knobs, held **fixed**
            across the tokenizer axis (~64K total params at k=3, ~2x Week
            8's ~30K). ``total_params`` is a measured *output* of the grid,
            not a target -- it grows with ``k`` purely via the token
            embedding table.
        loss_type: :class:`trl.DPOConfig` loss variant, e.g. ``"sigmoid"``
            (classic DPO) or ``"ipo"``.
        lora_r, lora_alpha: LoRA rank/alpha for :func:`apply_lora`.
        seed: random seed (data generation, training, sampling).
        output_dir: base directory for the Trainer's (unused) checkpoints.
        track: log each stage via :func:`predicting_tatabox.tracking.log_run`.
    """

    k: int = 3
    seq_len: int = 60
    prompt_len: int = 15
    n_sft: int = 200
    n_pref: int = 100
    n_eval: int = 50
    sft_epochs: float = 60.0
    dpo_epochs: float = 20.0
    batch_size: int = 8
    sft_learning_rate: float = 5e-4
    dpo_learning_rate: float = 5e-5
    beta: float = 0.05
    max_length: int = 80
    max_new_tokens: int = 50
    n_embd: int = 48
    n_layer: int = 2
    n_head: int = 2
    loss_type: str = "sigmoid"
    lora_r: int = 8
    lora_alpha: int = 16
    seed: int = 0
    output_dir: str = "outputs/tatabox-ablation"
    track: bool = True


def build_causal_model(tokenizer: Any, config: RunConfig) -> Any:
    """Build a tiny GPT-2 causal LM from scratch (no download)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    gpt2_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=config.max_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(gpt2_config)


class _CausalDataset:
    """A minimal torch-style dataset for causal LM training.

    Padding positions are masked to ``-100`` in ``labels`` so they do not
    contribute to the loss.
    """

    def __init__(self, encodings: dict[str, Any]) -> None:
        self._input_ids = encodings["input_ids"]
        self._attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        input_ids = self._input_ids[idx]
        attention_mask = self._attention_mask[idx]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _sft_dataset(tokenizer: Any, config: RunConfig) -> _CausalDataset:
    examples = make_examples(config.n_sft, seq_len=config.seq_len, seed=config.seed)
    texts = [f"[CLS] {to_kmers(e.sequence, config.k)} [SEP]" for e in examples]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return _CausalDataset(enc)


def train_sft(model: Any, tokenizer: Any, config: RunConfig) -> float:
    """Fine-tune ``model`` on the synthetic corpus and return the wall-clock time."""
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(config.seed)
    dataset = _sft_dataset(tokenizer, config)
    args = TrainingArguments(
        output_dir=f"{config.output_dir}/sft",
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.sft_learning_rate,
        seed=config.seed,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    start = time.perf_counter()
    trainer.train()
    return time.perf_counter() - start


def _random_seq(rng: random.Random, length: int) -> str:
    """Draw a uniform random DNA sequence of the given length."""
    return "".join(rng.choice(BASES) for _ in range(length))


def make_preference_pairs(config: RunConfig, *, seed: int) -> Any:
    """Build a ``datasets.Dataset`` of ``(prompt, chosen, rejected)`` triples.

    All triples share the random ``prompt`` (the first ``prompt_len`` bases).
    ``chosen`` completions have the TATA-box motif inserted; ``rejected``
    completions are guaranteed not to contain it. Prompt and completion k-mers
    come from a single :func:`~predicting_tatabox.data.to_kmers` call, split at
    ``n_prompt_kmers = prompt_len - k + 1``, so ``tokenize(prompt)`` is an
    exact prefix of ``tokenize(prompt + chosen)`` / ``tokenize(prompt +
    rejected)`` -- the consistency check :class:`trl.DPOTrainer` performs.
    """
    from datasets import Dataset

    rng = random.Random(seed)
    k = config.k
    n_prompt_kmers = config.prompt_len - k + 1
    completion_len = config.seq_len - config.prompt_len

    prompts: list[str] = []
    chosen: list[str] = []
    rejected: list[str] = []
    for _ in range(config.n_pref):
        prompt_seq = _random_seq(rng, config.prompt_len)
        # If the motif already sits inside the prompt itself, the rejection
        # loop below for `rejected` can never succeed (the shared prefix
        # alone would always contain it) -- redraw until it doesn't.
        while TATA_BOX in prompt_seq:
            prompt_seq = _random_seq(rng, config.prompt_len)

        comp = _random_seq(rng, completion_len)
        pos = rng.randint(0, completion_len - len(TATA_BOX))
        chosen_comp = comp[:pos] + TATA_BOX + comp[pos + len(TATA_BOX) :]
        chosen_seq = prompt_seq + chosen_comp

        rejected_comp = _random_seq(rng, completion_len)
        rejected_seq = prompt_seq + rejected_comp
        while TATA_BOX in rejected_seq:
            rejected_comp = _random_seq(rng, completion_len)
            rejected_seq = prompt_seq + rejected_comp

        chosen_kmers = to_kmers(chosen_seq, k).split(" ")
        rejected_kmers = to_kmers(rejected_seq, k).split(" ")
        prompts.append("[CLS] " + " ".join(chosen_kmers[:n_prompt_kmers]))
        chosen.append(" " + " ".join(chosen_kmers[n_prompt_kmers:]) + " [SEP]")
        rejected.append(" " + " ".join(rejected_kmers[n_prompt_kmers:]) + " [SEP]")

    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def run_dpo(model: Any, tokenizer: Any, config: RunConfig) -> float:
    """Run DPO training on ``model`` in place and return the wall-clock time.

    ``config.loss_type`` selects the :class:`trl.DPOConfig` loss variant
    (``"sigmoid"`` = classic DPO, ``"ipo"`` = IPO, ...). The reference policy
    is a frozen ``copy.deepcopy`` of ``model`` taken *before* training --
    passing ``ref_model=None`` crashes for a from-scratch model (see the
    module docstring), and this holds for the LoRA-wrapped ``model`` too.
    """
    from trl.trainer.dpo_config import DPOConfig
    from trl.trainer.dpo_trainer import DPOTrainer

    pref_dataset = make_preference_pairs(config, seed=config.seed + 1)
    ref_model = copy.deepcopy(model)
    args = DPOConfig(
        output_dir=f"{config.output_dir}/dpo",
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.dpo_epochs,
        learning_rate=config.dpo_learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        loss_type=[config.loss_type],
        seed=config.seed,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=pref_dataset,
        processing_class=tokenizer,
    )
    start = time.perf_counter()
    trainer.train()
    return time.perf_counter() - start


def apply_lora(model: Any, config: RunConfig) -> Any:
    """Wrap ``model`` with LoRA adapters on GPT-2's attention projection.

    PEFT auto-detects that ``c_attn`` is a :class:`transformers.GPT2`
    ``Conv1D`` layer (not ``nn.Linear``) and sets ``fan_in_fan_out=True``
    accordingly -- no extra configuration needed.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def count_parameters(model: Any) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def generate_sequences(model: Any, tokenizer: Any, config: RunConfig, *, seed: int) -> list[str]:
    """Sample ``config.n_eval`` sequences from ``model``, starting from ``[CLS]``."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    model.eval()
    input_ids = torch.full(
        (config.n_eval, 1), tokenizer.bos_token_id, dtype=torch.long, device=model.device
    )
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return list(tokenizer.batch_decode(out, skip_special_tokens=False))


def motif_rate(texts: list[str], motif_kmers: str) -> float:
    """Fraction of ``texts`` containing ``motif_kmers`` as a substring."""
    return sum(1 for t in texts if motif_kmers in t) / len(texts)


#: The SFT corpus is exactly 50% motif-containing by construction
#: (:func:`predicting_tatabox.data.make_examples` alternates labels).
BASELINE_MOTIF_RATE = 0.5
