"""K-mer vocabulary and tokenizer construction, generic for any ``k``.

This is the heart of the tokenizer axis of the ablation grid: the same
:func:`build_causal_tokenizer` is reused for every ``k`` in ``1..6``, and the
resulting vocabulary size (``4**k + 5`` special tokens) drives the
token-embedding table size of the otherwise-fixed model architecture (see
:mod:`predicting_tatabox.dpo`).
"""

from __future__ import annotations

import itertools
from typing import Any

from predicting_tatabox.data import BASES

#: Special tokens shared by every tokenizer in the grid, in vocab-index order.
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def all_kmers(k: int) -> list[str]:
    """All ``4**k`` k-mers over :data:`predicting_tatabox.data.BASES`, in order.

    >>> all_kmers(1)
    ['A', 'C', 'G', 'T']
    >>> len(all_kmers(3))
    64
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def vocab_size(k: int) -> int:
    """Total vocabulary size for k-mer size ``k``: ``4**k`` k-mers + 5 specials."""
    return int(4**k) + len(SPECIAL_TOKENS)


def build_causal_tokenizer(k: int) -> Any:
    """Build a WordLevel k-mer tokenizer with **no post-processor**.

    ``[CLS]``/``[SEP]`` are ordinary vocabulary tokens, placed explicitly as
    literal text by callers rather than auto-inserted. This keeps
    ``tokenize(prompt)`` an exact prefix of ``tokenize(prompt + completion)``,
    which :class:`trl.DPOTrainer` requires (see :mod:`predicting_tatabox.dpo`).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(list(SPECIAL_TOKENS) + all_kmers(k))}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[CLS]",
        eos_token="[SEP]",
        mask_token="[MASK]",
    )
