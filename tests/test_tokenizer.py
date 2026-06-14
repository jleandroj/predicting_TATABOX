"""Tests for the k-mer vocabulary and tokenizer construction.

``all_kmers``/``vocab_size`` are pure Python; ``build_causal_tokenizer`` needs
the optional ``[ml]`` stack (``tokenizers`` + ``transformers``) and is skipped
otherwise.
"""

from __future__ import annotations

import pytest

from predicting_tatabox.tokenizer import SPECIAL_TOKENS, all_kmers, vocab_size


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_all_kmers_count_and_shape(k: int) -> None:
    kmers = all_kmers(k)
    assert len(kmers) == 4**k
    assert len(set(kmers)) == 4**k  # all unique
    assert all(len(kmer) == k for kmer in kmers)


def test_all_kmers_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        all_kmers(0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_vocab_size_matches_4_pow_k_plus_specials(k: int) -> None:
    assert vocab_size(k) == 4**k + len(SPECIAL_TOKENS)


@pytest.mark.parametrize("k", [1, 3, 6])
def test_build_causal_tokenizer_vocab_size(k: int) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from predicting_tatabox.tokenizer import build_causal_tokenizer

    tok = build_causal_tokenizer(k)
    assert tok.vocab_size == vocab_size(k)
    assert tok.pad_token == "[PAD]"
    assert tok.bos_token == "[CLS]"
    assert tok.eos_token == "[SEP]"
