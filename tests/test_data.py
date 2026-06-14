"""Sanity checks for the synthetic DNA corpus (no ML deps needed)."""

from __future__ import annotations

import pytest

from predicting_tatabox.data import TATA_BOX, Example, make_examples, to_kmers


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_to_kmers_overlapping_windows(k: int) -> None:
    seq = "ACGTACGTAC"
    kmers = to_kmers(seq, k).split(" ")
    assert len(kmers) == len(seq) - k + 1
    assert all(len(kmer) == k for kmer in kmers)
    assert all(seq[i : i + k] == kmer for i, kmer in enumerate(kmers))


def test_to_kmers_shorter_than_k_returns_sequence() -> None:
    assert to_kmers("AC", k=3) == "AC"


def test_to_kmers_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        to_kmers("ACGT", k=0)


def test_make_examples_is_balanced_and_reproducible() -> None:
    examples = make_examples(20, seed=0)
    assert len(examples) == 20
    assert sum(e.label for e in examples) == 10

    again = make_examples(20, seed=0)
    assert examples == again


def test_make_examples_label_matches_motif_presence() -> None:
    examples = make_examples(20, seed=0)
    for e in examples:
        assert isinstance(e, Example)
        assert (TATA_BOX in e.sequence) == (e.label == 1)


def test_make_examples_rejects_seq_len_shorter_than_motif() -> None:
    with pytest.raises(ValueError):
        make_examples(2, seq_len=len(TATA_BOX) - 1)
