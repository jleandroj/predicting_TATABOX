"""Synthetic DNA corpus for the TATA-box generation task.

A *promoter* in molecular biology often contains a TATA-box: a short, conserved
motif (canonically ``TATAAA``) sitting upstream of the transcription start site.
This module builds a clean, balanced, fully reproducible toy corpus:

- half the sequences contain the motif (a promoter-like sequence);
- the other half are random sequences that do *not* contain it.

Sequences are converted to space-separated k-mers (the tokenization scheme used
by DNABERT-style models) by :func:`to_kmers`, which is generic for any ``k``
(this repo sweeps ``k=1..6``, see :mod:`predicting_tatabox.tokenizer`).

Everything is driven by a fixed seed: same seed in, same data out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: The four DNA bases.
BASES: tuple[str, ...] = ("A", "C", "G", "T")

#: Canonical TATA-box motif used as the positive signal.
TATA_BOX: str = "TATAAA"


@dataclass(frozen=True)
class Example:
    """One labelled DNA sequence.

    Attributes:
        sequence: The raw DNA string (only A/C/G/T).
        label: 1 if the sequence contains the promoter motif, else 0.
    """

    sequence: str
    label: int


def to_kmers(sequence: str, k: int = 3) -> str:
    """Convert a DNA sequence to space-separated overlapping k-mers.

    Generic for any ``k >= 1``: yields ``len(sequence) - k + 1`` overlapping
    windows, so sequence length in tokens stays roughly constant across the
    ``k=1..6`` tokenizer grid (this repo's main axis).

    >>> to_kmers("ATGCA", k=3)
    'ATG TGC GCA'
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(sequence) < k:
        return sequence
    return " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def _random_sequence(rng: random.Random, length: int) -> str:
    """Draw a uniform random DNA sequence of the given length."""
    return "".join(rng.choice(BASES) for _ in range(length))


def make_examples(
    n: int,
    *,
    seq_len: int = 60,
    motif: str = TATA_BOX,
    seed: int = 0,
) -> list[Example]:
    """Generate ``n`` balanced, labelled examples.

    Half the examples contain ``motif`` inserted at a random position (label 1);
    the other half are random sequences guaranteed not to contain it (label 0).
    The result is shuffled deterministically.

    Raises:
        ValueError: if ``seq_len`` is shorter than ``motif``.
    """
    if seq_len < len(motif):
        raise ValueError(f"seq_len ({seq_len}) must be >= len(motif) ({len(motif)})")
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(n):
        label = i % 2  # balanced by construction
        if label == 1:
            seq = _random_sequence(rng, seq_len)
            pos = rng.randint(0, seq_len - len(motif))
            seq = seq[:pos] + motif + seq[pos + len(motif) :]
        else:
            seq = _random_sequence(rng, seq_len)
            while motif in seq:  # keep negatives clean
                seq = _random_sequence(rng, seq_len)
        examples.append(Example(sequence=seq, label=label))
    rng.shuffle(examples)
    return examples
