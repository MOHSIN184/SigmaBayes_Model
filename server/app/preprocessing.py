from __future__ import annotations

import itertools

import numpy as np

DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
INPUT_LENGTH = 81


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def validate_sequence(sequence: str) -> str:
    normalized = normalize_sequence(sequence)
    if len(normalized) != INPUT_LENGTH:
        raise ValueError(
            f"Sequence must be exactly {INPUT_LENGTH} bp; received {len(normalized)} bp."
        )
    invalid = sorted(set(normalized) - set(DNA_TO_INDEX))
    if invalid:
        raise ValueError(
            "Sequence may contain only A, C, G, and T; "
            f"invalid characters: {', '.join(invalid)}."
        )
    return normalized


def gc_content(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def one_hot_encode(sequence: str, max_len: int = INPUT_LENGTH) -> np.ndarray:
    encoded = np.zeros((4, max_len), dtype=np.float32)
    for position, nucleotide in enumerate(sequence[:max_len]):
        encoded[DNA_TO_INDEX[nucleotide], position] = 1.0
    return encoded


def generate_kmers(k: int) -> list[str]:
    return ["".join(chars) for chars in itertools.product("ACGT", repeat=k)]


def normalized_kmer_vector(sequence: str, k: int) -> np.ndarray:
    """Match normalized count-vector logic used by repository k-mer scripts."""
    kmers = generate_kmers(k)
    lookup = {kmer: index for index, kmer in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    windows = len(sequence) - k + 1
    for start in range(windows):
        counts[lookup[sequence[start : start + k]]] += 1.0
    if windows > 0:
        counts /= float(windows)
    return counts
