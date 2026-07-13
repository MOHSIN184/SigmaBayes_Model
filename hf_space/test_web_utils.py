"""Focused regression tests for public web input behavior."""
from pathlib import Path

import pytest

from web_utils import collect_records, parse_fasta_text, parse_uploaded_file, validate_web_sequence

VALID = "AAGTCATGAAACGATTCAAACATGGCGCGAATATTTATGTGATGCCTCCTTTACCGTCGCTCTCTGGTTAACACCCCATGC"


def test_valid_81_bp_sequence_is_normalized() -> None:
    assert validate_web_sequence(VALID.lower()) == VALID


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        (VALID[:-1], "shorter than 81 bp"),
        (VALID + "A", "exceeds 81 bp"),
        (VALID[:-1] + "N", "Invalid DNA sequence"),
        ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYR", "Protein sequence detected"),
        ("MTEYK", "Protein sequence detected"),
        ("Hello", "Please enter only an 81 bp"),
        ("123456", "Please enter only an 81 bp"),
        ("ATGC ATGC", "Invalid DNA sequence"),
    ],
)
def test_invalid_sequences_receive_specific_feedback(sequence: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_web_sequence(sequence)


def test_multiple_fasta_records_and_headers() -> None:
    records = parse_fasta_text(f">promoter_1 description\n{VALID}\n>promoter_2\n{VALID}\n")
    assert [(record.identifier, record.sequence) for record in records] == [
        ("promoter_1", VALID),
        ("promoter_2", VALID),
    ]


def test_invalid_fasta_is_rejected() -> None:
    with pytest.raises(ValueError, match="must follow a header"):
        parse_fasta_text(VALID + "\n>later\n" + VALID)


def test_plain_text_file_supports_multiple_sequences(tmp_path: Path) -> None:
    path = tmp_path / "batch.txt"
    path.write_text(f"{VALID}\n{VALID}\n", encoding="utf-8")
    assert len(parse_uploaded_file(path)) == 2


def test_large_batch_is_bounded() -> None:
    with pytest.raises(ValueError, match="maximum of 5,000"):
        collect_records("\n".join([VALID] * 5_001), None)


def test_oversized_file_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "large.fasta"
    with path.open("wb") as handle:
        handle.seek(10 * 1024 * 1024)
        handle.write(b"x")
    with pytest.raises(ValueError, match="10 MB limit"):
        parse_uploaded_file(path)
