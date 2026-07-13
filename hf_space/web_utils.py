"""Input parsing and validation helpers for the BayesSigma web interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

INPUT_LENGTH = 81
DNA_ALPHABET = frozenset("ACGT")
AMINO_ACID_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_FILE_SUFFIXES = frozenset({".fasta", ".fa", ".txt"})
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 5_000


@dataclass(frozen=True)
class SequenceRecord:
    """One named sequence collected from manual input or a file."""

    identifier: str
    sequence: str
    source: str


def _unique_identifier(candidate: str, seen: dict[str, int]) -> str:
    base = candidate.strip().split()[0] if candidate.strip() else "Sequence"
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}_{seen[base]}"


def parse_fasta_text(text: str, source: str = "FASTA") -> list[SequenceRecord]:
    """Parse FASTA while preserving empty records for useful validation errors."""
    records: list[SequenceRecord] = []
    seen: dict[str, int] = {}
    current_id: str | None = None
    current_lines: list[str] = []

    def finish_record() -> None:
        if current_id is not None:
            records.append(
                SequenceRecord(current_id, "".join(current_lines).strip(), source)
            )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            finish_record()
            current_id = _unique_identifier(line[1:], seen)
            current_lines = []
        elif line:
            if current_id is None:
                raise ValueError(
                    "Invalid FASTA format. Sequence data must follow a header beginning with '>'."
                )
            current_lines.append(line)

    finish_record()
    if not records:
        raise ValueError("Invalid FASTA file. No sequence records were found.")
    return records


def parse_manual_input(text: str | None) -> list[SequenceRecord]:
    """Parse one sequence, newline-separated sequences, or pasted FASTA."""
    raw = text or ""
    if not raw.strip():
        return []
    if any(line.lstrip().startswith(">") for line in raw.splitlines()):
        return parse_fasta_text(raw, "Manual FASTA")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return [
        SequenceRecord(f"Sequence_{index}", line, "Manual input")
        for index, line in enumerate(lines, start=1)
    ]


def _file_path(upload: Any) -> Path:
    if isinstance(upload, (str, Path)):
        return Path(upload)
    name = getattr(upload, "name", None)
    if name:
        return Path(name)
    raise ValueError("The uploaded file could not be read.")


def parse_uploaded_file(upload: Any) -> list[SequenceRecord]:
    """Safely read an uploaded FASTA or plain-text sequence file."""
    path = _file_path(upload)
    if path.suffix.lower() not in ALLOWED_FILE_SUFFIXES:
        raise ValueError("Upload a .fasta, .fa, or .txt file.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("The uploaded file exceeds the 10 MB limit.")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The uploaded file must be UTF-8 plain text.") from error

    if any(line.lstrip().startswith(">") for line in text.splitlines()):
        records = parse_fasta_text(text, path.name)
    else:
        records = [
            SequenceRecord(f"Sequence_{index}", line.strip(), path.name)
            for index, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
        if not records:
            raise ValueError("The uploaded file does not contain any sequences.")
    return records


def collect_records(manual_text: str | None, upload: Any) -> list[SequenceRecord]:
    """Combine both supported input modes and enforce a bounded batch size."""
    records = parse_manual_input(manual_text)
    if upload:
        records.extend(parse_uploaded_file(upload))
    if not records:
        raise ValueError("Please enter a DNA sequence or upload a FASTA file.")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"A maximum of {MAX_RECORDS:,} sequences is allowed per batch.")
    unique_records: list[SequenceRecord] = []
    seen: dict[str, int] = {}
    for record in records:
        unique_records.append(
            SequenceRecord(
                _unique_identifier(record.identifier, seen),
                record.sequence,
                record.source,
            )
        )
    return unique_records


def validate_web_sequence(sequence: str) -> str:
    """Validate UI input with actionable DNA-, protein-, and length-specific messages."""
    raw = str(sequence).strip()
    if not raw:
        raise ValueError("Please enter only an 81 bp DNA promoter sequence.")
    if any(character.isspace() for character in raw):
        raise ValueError("Invalid DNA sequence. Please enter only A, T, G, and C.")

    normalized = raw.upper()
    characters = set(normalized)
    invalid = characters - DNA_ALPHABET
    if invalid:
        if (
            len(normalized) >= 5
            and normalized.isalpha()
            and characters <= AMINO_ACID_ALPHABET
            and sum(base in DNA_ALPHABET for base in normalized) / len(normalized) < 0.8
        ):
            raise ValueError(
                "Protein sequence detected. Please enter a DNA promoter sequence."
            )
        if normalized.isalpha() and len(normalized) >= 10:
            raise ValueError("Invalid DNA sequence. Please enter only A, T, G, and C.")
        if any(character in DNA_ALPHABET for character in characters):
            raise ValueError("Invalid DNA sequence. Please enter only A, T, G, and C.")
        raise ValueError("Please enter only an 81 bp DNA promoter sequence.")

    if len(normalized) < INPUT_LENGTH:
        raise ValueError("Sequence length is shorter than 81 bp.")
    if len(normalized) > INPUT_LENGTH:
        raise ValueError("Sequence length exceeds 81 bp.")
    return normalized
