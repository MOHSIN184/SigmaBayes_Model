from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VALID_DNA_CHARACTERS = {"A", "C", "G", "T"}


def find_data_file(data_dir, filename):
    """Find a data file in data_dir, data_dir/processed, or data_dir/raw."""
    base_dir = Path(data_dir)
    searched_paths = [
        base_dir / filename,
        base_dir / "processed" / filename,
        base_dir / "raw" / filename,
    ]

    for path in searched_paths:
        if path.is_file():
            return path

    searched = "\n".join(f"- {path}" for path in searched_paths)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched these paths:\n{searched}"
    )


def read_fasta(file_path):
    """Read a FASTA file into dictionaries with id and sequence fields."""
    records = []
    current_id = None
    sequence_parts = []

    with Path(file_path).open("r", encoding="utf-8") as fasta_file:
        for raw_line in fasta_file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    records.append(
                        {
                            "id": current_id,
                            "sequence": "".join(sequence_parts).upper(),
                        }
                    )
                current_id = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append("".join(line.split()))

    if current_id is not None:
        records.append(
            {
                "id": current_id,
                "sequence": "".join(sequence_parts).upper(),
            }
        )

    return records


def _load_fasta_group(data_dir, entries):
    rows = []
    for filename, label, label_name, split_name in entries:
        path = find_data_file(data_dir, filename)
        for record in read_fasta(path):
            rows.append(
                {
                    "id": record["id"],
                    "sequence": record["sequence"],
                    "label": label,
                    "label_name": label_name,
                    "source_file": path.name,
                    "split": split_name,
                }
            )

    df = pd.DataFrame(
        rows, columns=["id", "sequence", "label", "label_name", "source_file", "split"]
    )
    train_df = (
        df[df["split"] == "train"]
        .drop(columns="split")
        .reset_index(drop=True)
    )
    test_df = (
        df[df["split"] == "test"]
        .drop(columns="split")
        .reset_index(drop=True)
    )
    return train_df, test_df


def load_binary_dataset(data_dir="data"):
    entries = [
        ("Non_Promoter_cdhit_train.fasta", 0, "Non-Promoter", "train"),
        ("Promoter_cdhit_train.fasta", 1, "Promoter", "train"),
        ("Non_Promoter_cdhit_test.fasta", 0, "Non-Promoter", "test"),
        ("Promoter_cdhit_test.fasta", 1, "Promoter", "test"),
    ]
    return _load_fasta_group(data_dir, entries)


def load_sigma_dataset(data_dir="data"):
    sigma_labels = {
        "Sigma24": 0,
        "Sigma28": 1,
        "Sigma32": 2,
        "Sigma38": 3,
        "Sigma54": 4,
        "Sigma70": 5,
    }

    entries = []
    for label_name, label in sigma_labels.items():
        entries.extend(
            [
                (f"{label_name}_cdhit_train.fasta", label, label_name, "train"),
                (f"{label_name}_cdhit_test.fasta", label, label_name, "test"),
            ]
        )

    return _load_fasta_group(data_dir, entries)


def split_train_val_calibration(
    train_df, val_size=0.15, cal_size=0.15, random_state=42
):
    if val_size <= 0 or cal_size <= 0:
        raise ValueError("val_size and cal_size must be positive.")
    if val_size + cal_size >= 1:
        raise ValueError("val_size + cal_size must be less than 1.")

    remaining_size = val_size + cal_size
    train_split, temp_split = train_test_split(
        train_df,
        test_size=remaining_size,
        random_state=random_state,
        stratify=train_df["label"],
    )

    calibration_fraction = cal_size / remaining_size
    val_split, calibration_split = train_test_split(
        temp_split,
        test_size=calibration_fraction,
        random_state=random_state,
        stratify=temp_split["label"],
    )

    return (
        train_split.reset_index(drop=True),
        val_split.reset_index(drop=True),
        calibration_split.reset_index(drop=True),
    )


def check_dataset(df):
    sequence_lengths = df["sequence"].fillna("").astype(str).str.len()
    invalid_characters = sorted(
        {
            char
            for sequence in df["sequence"].fillna("").astype(str).str.upper()
            for char in sequence
            if char not in VALID_DNA_CHARACTERS
        }
    )
    invalid_sequence_count = int(
        df["sequence"]
        .fillna("")
        .astype(str)
        .str.upper()
        .apply(lambda seq: any(char not in VALID_DNA_CHARACTERS for char in seq))
        .sum()
    )

    return {
        "total_sequences": int(len(df)),
        "class_distribution": df["label_name"].value_counts().to_dict(),
        "min_length": int(sequence_lengths.min()) if len(sequence_lengths) else 0,
        "max_length": int(sequence_lengths.max()) if len(sequence_lengths) else 0,
        "mean_length": float(np.mean(sequence_lengths)) if len(sequence_lengths) else 0.0,
        "invalid_sequence_count": invalid_sequence_count,
        "invalid_characters": invalid_characters,
    }


def print_dataset_report(df, title="Dataset"):
    report = check_dataset(df)
    mean_length = report["mean_length"]

    print(f"=== {title} ===")
    print(f"Total records: {report['total_sequences']}")
    print("Class counts:")
    for label_name, count in report["class_distribution"].items():
        print(f"  {label_name}: {count}")
    print(
        "Sequence length: "
        f"min={report['min_length']}, "
        f"max={report['max_length']}, "
        f"mean={mean_length:.2f}"
    )
    print(f"Invalid sequence count: {report['invalid_sequence_count']}")
    print(f"Invalid characters: {report['invalid_characters']}")
