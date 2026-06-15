from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    check_dataset,
    load_binary_dataset,
    load_sigma_dataset,
    print_dataset_report,
)


def ensure_output_dirs():
    table_dir = PROJECT_ROOT / "results" / "tables"
    figure_dir = PROJECT_ROOT / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def build_dataset_summary(datasets):
    rows = []
    for dataset_name, df in datasets.items():
        report = check_dataset(df)
        rows.append(
            {
                "dataset": dataset_name,
                "total_sequences": report["total_sequences"],
                "class_distribution": str(report["class_distribution"]),
                "min_length": report["min_length"],
                "max_length": report["max_length"],
                "mean_length": report["mean_length"],
                "invalid_sequence_count": report["invalid_sequence_count"],
                "invalid_characters": str(report["invalid_characters"]),
            }
        )
    return pd.DataFrame(rows)


def build_class_distribution(datasets):
    rows = []
    for dataset_name, df in datasets.items():
        counts = df["label_name"].value_counts()
        total = len(df)
        for class_name, count in counts.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "count": int(count),
                    "percentage": 100.0 * count / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(df, title, save_path):
    counts = df["label_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def print_length_summary(name, df):
    lengths = df["sequence"].str.len()
    print(
        f"{name}: min={lengths.min()}, max={lengths.max()}, "
        f"mean={lengths.mean():.2f}"
    )


def run_dataset_check():
    table_dir, figure_dir = ensure_output_dirs()

    print("Loading binary dataset...")
    binary_train, binary_test = load_binary_dataset(PROJECT_ROOT / "data")

    print("Loading sigma dataset...")
    sigma_train, sigma_test = load_sigma_dataset(PROJECT_ROOT / "data")

    datasets = {
        "Binary Train": binary_train,
        "Binary Test": binary_test,
        "Sigma Train": sigma_train,
        "Sigma Test": sigma_test,
    }

    print("\nBinary dataset stats:")
    print_dataset_report(binary_train, "Binary Train")
    print_dataset_report(binary_test, "Binary Test")

    print("\nSigma dataset stats:")
    print_dataset_report(sigma_train, "Sigma Train")
    print_dataset_report(sigma_test, "Sigma Test")

    print("\nSequence length distributions:")
    for dataset_name, df in datasets.items():
        print_length_summary(dataset_name, df)

    print("\nInvalid DNA character checks:")
    for dataset_name, df in datasets.items():
        report = check_dataset(df)
        print(
            f"{dataset_name}: invalid_sequence_count="
            f"{report['invalid_sequence_count']}, "
            f"invalid_characters={report['invalid_characters']}"
        )

    summary_df = build_dataset_summary(datasets)
    class_distribution_df = build_class_distribution(datasets)

    summary_path = table_dir / "dataset_summary.csv"
    class_distribution_path = table_dir / "dataset_class_distribution.csv"
    binary_plot_path = figure_dir / "binary_class_distribution.png"
    sigma_plot_path = figure_dir / "sigma_class_distribution.png"

    summary_df.to_csv(summary_path, index=False)
    class_distribution_df.to_csv(class_distribution_path, index=False)
    plot_class_distribution(binary_train, "Binary Train Class Distribution", binary_plot_path)
    plot_class_distribution(sigma_train, "Sigma Train Class Distribution", sigma_plot_path)

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {class_distribution_path}")
    print(f"  {binary_plot_path}")
    print(f"  {sigma_plot_path}")
    print("\nDataset check completed successfully.")

    return {
        "binary_train": binary_train,
        "binary_test": binary_test,
        "sigma_train": sigma_train,
        "sigma_test": sigma_test,
        "summary": summary_df,
        "class_distribution": class_distribution_df,
    }


if __name__ == "__main__":
    run_dataset_check()
