from pathlib import Path
import itertools
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_binary_dataset, load_sigma_dataset  # noqa: E402


BASES = ["A", "C", "G", "T"]
BINARY_CLASS_NAMES = ["Non-Promoter", "Promoter"]
SIGMA_CLASS_NAMES = ["Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"]
MAX_LEN = 81


def warn(message):
    print(f"Warning: {message}")


def read_optional_csv(path, index_col=None):
    path = Path(path)
    if not path.is_file():
        warn(f"Missing optional file: {path}")
        return None
    return pd.read_csv(path, index_col=index_col)


def generate_kmers(k):
    return ["".join(chars) for chars in itertools.product("ACGT", repeat=k)]


def kmer_frequency_vector(sequence, kmers):
    sequence = str(sequence).upper()
    k = len(kmers[0]) if kmers else 0
    counts = np.zeros(len(kmers), dtype=np.float64)
    if len(sequence) < k:
        return counts
    index = {kmer: i for i, kmer in enumerate(kmers)}
    valid = 0
    for start in range(len(sequence) - k + 1):
        kmer = sequence[start : start + k]
        if kmer in index:
            counts[index[kmer]] += 1.0
            valid += 1
    if valid:
        counts /= valid
    return counts


def mean_kmer_profiles(df, class_names, k):
    kmers = generate_kmers(k)
    profiles = []
    for class_name in class_names:
        class_df = df[df["label_name"] == class_name]
        if class_df.empty:
            profiles.append(np.zeros(len(kmers), dtype=np.float64))
            continue
        matrix = np.stack(
            [kmer_frequency_vector(seq, kmers) for seq in class_df["sequence"]],
            axis=0,
        )
        profiles.append(matrix.mean(axis=0))
    return np.stack(profiles, axis=0)


def position_frequency_table(df, task, class_names):
    rows = []
    for class_name in class_names:
        class_df = df[df["label_name"] == class_name]
        sequences = class_df["sequence"].fillna("").astype(str).str.upper().tolist()
        for position in range(MAX_LEN):
            counts = {base: 0 for base in BASES}
            total = 0
            for sequence in sequences:
                if position < len(sequence):
                    base = sequence[position]
                    if base in counts:
                        counts[base] += 1
                        total += 1
            frequencies = {
                base: (counts[base] / total if total else 0.0) for base in BASES
            }
            consensus_base = max(BASES, key=lambda base: frequencies[base])
            rows.append(
                {
                    "task": task,
                    "class_name": class_name,
                    "position": position + 1,
                    **frequencies,
                    "consensus_base": consensus_base,
                    "consensus_frequency": frequencies[consensus_base],
                }
            )
    return pd.DataFrame(rows)


def consensus_sequences(freq_df):
    rows = []
    for (task, class_name), group in freq_df.groupby(["task", "class_name"]):
        group = group.sort_values("position")
        rows.append(
            {
                "task": task,
                "class_name": class_name,
                "consensus_sequence": "".join(group["consensus_base"].astype(str)),
                "mean_consensus_strength": group["consensus_frequency"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_position_frequency_heatmap(freq_df, class_names, title, save_path):
    matrix = []
    for class_name in class_names:
        values = (
            freq_df[freq_df["class_name"] == class_name]
            .sort_values("position")["consensus_frequency"]
            .to_numpy()
        )
        matrix.append(values)
    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(13, max(3, len(class_names) * 0.7)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.25, vmax=1.0)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xticks(np.arange(0, MAX_LEN, 10))
    ax.set_xticklabels(np.arange(1, MAX_LEN + 1, 10))
    ax.set_xlabel("Position")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Consensus frequency")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_similarity(task, class_names, profiles, table_dir, figure_dir):
    similarity = cosine_similarity(profiles)
    sim_df = pd.DataFrame(similarity, index=class_names, columns=class_names)
    sim_df.index.name = "class_name"
    sim_df.to_csv(table_dir / f"analysis_{task}_kmer_cosine_similarity.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(f"{task.title()} mean k-mer cosine similarity")
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", color="white")
    fig.tight_layout()
    fig.savefig(figure_dir / f"analysis_{task}_kmer_cosine_similarity.png", dpi=300)
    plt.close(fig)
    return sim_df


def class_imbalance_summary(binary_train, binary_test, sigma_train, sigma_test):
    rows = []
    for task, train_df, test_df, class_names in [
        ("binary", binary_train, binary_test, BINARY_CLASS_NAMES),
        ("sigma", sigma_train, sigma_test, SIGMA_CLASS_NAMES),
    ]:
        total_df = pd.concat([train_df, test_df], ignore_index=True)
        for class_name in class_names:
            train_count = int((train_df["label_name"] == class_name).sum())
            test_count = int((test_df["label_name"] == class_name).sum())
            total_count = int((total_df["label_name"] == class_name).sum())
            rows.append(
                {
                    "task": task,
                    "class_name": class_name,
                    "train_count": train_count,
                    "test_count": test_count,
                    "total_count": total_count,
                    "train_percentage": train_count / len(train_df) if len(train_df) else 0.0,
                    "test_percentage": test_count / len(test_df) if len(test_df) else 0.0,
                    "total_percentage": total_count / len(total_df) if len(total_df) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_sigma_imbalance(imbalance_df, figure_dir):
    sigma_df = imbalance_df[imbalance_df["task"] == "sigma"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#B74A4A" if name in {"Sigma54", "Sigma28", "Sigma38"} else "#3B82A0" for name in sigma_df["class_name"]]
    ax.bar(sigma_df["class_name"], sigma_df["total_count"], color=colors)
    ax.set_ylabel("Total sequences")
    ax.set_title("Sigma class imbalance")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "analysis_sigma_class_imbalance.png", dpi=300)
    plt.close(fig)


def build_sigma_diagnostic(table_dir, figure_dir):
    report = read_optional_csv(table_dir / "sigma_safe_classification_report.csv", index_col=0)
    uncertainty = read_optional_csv(table_dir / "sigma_safe_uncertainty_by_class.csv")
    conformal = read_optional_csv(table_dir / "sigma_safe_conformal_classwise_summary.csv")
    if report is None:
        return pd.DataFrame()

    report = report.loc[[name for name in SIGMA_CLASS_NAMES if name in report.index]].reset_index()
    report = report.rename(
        columns={
            "index": "class_name",
            "f1-score": "f1_score",
            "support": "test_count",
        }
    )
    diagnostic = report[["class_name", "test_count", "precision", "recall", "f1_score"]].copy()

    if uncertainty is not None:
        uncertainty = uncertainty.rename(columns={"true_label_name": "class_name"})
        diagnostic = diagnostic.merge(uncertainty, on="class_name", how="left")
    if conformal is not None:
        diagnostic = diagnostic.merge(conformal, on=["class_name", "test_count"], how="left")

    def difficulty(f1):
        if pd.isna(f1):
            return "Unknown"
        if f1 < 0.20:
            return "High difficulty"
        if f1 < 0.50:
            return "Moderate difficulty"
        return "Lower difficulty"

    diagnostic["difficulty_label"] = diagnostic["f1_score"].apply(difficulty)
    diagnostic.to_csv(table_dir / "analysis_sigma_class_diagnostic.csv", index=False)

    if {"f1_score", "mean_entropy"}.issubset(diagnostic.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(diagnostic["f1_score"], diagnostic["mean_entropy"], s=70, color="#3B82A0")
        for _, row in diagnostic.iterrows():
            ax.annotate(row["class_name"], (row["f1_score"], row["mean_entropy"]), xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("F1 score")
        ax.set_ylabel("Mean entropy")
        ax.set_title("Sigma F1 vs uncertainty")
        fig.tight_layout()
        fig.savefig(figure_dir / "analysis_sigma_f1_vs_uncertainty.png", dpi=300)
        plt.close(fig)
    return diagnostic


def build_confusion_pairs(table_dir, figure_dir):
    predictions = read_optional_csv(table_dir / "sigma_safe_uncertainty_results.csv")
    if predictions is None:
        predictions = read_optional_csv(table_dir / "sigma_safe_conformal_predictions_90.csv")
    if predictions is None:
        return pd.DataFrame()
    required = {"true_label_name", "predicted_label_name"}
    if not required.issubset(predictions.columns):
        warn("Prediction file lacks true_label_name/predicted_label_name columns.")
        return pd.DataFrame()

    counts = (
        predictions.groupby(["true_label_name", "predicted_label_name"])
        .size()
        .reset_index(name="count")
    )
    true_totals = predictions.groupby("true_label_name").size().rename("true_total")
    counts = counts.merge(true_totals, on="true_label_name", how="left")
    counts["percentage_within_true_class"] = counts["count"] / counts["true_total"]
    counts = counts.drop(columns="true_total").sort_values("count", ascending=False)
    counts.to_csv(table_dir / "analysis_sigma_confusion_pairs.csv", index=False)

    matrix = counts.pivot_table(
        index="true_label_name",
        columns="predicted_label_name",
        values="count",
        fill_value=0,
    ).reindex(index=SIGMA_CLASS_NAMES, columns=SIGMA_CLASS_NAMES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(len(SIGMA_CLASS_NAMES)))
    ax.set_yticks(np.arange(len(SIGMA_CLASS_NAMES)))
    ax.set_xticklabels(SIGMA_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(SIGMA_CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Sigma confusion pair counts")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(figure_dir / "analysis_sigma_confusion_pair_heatmap.png", dpi=300)
    plt.close(fig)
    return counts


def summarize_optional_evidence(table_dir):
    evidence = {}
    for name in [
        "binary_metrics.csv",
        "kmer_reliability_binary_metrics.csv",
        "kmer_reliability_binary_conformal_metrics.csv",
        "sigma_safe_metrics.csv",
        "kmer_reliability_sigma_metrics.csv",
        "kmer_reliability_sigma_classification_report.csv",
        "cv_summary_wide.csv",
        "cv_fold_metrics.csv",
    ]:
        path = table_dir / name
        if path.exists():
            evidence[name] = "available"
        else:
            warn(f"Optional evidence file missing: {path}")
            evidence[name] = "missing"
    return evidence


def write_report(table_dir, binary_sim, sigma_sim, imbalance_df, diagnostic_df, confusion_df):
    sigma_max_similarity = np.nan
    if not sigma_sim.empty:
        arr = sigma_sim.to_numpy(dtype=float)
        mask = ~np.eye(arr.shape[0], dtype=bool)
        sigma_max_similarity = float(np.max(arr[mask]))
    binary_offdiag = np.nan
    if not binary_sim.empty and binary_sim.shape == (2, 2):
        binary_offdiag = float(binary_sim.iloc[0, 1])

    hardest = ""
    if not diagnostic_df.empty:
        hardest_df = diagnostic_df.sort_values("f1_score").head(3)
        hardest = ", ".join(
            f"{row.class_name} (F1={row.f1_score:.3f})"
            for row in hardest_df.itertuples()
        )
    top_confusions = ""
    if not confusion_df.empty:
        top_confusions = "\n".join(
            f"- {row.true_label_name} -> {row.predicted_label_name}: {row.count} "
            f"({row.percentage_within_true_class:.1%})"
            for row in confusion_df.head(8).itertuples()
        )

    report = f"""# BayesSigma Biological and Error Analysis Report

## Summary of Current Evidence

- Binary classification is stable and strongest with the calibrated 5-mer Random Forest.
- Sigma classification remains weak.
- Cross-validation confirms sigma macro F1 remains below 0.50.
- CNN models are not consistently superior to classical k-mer models.
- The hybrid CNN+k-mer model did not improve performance.

## Biological Interpretation

Position-frequency and consensus summaries were generated for binary and sigma classes. The binary promoter/non-promoter comparison has a mean 5-mer cosine similarity of approximately {binary_offdiag:.3f}. Sigma classes show high 3-mer profile overlap, with the maximum off-diagonal cosine similarity approximately {sigma_max_similarity:.3f}. This supports the interpretation that sigma-factor classes are not cleanly separable using short local composition alone.

Class imbalance is substantial in the sigma dataset. Sigma70 is dominant, while Sigma54, Sigma28, and Sigma38 are small classes. This imbalance makes macro F1 a more honest endpoint than accuracy and helps explain why models can achieve acceptable raw accuracy while failing minority sigma classes.

The sigma diagnostic table identifies the hardest classes as: {hardest if hardest else "unavailable"}. Low F1, high uncertainty, and large conformal set sizes indicate that reliability outputs are meaningful: the system tends to express uncertainty where sigma labels are biologically or statistically difficult.

## Confusion Patterns

Top sigma confusion pairs:

{top_confusions if top_confusions else "- Confusion predictions were unavailable."}

These errors suggest overlapping sequence patterns and minority-class fragility rather than a simple neural-network architecture problem.

## Publication Decision

BayesSigma is not recommended yet for a well-known performance-focused journal as a high-performance sigma-factor classifier. Binary promoter classification is comparatively stable, but sigma-factor classification remains too weak for strong performance claims.

The project may be suitable after reframing as a trustworthy benchmarking and reliability framework. The current evidence is strongest when presented as a comparative study of calibrated classical and neural models, uncertainty estimation, conformal prediction, and stability analysis.

## Minimum Improvements Before Submission

- Add external datasets if possible.
- Add more sigma promoter samples, especially Sigma54, Sigma28, and Sigma38.
- Add motif/logo interpretation with biological discussion.
- Add calibrated/conformal reliability comparison across best models.
- Add cross-validation results.
- Discuss class imbalance and overlapping sigma motifs.
- Avoid claiming a state-of-the-art classifier.

## Recommended Final Framing

BayesSigma should be presented as:

"A reliability-aware benchmark for bacterial promoter and sigma-factor prediction using calibrated machine learning, uncertainty estimation, and conformal prediction."

## Final Decision

Proceed only as a benchmark/reliability paper, not as a high-performance classifier paper.

Do not submit to a well-known journal until sigma performance or biological validation improves.
"""
    path = PROJECT_ROOT / "results" / "analysis_publication_decision_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main():
    results_dir = PROJECT_ROOT / "results"
    table_dir = results_dir / "tables"
    figure_dir = results_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    binary_train, binary_test = load_binary_dataset(PROJECT_ROOT / "data")
    sigma_train, sigma_test = load_sigma_dataset(PROJECT_ROOT / "data")
    binary_full = pd.concat([binary_train, binary_test], ignore_index=True)
    sigma_full = pd.concat([sigma_train, sigma_test], ignore_index=True)
    summarize_optional_evidence(table_dir)

    binary_freq = position_frequency_table(binary_full, "binary", BINARY_CLASS_NAMES)
    sigma_freq = position_frequency_table(sigma_full, "sigma", SIGMA_CLASS_NAMES)
    binary_freq.to_csv(table_dir / "analysis_binary_position_frequency.csv", index=False)
    sigma_freq.to_csv(table_dir / "analysis_sigma_position_frequency.csv", index=False)
    consensus_df = pd.concat(
        [consensus_sequences(binary_freq), consensus_sequences(sigma_freq)],
        ignore_index=True,
    )
    consensus_df.to_csv(table_dir / "analysis_consensus_sequences.csv", index=False)
    plot_position_frequency_heatmap(
        binary_freq,
        BINARY_CLASS_NAMES,
        "Binary consensus frequency by position",
        figure_dir / "analysis_binary_position_frequency_heatmap.png",
    )
    plot_position_frequency_heatmap(
        sigma_freq,
        SIGMA_CLASS_NAMES,
        "Sigma consensus frequency by position",
        figure_dir / "analysis_sigma_position_frequency_heatmap.png",
    )

    binary_sim = save_similarity(
        "binary",
        BINARY_CLASS_NAMES,
        mean_kmer_profiles(binary_full, BINARY_CLASS_NAMES, k=5),
        table_dir,
        figure_dir,
    )
    sigma_sim = save_similarity(
        "sigma",
        SIGMA_CLASS_NAMES,
        mean_kmer_profiles(sigma_full, SIGMA_CLASS_NAMES, k=3),
        table_dir,
        figure_dir,
    )

    imbalance_df = class_imbalance_summary(binary_train, binary_test, sigma_train, sigma_test)
    imbalance_df.to_csv(table_dir / "analysis_class_imbalance_summary.csv", index=False)
    plot_sigma_imbalance(imbalance_df, figure_dir)

    diagnostic_df = build_sigma_diagnostic(table_dir, figure_dir)
    confusion_df = build_confusion_pairs(table_dir, figure_dir)
    report_path = write_report(table_dir, binary_sim, sigma_sim, imbalance_df, diagnostic_df, confusion_df)

    saved_files = [
        table_dir / "analysis_binary_position_frequency.csv",
        table_dir / "analysis_sigma_position_frequency.csv",
        table_dir / "analysis_consensus_sequences.csv",
        table_dir / "analysis_sigma_kmer_cosine_similarity.csv",
        table_dir / "analysis_binary_kmer_cosine_similarity.csv",
        table_dir / "analysis_class_imbalance_summary.csv",
        table_dir / "analysis_sigma_class_diagnostic.csv",
        table_dir / "analysis_sigma_confusion_pairs.csv",
        report_path,
    ]
    print("\nSaved analysis files")
    for path in saved_files:
        if path.exists():
            print(f"  {path}")

    print("\nSigma diagnostic table")
    if diagnostic_df.empty:
        print("  Not available")
    else:
        print(diagnostic_df.to_string(index=False))

    print("\nTop sigma confusion pairs")
    if confusion_df.empty:
        print("  Not available")
    else:
        print(confusion_df.head(12).to_string(index=False))

    print("\nPublication decision summary")
    print("  Proceed only as a benchmark/reliability paper, not as a high-performance classifier paper.")
    print("  Do not submit to a well-known journal until sigma performance or biological validation improves.")
    print("\nBiological and error analysis completed successfully.")


if __name__ == "__main__":
    main()
