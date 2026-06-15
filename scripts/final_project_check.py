from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_FILES = [
    "results/models/binary_cnn.pt",
    "results/models/sigma_cnn.pt",
    "results/models/sigma_improved_cnn.pt",
    "results/models/sigma_safe_cnn.pt",
    "results/tables/dataset_summary.csv",
    "results/tables/dataset_class_distribution.csv",
    "results/tables/binary_metrics.csv",
    "results/tables/binary_classification_report.csv",
    "results/tables/binary_uncertainty_results.csv",
    "results/tables/binary_calibration_metrics.csv",
    "results/tables/binary_conformal_metrics.csv",
    "results/tables/sigma_metrics.csv",
    "results/tables/sigma_improved_metrics.csv",
    "results/tables/sigma_safe_metrics.csv",
    "results/tables/sigma_comparison_all.csv",
    "results/tables/sigma_safe_classification_report.csv",
    "results/tables/sigma_safe_uncertainty_by_class.csv",
    "results/tables/sigma_safe_calibration_metrics.csv",
    "results/tables/sigma_safe_conformal_metrics.csv",
    "results/tables/sigma_safe_conformal_classwise_summary.csv",
    "results/figures/binary_confusion_matrix.png",
    "results/figures/binary_roc_curve.png",
    "results/figures/binary_pr_curve.png",
    "results/figures/binary_reliability_before.png",
    "results/figures/binary_reliability_after.png",
    "results/figures/sigma_safe_confusion_matrix.png",
    "results/figures/sigma_safe_reliability_before.png",
    "results/figures/sigma_safe_reliability_after.png",
    "README.md",
]


METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "auroc",
    "auprc",
    "auroc_macro_ovr",
    "auprc_macro_ovr",
]


def ensure_output_dirs():
    table_dir = PROJECT_ROOT / "results" / "tables"
    summary_dir = PROJECT_ROOT / "results"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, summary_dir


def create_file_check(table_dir):
    rows = []
    for relative_path in REQUIRED_FILES:
        rows.append(
            {
                "file_path": relative_path,
                "exists": (PROJECT_ROOT / relative_path).exists(),
            }
        )

    file_check_df = pd.DataFrame(rows)
    file_check_df.to_csv(table_dir / "final_file_check.csv", index=False)
    return file_check_df


def read_metrics_csv(relative_path):
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    metrics = {}
    for _, row in df.iterrows():
        metric = row.get("metric")
        value = row.get("value")
        try:
            metrics[metric] = float(value)
        except (TypeError, ValueError):
            metrics[metric] = np.nan
    return metrics


def create_final_metrics_summary(table_dir):
    experiments = [
        ("binary_cnn", "results/tables/binary_metrics.csv"),
        ("original_sigma_cnn", "results/tables/sigma_metrics.csv"),
        ("improved_sampler_sigma_cnn", "results/tables/sigma_improved_metrics.csv"),
        ("safe_f1_selected_sigma_cnn", "results/tables/sigma_safe_metrics.csv"),
    ]

    rows = []
    for experiment_name, metrics_path in experiments:
        metrics = read_metrics_csv(metrics_path)
        row = {"experiment": experiment_name}
        for column in METRIC_COLUMNS:
            row[column] = metrics.get(column, np.nan)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(table_dir / "final_metrics_summary.csv", index=False)
    return summary_df


def format_metric(metrics_df, experiment, metric):
    row = metrics_df.loc[metrics_df["experiment"] == experiment]
    if row.empty:
        return "N/A"
    value = row.iloc[0].get(metric)
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def create_research_summary(metrics_df, summary_dir):
    binary_experiment = "binary_cnn"
    sigma_experiment = "safe_f1_selected_sigma_cnn"

    markdown = f"""# BayesSigma: Trustworthy Promoter Classification via Calibrated Deep Learning and Conformal Prediction

## Dataset Summary

Binary:

- Train: 5258
- Test: 1315
- Sequence length: 81 bp
- Invalid DNA characters: 0

Sigma:

- Train: 2315
- Test: 583
- Sequence length: 81 bp
- Invalid DNA characters: 0

## Final Binary Result

Final binary model: `binary_cnn`

- Accuracy: {format_metric(metrics_df, binary_experiment, "accuracy")}
- Precision macro: {format_metric(metrics_df, binary_experiment, "precision_macro")}
- Recall macro: {format_metric(metrics_df, binary_experiment, "recall_macro")}
- Macro F1: {format_metric(metrics_df, binary_experiment, "f1_macro")}
- Weighted F1: {format_metric(metrics_df, binary_experiment, "f1_weighted")}
- MCC: {format_metric(metrics_df, binary_experiment, "mcc")}
- AUROC: {format_metric(metrics_df, binary_experiment, "auroc")}
- AUPRC: {format_metric(metrics_df, binary_experiment, "auprc")}

## Final Sigma Result

Final sigma model: `safe_f1_selected_sigma_cnn`

- Accuracy: {format_metric(metrics_df, sigma_experiment, "accuracy")}
- Precision macro: {format_metric(metrics_df, sigma_experiment, "precision_macro")}
- Recall macro: {format_metric(metrics_df, sigma_experiment, "recall_macro")}
- Macro F1: {format_metric(metrics_df, sigma_experiment, "f1_macro")}
- Weighted F1: {format_metric(metrics_df, sigma_experiment, "f1_weighted")}
- MCC: {format_metric(metrics_df, sigma_experiment, "mcc")}
- AUROC macro OVR: {format_metric(metrics_df, sigma_experiment, "auroc_macro_ovr")}
- AUPRC macro OVR: {format_metric(metrics_df, sigma_experiment, "auprc_macro_ovr")}

## Sigma Experiment Comparison

- `original_sigma_cnn`: baseline class-weighted CNN.
- `improved_sampler_sigma_cnn`: weighted sampler plus class weights; this over-corrected imbalance and harmed Sigma70.
- `safe_f1_selected_sigma_cnn`: normal shuffled batches, class weights, lower learning rate, and model selection by validation macro F1.

The safe F1-selected sigma CNN is the final sigma model because it gives the best balanced sigma result among the completed sigma experiments.

## Main Research Interpretation

Binary promoter classification is acceptable. Sigma-factor classification is more difficult because classes are imbalanced and sigma-factor promoter patterns can be biologically similar. Calibration and conformal prediction provide reliability information beyond ordinary accuracy scores. The safe sigma model gives the best balanced sigma result. WeightedRandomSampler with class weights over-corrected imbalance and harmed Sigma70.

## Final Claim

BayesSigma is not only a classifier. It is a trustworthy prediction framework that outputs:

- class prediction
- calibrated confidence
- uncertainty estimates
- conformal prediction sets

## Limitations

- No cross-species validation
- No wet-lab validation
- Dataset-specific
- Sigma classes are imbalanced
- Small classes like Sigma54 remain difficult
"""

    summary_path = summary_dir / "final_research_summary.md"
    summary_path.write_text(markdown, encoding="utf-8")
    return summary_path


def run_final_project_check():
    table_dir, summary_dir = ensure_output_dirs()

    file_check_df = create_file_check(table_dir)
    metrics_summary_df = create_final_metrics_summary(table_dir)
    summary_path = create_research_summary(metrics_summary_df, summary_dir)

    print("File check table:")
    print(file_check_df.to_string(index=False))

    print("\nFinal metrics summary:")
    print(metrics_summary_df.to_string(index=False))

    print(f"\nFinal markdown summary: {summary_path}")
    print("\nFinal project check completed successfully.")

    return {
        "file_check": file_check_df,
        "metrics_summary": metrics_summary_df,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_final_project_check()
