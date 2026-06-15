from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (  # noqa: E402
    TemperatureScaler,
    calibrate_and_evaluate,
    plot_reliability_diagram,
    save_calibrated_probabilities,
    save_calibration_metrics,
)
from src.conformal import (  # noqa: E402
    run_conformal_prediction,
    save_conformal_metrics,
    save_conformal_results,
)
from src.data_loader import (  # noqa: E402
    load_sigma_dataset,
    print_dataset_report,
    split_train_val_calibration,
)
from src.encoding import create_torch_dataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    evaluate_classification,
    mc_dropout_predict,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_training_history,
    plot_uncertainty_histogram,
    predict_classes_from_proba,
    predict_logits,
    predict_proba,
    save_classification_report,
    save_metrics_table,
    save_uncertainty_results,
)
from src.models import CNNPromoterClassifier  # noqa: E402
from src.train import create_data_loader, get_device, set_seed, train_model  # noqa: E402


@dataclass
class ImprovedSigmaExperimentConfig:
    seed: int = 42
    max_len: int = 81
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 5e-4
    patience: int = 15
    dropout: float = 0.5
    mc_dropout_passes: int = 30
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    prefix: str = "sigma_improved"
    class_names: tuple = (
        "Sigma24",
        "Sigma28",
        "Sigma32",
        "Sigma38",
        "Sigma54",
        "Sigma70",
    )


def ensure_results_dirs(config):
    model_dir = config.results_dir / "models"
    figure_dir = config.results_dir / "figures"
    table_dir = config.results_dir / "tables"

    for directory in [model_dir, figure_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return model_dir, figure_dir, table_dir


def load_and_split_data(config):
    print("Loading sigma-factor dataset...")
    train_df, test_df = load_sigma_dataset(config.data_dir)
    print_dataset_report(train_df, "Sigma Original Train")
    print_dataset_report(test_df, "Sigma Untouched Test")

    print("\nSplitting original train into train/validation/calibration...")
    train_split, val_split, calibration_split = split_train_val_calibration(train_df)
    print(f"Train split: {len(train_split)}")
    print(f"Validation split: {len(val_split)}")
    print(f"Calibration split: {len(calibration_split)}")
    print(f"Untouched test set: {len(test_df)}")

    return train_split, val_split, calibration_split, test_df


def create_balanced_train_loader(train_dataset, train_labels, batch_size):
    labels = np.asarray(train_labels, dtype=int)
    class_counts = np.bincount(labels)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def create_loaders(config, train_split, val_split, calibration_split, test_df):
    print("\nCreating datasets and dataloaders...")
    train_dataset = create_torch_dataset(train_split, max_len=config.max_len)
    val_dataset = create_torch_dataset(val_split, max_len=config.max_len)
    calibration_dataset = create_torch_dataset(
        calibration_split, max_len=config.max_len
    )
    test_dataset = create_torch_dataset(test_df, max_len=config.max_len)

    train_loader = create_balanced_train_loader(
        train_dataset,
        train_split["label"],
        batch_size=config.batch_size,
    )
    val_loader = create_data_loader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    calibration_loader = create_data_loader(
        calibration_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = create_data_loader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    print("Balanced training DataLoader uses WeightedRandomSampler.")
    print("Validation, calibration, and test DataLoaders are normal non-shuffled loaders.")
    return train_loader, val_loader, calibration_loader, test_loader


def train_improved_sigma_cnn(
    config, train_loader, val_loader, train_split, model_path, history_path
):
    print("\nTraining improved sigma CNN...")
    device = get_device()
    print(f"Using device: {device}")
    print("Class weights remain enabled in CrossEntropyLoss.")

    model = CNNPromoterClassifier(num_classes=6, dropout=config.dropout)
    model, history_df = train_model(
        model,
        train_loader,
        val_loader,
        num_classes=6,
        epochs=config.epochs,
        lr=config.learning_rate,
        patience=config.patience,
        device=device,
        use_class_weights=True,
        class_labels=train_split["label"],
        model_save_path=model_path,
    )

    history_df.to_csv(history_path, index=False)
    print(f"Saved best model: {model_path}")
    print(f"Saved training history: {history_path}")

    return model, history_df, device


def print_per_class_metrics(metrics):
    print("Per-class precision/recall/F1:")
    precision = metrics["per_class_precision"]
    recall = metrics["per_class_recall"]
    f1 = metrics["per_class_f1"]
    for class_name in precision:
        print(
            f"  {class_name}: "
            f"precision={precision[class_name]:.4f}, "
            f"recall={recall[class_name]:.4f}, "
            f"f1={f1[class_name]:.4f}"
        )


def evaluate_test_set(config, model, test_loader, figure_dir, table_dir, device):
    print("\nEvaluating improved model on untouched sigma test set...")
    test_proba, test_labels = predict_proba(model, test_loader, device=device)
    test_pred = predict_classes_from_proba(test_proba)
    metrics = evaluate_classification(
        test_labels, test_pred, test_proba, class_names=list(config.class_names)
    )

    save_metrics_table(metrics, table_dir / f"{config.prefix}_metrics.csv")
    save_classification_report(
        test_labels,
        test_pred,
        list(config.class_names),
        table_dir / f"{config.prefix}_classification_report.csv",
    )
    plot_confusion_matrix(
        test_labels,
        test_pred,
        list(config.class_names),
        save_path=figure_dir / f"{config.prefix}_confusion_matrix.png",
    )
    curve_paths = plot_roc_pr_curves(
        test_labels,
        test_proba,
        list(config.class_names),
        save_dir=figure_dir,
        prefix=config.prefix,
    )

    print("Final improved overall metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    print_per_class_metrics(metrics)
    print("Generated ROC/PR curve paths:")
    for key, value in curve_paths.items():
        print(f"  {key}: {value}")

    return test_labels, test_proba, test_pred, metrics, curve_paths


def create_uncertainty_by_class_summary(uncertainty_df):
    return (
        uncertainty_df.groupby("true_label_name")
        .agg(
            count=("true_label_name", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_variance=("variance", "mean"),
            mean_mutual_information=("mutual_information", "mean"),
        )
        .reset_index()
    )


def plot_uncertainty_by_class(summary_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    plot_specs = [
        ("accuracy", "Accuracy"),
        ("mean_confidence", "Mean Confidence"),
        ("mean_entropy", "Mean Entropy"),
        ("mean_mutual_information", "Mean Mutual Information"),
    ]

    for ax, (column, title) in zip(axes, plot_specs):
        ax.bar(summary_df["true_label_name"], summary_df[column])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def run_mc_dropout(config, model, test_loader, figure_dir, table_dir, device):
    print("\nRunning MC Dropout uncertainty estimation...")
    mc_result = mc_dropout_predict(
        model, test_loader, T=config.mc_dropout_passes, device=device
    )
    uncertainty_df = save_uncertainty_results(
        mc_result,
        list(config.class_names),
        table_dir / f"{config.prefix}_uncertainty_results.csv",
    )
    plot_uncertainty_histogram(
        uncertainty_df,
        save_path=figure_dir / f"{config.prefix}_uncertainty_histogram.png",
        column="entropy",
    )

    uncertainty_summary = create_uncertainty_by_class_summary(uncertainty_df)
    uncertainty_summary.to_csv(
        table_dir / f"{config.prefix}_uncertainty_by_class.csv", index=False
    )
    plot_uncertainty_by_class(
        uncertainty_summary, figure_dir / f"{config.prefix}_uncertainty_by_class.png"
    )

    print("Uncertainty summary by class:")
    print(uncertainty_summary.to_string(index=False))

    return mc_result, uncertainty_df, uncertainty_summary


def run_temperature_scaling(
    config,
    model,
    val_loader,
    calibration_loader,
    test_loader,
    test_labels,
    figure_dir,
    table_dir,
    device,
):
    print("\nRunning temperature scaling calibration...")
    val_logits, val_labels = predict_logits(model, val_loader, device=device)
    test_logits, test_labels_from_loader = predict_logits(model, test_loader, device=device)

    calibration_result = calibrate_and_evaluate(
        val_logits,
        val_labels,
        test_logits,
        test_labels_from_loader,
        n_bins=15,
    )

    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_before"],
        save_path=figure_dir / f"{config.prefix}_reliability_before.png",
        n_bins=15,
        title="Improved Sigma Reliability Before Temperature Scaling",
    )
    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_after"],
        save_path=figure_dir / f"{config.prefix}_reliability_after.png",
        n_bins=15,
        title="Improved Sigma Reliability After Temperature Scaling",
    )
    save_calibration_metrics(
        calibration_result["before"],
        calibration_result["after"],
        calibration_result["temperature"],
        table_dir / f"{config.prefix}_calibration_metrics.csv",
    )
    save_calibrated_probabilities(
        test_labels,
        calibration_result["test_proba_before"],
        calibration_result["test_proba_after"],
        table_dir / f"{config.prefix}_calibrated_probabilities.csv",
    )

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    calibration_logits, calibration_labels = predict_logits(
        model, calibration_loader, device=device
    )
    calibrated_calibration_proba = scaler.predict_proba(calibration_logits)
    calibrated_test_proba = scaler.predict_proba(test_logits)

    print("Final calibration metrics:")
    print(f"  temperature: {calibration_result['temperature']}")
    print("  before:")
    for key, value in calibration_result["before"].items():
        print(f"    {key}: {value}")
    print("  after:")
    for key, value in calibration_result["after"].items():
        print(f"    {key}: {value}")

    return {
        "calibration_result": calibration_result,
        "calibration_labels": calibration_labels,
        "calibrated_calibration_proba": calibrated_calibration_proba,
        "calibrated_test_proba": calibrated_test_proba,
    }


def create_conformal_classwise_summary(config, test_labels, result_90, result_95):
    metrics_90 = result_90["metrics"]
    metrics_95 = result_95["metrics"]
    rows = []

    for class_index, class_name in enumerate(config.class_names):
        rows.append(
            {
                "class_name": class_name,
                "test_count": int(np.sum(np.asarray(test_labels) == class_index)),
                "coverage_90": metrics_90["class_wise_coverage"].get(class_name),
                "coverage_95": metrics_95["class_wise_coverage"].get(class_name),
                "average_set_size_90": metrics_90[
                    "class_wise_average_set_size"
                ].get(class_name),
                "average_set_size_95": metrics_95[
                    "class_wise_average_set_size"
                ].get(class_name),
            }
        )

    return pd.DataFrame(rows)


def run_conformal(
    config,
    calibration_labels,
    calibrated_calibration_proba,
    test_labels,
    calibrated_test_proba,
    table_dir,
):
    print("\nRunning split conformal prediction with calibrated probabilities...")
    results_by_alpha = []
    outputs = {}

    for alpha, suffix in [(0.1, "90"), (0.05, "95")]:
        result = run_conformal_prediction(
            calibration_labels,
            calibrated_calibration_proba,
            test_labels,
            calibrated_test_proba,
            class_names=list(config.class_names),
            alpha=alpha,
        )
        save_conformal_results(
            result,
            test_labels,
            calibrated_test_proba,
            table_dir / f"{config.prefix}_conformal_predictions_{suffix}.csv",
            class_names=list(config.class_names),
        )
        results_by_alpha.append(result)
        outputs[alpha] = result

        print(f"Conformal metrics at alpha={alpha}:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")

    save_conformal_metrics(
        results_by_alpha, table_dir / f"{config.prefix}_conformal_metrics.csv"
    )

    classwise_summary = create_conformal_classwise_summary(
        config, test_labels, outputs[0.1], outputs[0.05]
    )
    classwise_summary.to_csv(
        table_dir / f"{config.prefix}_conformal_classwise_summary.csv", index=False
    )

    print("Conformal class-wise coverage:")
    print(classwise_summary.to_string(index=False))

    return outputs, classwise_summary


def _read_metric_csv(path):
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    values = {}
    for _, row in df.iterrows():
        metric = row["metric"]
        value = row["value"]
        try:
            values[metric] = float(value)
        except (TypeError, ValueError):
            values[metric] = np.nan
    return values


def save_original_vs_improved_comparison(config, improved_metrics, table_dir):
    original_path = table_dir / "sigma_metrics.csv"
    original_metrics = _read_metric_csv(original_path)
    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "mcc",
        "auroc_macro_ovr",
        "auprc_macro_ovr",
    ]

    rows = []
    for experiment, metrics in [
        ("original_sigma_cnn", original_metrics),
        ("improved_sigma_cnn", improved_metrics),
    ]:
        row = {"experiment": experiment}
        for column in metric_columns:
            row[column] = metrics.get(column, np.nan)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_path = table_dir / "sigma_comparison_original_vs_improved.csv"
    comparison_df.to_csv(comparison_path, index=False)

    if original_metrics:
        print("Original vs improved comparison table:")
    else:
        print(
            "Original sigma metrics file was not found; comparison includes "
            "NaN values for original_sigma_cnn."
        )
        print("Original vs improved comparison table:")
    print(comparison_df.to_string(index=False))
    return comparison_df


def run_experiment(config=None):
    if config is None:
        config = ImprovedSigmaExperimentConfig()

    set_seed(config.seed)
    model_dir, figure_dir, table_dir = ensure_results_dirs(config)

    train_split, val_split, calibration_split, test_df = load_and_split_data(config)
    train_loader, val_loader, calibration_loader, test_loader = create_loaders(
        config, train_split, val_split, calibration_split, test_df
    )

    model, history_df, device = train_improved_sigma_cnn(
        config,
        train_loader,
        val_loader,
        train_split,
        model_dir / f"{config.prefix}_cnn.pt",
        table_dir / f"{config.prefix}_training_history.csv",
    )
    plot_training_history(
        history_df, save_path=figure_dir / f"{config.prefix}_training_history.png"
    )

    test_labels, test_proba, test_pred, test_metrics, curve_paths = evaluate_test_set(
        config, model, test_loader, figure_dir, table_dir, device
    )
    mc_result, uncertainty_df, uncertainty_summary = run_mc_dropout(
        config, model, test_loader, figure_dir, table_dir, device
    )
    calibration_outputs = run_temperature_scaling(
        config,
        model,
        val_loader,
        calibration_loader,
        test_loader,
        test_labels,
        figure_dir,
        table_dir,
        device,
    )
    conformal_outputs, conformal_classwise_summary = run_conformal(
        config,
        calibration_outputs["calibration_labels"],
        calibration_outputs["calibrated_calibration_proba"],
        test_labels,
        calibration_outputs["calibrated_test_proba"],
        table_dir,
    )
    comparison_df = save_original_vs_improved_comparison(
        config, test_metrics, table_dir
    )

    print("\nImproved sigma experiment completed successfully.")

    return {
        "model": model,
        "history": history_df,
        "test_labels": test_labels,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
        "curve_paths": curve_paths,
        "mc_result": mc_result,
        "uncertainty": uncertainty_df,
        "uncertainty_summary": uncertainty_summary,
        "calibration": calibration_outputs,
        "conformal": conformal_outputs,
        "conformal_classwise_summary": conformal_classwise_summary,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    run_experiment()
