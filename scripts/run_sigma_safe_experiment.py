from dataclasses import dataclass
from pathlib import Path
import copy
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

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
from src.train import (  # noqa: E402
    compute_class_weights,
    create_data_loader,
    evaluate_on_loader,
    get_device,
    set_seed,
    train_one_epoch,
)


@dataclass
class SafeSigmaExperimentConfig:
    seed: int = 42
    max_len: int = 81
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 3e-4
    patience: int = 20
    dropout: float = 0.3
    mc_dropout_passes: int = 30
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    prefix: str = "sigma_safe"
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


def create_loaders(config, train_split, val_split, calibration_split, test_df):
    print("\nCreating normal PyTorch dataloaders...")
    train_loader = create_data_loader(
        create_torch_dataset(train_split, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = create_data_loader(
        create_torch_dataset(val_split, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    calibration_loader = create_data_loader(
        create_torch_dataset(calibration_split, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = create_data_loader(
        create_torch_dataset(test_df, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    print("Training loader is shuffled normally. No WeightedRandomSampler is used.")
    return train_loader, val_loader, calibration_loader, test_loader


def train_model_select_by_f1(
    model,
    train_loader,
    val_loader,
    num_classes,
    class_labels,
    epochs=100,
    lr=3e-4,
    patience=20,
    device=None,
    model_save_path=None,
):
    if device is None:
        device = get_device()
    device = torch.device(device)
    model = model.to(device)

    class_weights = compute_class_weights(class_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, patience // 2),
    )

    best_f1 = -np.inf
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_on_loader(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1_macro"]
        scheduler.step(val_f1)
        learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_f1,
                "learning_rate": learning_rate,
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            if model_save_path is not None:
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs based on validation macro F1.")
            break

    model.load_state_dict(best_state_dict)
    model.to(device)
    return model, pd.DataFrame(history)


def train_safe_sigma_cnn(config, train_loader, val_loader, train_split, model_path, history_path):
    print("\nTraining safe sigma CNN selected by validation macro F1...")
    device = get_device()
    print(f"Using device: {device}")

    model = CNNPromoterClassifier(num_classes=6, dropout=config.dropout)
    model, history_df = train_model_select_by_f1(
        model,
        train_loader,
        val_loader,
        num_classes=6,
        class_labels=train_split["label"],
        epochs=config.epochs,
        lr=config.learning_rate,
        patience=config.patience,
        device=device,
        model_save_path=model_path,
    )
    history_df.to_csv(history_path, index=False)
    print(f"Saved best model: {model_path}")
    print(f"Saved training history: {history_path}")
    return model, history_df, device


def print_per_class_metrics(metrics):
    print("Per-class precision/recall/F1:")
    for class_name in metrics["per_class_precision"]:
        print(
            f"  {class_name}: "
            f"precision={metrics['per_class_precision'][class_name]:.4f}, "
            f"recall={metrics['per_class_recall'][class_name]:.4f}, "
            f"f1={metrics['per_class_f1'][class_name]:.4f}"
        )


def evaluate_test_set(config, model, test_loader, figure_dir, table_dir, device):
    print("\nEvaluating safe model on untouched sigma test set...")
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
    plot_roc_pr_curves(
        test_labels,
        test_proba,
        list(config.class_names),
        save_dir=figure_dir,
        prefix=config.prefix,
    )

    print("Final safe sigma overall metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    print_per_class_metrics(metrics)
    return test_labels, test_proba, test_pred, metrics


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
    for ax, (column, title) in zip(
        axes,
        [
            ("accuracy", "Accuracy"),
            ("mean_confidence", "Mean Confidence"),
            ("mean_entropy", "Mean Entropy"),
            ("mean_mutual_information", "Mean Mutual Information"),
        ],
    ):
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
        title="Safe Sigma Reliability Before Temperature Scaling",
    )
    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_after"],
        save_path=figure_dir / f"{config.prefix}_reliability_after.png",
        n_bins=15,
        title="Safe Sigma Reliability After Temperature Scaling",
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
    rows = []
    metrics_90 = result_90["metrics"]
    metrics_95 = result_95["metrics"]
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
    metrics = {}
    for _, row in df.iterrows():
        try:
            metrics[row["metric"]] = float(row["value"])
        except (TypeError, ValueError):
            metrics[row["metric"]] = np.nan
    return metrics


def save_all_sigma_comparison(safe_metrics, table_dir):
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
    experiments = [
        ("original_sigma_cnn", _read_metric_csv(table_dir / "sigma_metrics.csv")),
        (
            "improved_sampler_sigma_cnn",
            _read_metric_csv(table_dir / "sigma_improved_metrics.csv"),
        ),
        ("safe_f1_selected_sigma_cnn", safe_metrics),
    ]

    rows = []
    for experiment_name, metrics in experiments:
        row = {"experiment": experiment_name}
        for column in metric_columns:
            row[column] = metrics.get(column, np.nan)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(table_dir / "sigma_comparison_all.csv", index=False)
    print("Comparison table across all sigma experiments:")
    print(comparison_df.to_string(index=False))
    return comparison_df


def run_experiment(config=None):
    if config is None:
        config = SafeSigmaExperimentConfig()

    set_seed(config.seed)
    model_dir, figure_dir, table_dir = ensure_results_dirs(config)

    train_split, val_split, calibration_split, test_df = load_and_split_data(config)
    train_loader, val_loader, calibration_loader, test_loader = create_loaders(
        config, train_split, val_split, calibration_split, test_df
    )

    model, history_df, device = train_safe_sigma_cnn(
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

    test_labels, test_proba, test_pred, test_metrics = evaluate_test_set(
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
    comparison_df = save_all_sigma_comparison(test_metrics, table_dir)

    print("\nSafe sigma experiment completed successfully.")

    return {
        "model": model,
        "history": history_df,
        "test_labels": test_labels,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
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
