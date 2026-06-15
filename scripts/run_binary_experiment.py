from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

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
    load_binary_dataset,
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
class BinaryExperimentConfig:
    seed: int = 42
    max_len: int = 81
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 8
    dropout: float = 0.3
    mc_dropout_passes: int = 30
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    class_names: tuple = ("Non-Promoter", "Promoter")


def ensure_results_dirs(config):
    model_dir = config.results_dir / "models"
    figure_dir = config.results_dir / "figures"
    table_dir = config.results_dir / "tables"

    for directory in [model_dir, figure_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return model_dir, figure_dir, table_dir


def load_and_split_data(config):
    print("Loading binary dataset...")
    train_df, test_df = load_binary_dataset(config.data_dir)
    print_dataset_report(train_df, "Binary Original Train")
    print_dataset_report(test_df, "Binary Untouched Test")

    print("\nSplitting original train into train/validation/calibration...")
    train_split, val_split, calibration_split = split_train_val_calibration(train_df)
    print(f"Train split: {len(train_split)}")
    print(f"Validation split: {len(val_split)}")
    print(f"Calibration split: {len(calibration_split)}")
    print(f"Untouched test set: {len(test_df)}")

    return train_split, val_split, calibration_split, test_df


def create_loaders(config, train_split, val_split, calibration_split, test_df):
    print("\nCreating PyTorch datasets and dataloaders...")
    train_dataset = create_torch_dataset(train_split, max_len=config.max_len)
    val_dataset = create_torch_dataset(val_split, max_len=config.max_len)
    calibration_dataset = create_torch_dataset(
        calibration_split, max_len=config.max_len
    )
    test_dataset = create_torch_dataset(test_df, max_len=config.max_len)

    train_loader = create_data_loader(
        train_dataset, batch_size=config.batch_size, shuffle=True
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

    return train_loader, val_loader, calibration_loader, test_loader


def train_binary_cnn(config, train_loader, val_loader, train_split, model_path, history_path):
    print("\nTraining binary CNN...")
    device = get_device()
    print(f"Using device: {device}")

    model = CNNPromoterClassifier(num_classes=2, dropout=config.dropout)
    model, history_df = train_model(
        model,
        train_loader,
        val_loader,
        num_classes=2,
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


def evaluate_test_set(config, model, test_loader, figure_dir, table_dir, device):
    print("\nEvaluating on untouched test set...")
    test_proba, test_labels = predict_proba(model, test_loader, device=device)
    test_pred = predict_classes_from_proba(test_proba)
    metrics = evaluate_classification(
        test_labels, test_pred, test_proba, class_names=list(config.class_names)
    )

    save_metrics_table(metrics, table_dir / "binary_metrics.csv")
    save_classification_report(
        test_labels,
        test_pred,
        list(config.class_names),
        table_dir / "binary_classification_report.csv",
    )
    plot_confusion_matrix(
        test_labels,
        test_pred,
        list(config.class_names),
        save_path=figure_dir / "binary_confusion_matrix.png",
    )
    plot_roc_pr_curves(
        test_labels,
        test_proba,
        list(config.class_names),
        save_dir=figure_dir,
        prefix="binary",
    )

    print("Final test metrics:")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    return test_labels, test_proba, test_pred, metrics


def run_mc_dropout(config, model, test_loader, figure_dir, table_dir, device):
    print("\nRunning MC Dropout uncertainty estimation...")
    mc_result = mc_dropout_predict(
        model, test_loader, T=config.mc_dropout_passes, device=device
    )
    uncertainty_df = save_uncertainty_results(
        mc_result,
        list(config.class_names),
        table_dir / "binary_uncertainty_results.csv",
    )
    plot_uncertainty_histogram(
        uncertainty_df,
        save_path=figure_dir / "binary_uncertainty_histogram.png",
        column="entropy",
    )
    print(f"Saved uncertainty results: {table_dir / 'binary_uncertainty_results.csv'}")
    return mc_result, uncertainty_df


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
        save_path=figure_dir / "binary_reliability_before.png",
        n_bins=15,
        title="Binary Reliability Before Temperature Scaling",
    )
    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_after"],
        save_path=figure_dir / "binary_reliability_after.png",
        n_bins=15,
        title="Binary Reliability After Temperature Scaling",
    )
    save_calibration_metrics(
        calibration_result["before"],
        calibration_result["after"],
        calibration_result["temperature"],
        table_dir / "binary_calibration_metrics.csv",
    )
    save_calibrated_probabilities(
        test_labels,
        calibration_result["test_proba_before"],
        calibration_result["test_proba_after"],
        table_dir / "binary_calibrated_probabilities.csv",
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
            table_dir / f"binary_conformal_predictions_{suffix}.csv",
            class_names=list(config.class_names),
        )
        results_by_alpha.append(result)
        outputs[alpha] = result

        print(f"Conformal metrics at alpha={alpha}:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")

    save_conformal_metrics(results_by_alpha, table_dir / "binary_conformal_metrics.csv")
    return outputs


def run_experiment(config=None):
    if config is None:
        config = BinaryExperimentConfig()

    set_seed(config.seed)
    model_dir, figure_dir, table_dir = ensure_results_dirs(config)

    train_split, val_split, calibration_split, test_df = load_and_split_data(config)
    train_loader, val_loader, calibration_loader, test_loader = create_loaders(
        config, train_split, val_split, calibration_split, test_df
    )

    model, history_df, device = train_binary_cnn(
        config,
        train_loader,
        val_loader,
        train_split,
        model_dir / "binary_cnn.pt",
        table_dir / "binary_training_history.csv",
    )
    plot_training_history(
        history_df, save_path=figure_dir / "binary_training_history.png"
    )

    test_labels, test_proba, test_pred, test_metrics = evaluate_test_set(
        config, model, test_loader, figure_dir, table_dir, device
    )
    mc_result, uncertainty_df = run_mc_dropout(
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
    conformal_outputs = run_conformal(
        config,
        calibration_outputs["calibration_labels"],
        calibration_outputs["calibrated_calibration_proba"],
        test_labels,
        calibration_outputs["calibrated_test_proba"],
        table_dir,
    )

    print("\nBinary experiment completed successfully.")

    return {
        "model": model,
        "history": history_df,
        "test_labels": test_labels,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
        "mc_result": mc_result,
        "uncertainty": uncertainty_df,
        "calibration": calibration_outputs,
        "conformal": conformal_outputs,
    }


if __name__ == "__main__":
    run_experiment()
