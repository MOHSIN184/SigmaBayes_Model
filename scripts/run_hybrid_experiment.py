from dataclasses import dataclass
from pathlib import Path
import copy
import itertools
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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
    load_sigma_dataset,
    print_dataset_report,
    split_train_val_calibration,
)
from src.encoding import one_hot_encode_sequence  # noqa: E402
from src.evaluate import (  # noqa: E402
    classification_report_dataframe,
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_training_history,
    plot_uncertainty_histogram,
    save_metrics_table,
    save_uncertainty_results,
)
from src.hybrid_models import HybridCNNKmerClassifier  # noqa: E402


BINARY_K = 5
SIGMA_K = 3
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


@dataclass
class HybridExperimentConfig:
    task: str
    prefix: str
    k: int
    num_classes: int
    class_names: tuple
    epochs: int
    seed: int = 42
    max_len: int = 81
    batch_size: int = 64
    learning_rate: float = 3e-4
    patience: int = 15
    dropout: float = 0.3
    mc_dropout_passes: int = 30
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"


class HybridDNADataset(Dataset):
    def __init__(self, df, k, max_len=81):
        self.ids = df["id"].tolist()
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].astype(int).to_numpy()
        self.max_len = max_len
        self.k = k
        self.kmers = generate_kmers(k)
        self.x_seq = np.stack(
            [one_hot_encode_sequence(seq, max_len=max_len) for seq in self.sequences],
            axis=0,
        ).astype(np.float32)
        self.x_kmer = build_kmer_matrix(self.sequences, k, normalize=True).astype(
            np.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(self.x_seq[index], dtype=torch.float32),
            torch.tensor(self.x_kmer[index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_kmers(k):
    return ["".join(chars) for chars in itertools.product("ACGT", repeat=k)]


def kmer_frequency_vector(sequence, kmers):
    sequence = str(sequence).upper()
    k = len(kmers[0]) if kmers else 0
    counts = np.zeros(len(kmers), dtype=np.float32)
    if k <= 0 or len(sequence) < k:
        return counts

    kmer_to_index = {kmer: index for index, kmer in enumerate(kmers)}
    total_windows = len(sequence) - k + 1
    valid_windows = 0
    for start in range(total_windows):
        kmer = sequence[start : start + k]
        index = kmer_to_index.get(kmer)
        if index is not None:
            counts[index] += 1.0
            valid_windows += 1

    if valid_windows > 0:
        counts /= float(valid_windows)
    return counts


def build_kmer_matrix(sequences, k, normalize=True):
    kmers = generate_kmers(k)
    matrix = np.stack([kmer_frequency_vector(seq, kmers) for seq in sequences], axis=0)
    if not normalize:
        lengths = np.array([max(len(str(seq)) - k + 1, 0) for seq in sequences])
        matrix = matrix * lengths[:, None]
    return matrix.astype(np.float32)


def ensure_results_dirs(config):
    model_dir = config.results_dir / "models"
    figure_dir = config.results_dir / "figures"
    table_dir = config.results_dir / "tables"
    for directory in [model_dir, figure_dir, table_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return model_dir, figure_dir, table_dir


def create_loaders(config, train_split, val_split, calibration_split, test_df):
    train_loader = DataLoader(
        HybridDNADataset(train_split, config.k, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        HybridDNADataset(val_split, config.k, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    calibration_loader = DataLoader(
        HybridDNADataset(calibration_split, config.k, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        HybridDNADataset(test_df, config.k, max_len=config.max_len),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, calibration_loader, test_loader


def compute_class_weights(labels, num_classes):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.asarray(labels, dtype=int),
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_one_hybrid_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_seq, x_kmer, y in tqdm(train_loader, desc="Training", leave=False):
        x_seq = x_seq.to(device)
        x_kmer = x_kmer.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_seq, x_kmer)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples if total_samples else 0.0


def evaluate_hybrid_loader(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for x_seq, x_kmer, y in tqdm(data_loader, desc="Evaluating", leave=False):
            x_seq = x_seq.to(device)
            x_kmer = x_kmer.to(device)
            y = y.to(device)
            logits = model(x_seq, x_kmer)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    return {
        "loss": total_loss / total_samples if total_samples else 0.0,
        "accuracy": accuracy_score(all_labels, all_predictions) if all_labels else 0.0,
        "f1_macro": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        if all_labels
        else 0.0,
    }


def train_hybrid_model_select_by_f1(
    model,
    train_loader,
    val_loader,
    num_classes,
    class_labels,
    epochs,
    lr,
    patience,
    device,
    model_save_path,
):
    model = model.to(device)
    class_weights = compute_class_weights(class_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 2)
    )

    best_f1 = -np.inf
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_hybrid_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate_hybrid_loader(model, val_loader, criterion, device)
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
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs based on validation macro F1.")
            break

    model.load_state_dict(best_state_dict)
    model.to(device)
    return model, pd.DataFrame(history)


def predict_hybrid_logits(model, data_loader, device):
    model = model.to(device)
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x_seq, x_kmer, y in data_loader:
            logits = model(x_seq.to(device), x_kmer.to(device))
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def predict_hybrid_proba(model, data_loader, device):
    logits, labels = predict_hybrid_logits(model, data_loader, device)
    probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    return probabilities, labels


def create_uncertainty_dataframe(mc_result, class_names):
    y_true = np.asarray(mc_result["y_true"], dtype=int)
    y_pred = np.asarray(mc_result["y_pred"], dtype=int)
    return pd.DataFrame(
        {
            "true_label": y_true,
            "true_label_name": [class_names[label] for label in y_true],
            "predicted_label": y_pred,
            "predicted_label_name": [class_names[label] for label in y_pred],
            "confidence": mc_result["confidence"],
            "entropy": mc_result["entropy"],
            "variance": mc_result["variance"],
            "mutual_information": mc_result["mutual_information"],
            "correct": y_true == y_pred,
        }
    )


def predictive_entropy(probs, eps=1e-12):
    probs = np.asarray(probs)
    return -np.sum(probs * np.log(probs + eps), axis=1)


def hybrid_mc_dropout_predict(model, data_loader, T, device):
    model = model.to(device)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    all_passes = []
    y_true = None
    with torch.no_grad():
        for _ in range(T):
            pass_probs = []
            pass_labels = []
            for x_seq, x_kmer, y in data_loader:
                logits = model(x_seq.to(device), x_kmer.to(device))
                probs = torch.softmax(logits, dim=1)
                pass_probs.append(probs.detach().cpu().numpy())
                pass_labels.append(y.detach().cpu().numpy())
            all_passes.append(np.concatenate(pass_probs, axis=0))
            if y_true is None:
                y_true = np.concatenate(pass_labels, axis=0)

    mc_proba = np.stack(all_passes, axis=0)
    mean_proba = mc_proba.mean(axis=0)
    expected_entropy = np.mean(
        -np.sum(mc_proba * np.log(mc_proba + 1e-12), axis=2), axis=0
    )
    entropy_mean = predictive_entropy(mean_proba)
    return {
        "mean_proba": mean_proba,
        "mc_proba": mc_proba,
        "y_true": y_true,
        "y_pred": np.argmax(mean_proba, axis=1),
        "confidence": np.max(mean_proba, axis=1),
        "entropy": entropy_mean,
        "variance": np.var(mc_proba, axis=0).mean(axis=1),
        "mutual_information": entropy_mean - expected_entropy,
    }


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
    plt.close(fig)


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


def run_calibration_and_conformal(
    config, model, val_loader, calibration_loader, test_loader, test_labels, figure_dir, table_dir, device
):
    print(f"\nRunning {config.task} temperature scaling and conformal prediction...")
    val_logits, val_labels = predict_hybrid_logits(model, val_loader, device)
    test_logits, test_labels_from_loader = predict_hybrid_logits(model, test_loader, device)
    calibration_result = calibrate_and_evaluate(
        val_logits, val_labels, test_logits, test_labels_from_loader, n_bins=15
    )

    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_before"],
        save_path=figure_dir / f"{config.prefix}_reliability_before.png",
        n_bins=15,
        title=f"{config.task.title()} Hybrid Reliability Before Temperature Scaling",
    )
    plot_reliability_diagram(
        test_labels_from_loader,
        calibration_result["test_proba_after"],
        save_path=figure_dir / f"{config.prefix}_reliability_after.png",
        n_bins=15,
        title=f"{config.task.title()} Hybrid Reliability After Temperature Scaling",
    )
    plt.close("all")
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
    scaler.fit(val_logits, val_labels, device=device)
    calibration_logits, calibration_labels = predict_hybrid_logits(
        model, calibration_loader, device
    )
    calibrated_calibration_proba = scaler.predict_proba(calibration_logits)
    calibrated_test_proba = scaler.predict_proba(test_logits)

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

    save_conformal_metrics(
        results_by_alpha, table_dir / f"{config.prefix}_conformal_metrics.csv"
    )
    if config.num_classes > 2:
        classwise_summary = create_conformal_classwise_summary(
            config, test_labels, outputs[0.1], outputs[0.05]
        )
        classwise_summary.to_csv(
            table_dir / f"{config.prefix}_conformal_classwise_summary.csv", index=False
        )

    return calibration_result, outputs


def run_mc_dropout(config, model, test_loader, figure_dir, table_dir, device):
    print(f"\nRunning {config.task} MC Dropout uncertainty...")
    mc_result = hybrid_mc_dropout_predict(
        model, test_loader, T=config.mc_dropout_passes, device=device
    )
    uncertainty_df = create_uncertainty_dataframe(mc_result, list(config.class_names))
    uncertainty_df.to_csv(
        table_dir / f"{config.prefix}_uncertainty_results.csv", index=False
    )
    plot_uncertainty_histogram(
        uncertainty_df,
        save_path=figure_dir / f"{config.prefix}_uncertainty_histogram.png",
        column="entropy",
    )
    plt.close("all")
    uncertainty_summary = None
    if config.num_classes > 2:
        uncertainty_summary = create_uncertainty_by_class_summary(uncertainty_df)
        uncertainty_summary.to_csv(
            table_dir / f"{config.prefix}_uncertainty_by_class.csv", index=False
        )
        plot_uncertainty_by_class(
            uncertainty_summary, figure_dir / f"{config.prefix}_uncertainty_by_class.png"
        )
    return mc_result, uncertainty_df, uncertainty_summary


def run_hybrid_experiment(config):
    print(f"\n===== Running {config.task} hybrid CNN+k-mer experiment =====")
    set_seed(config.seed)
    model_dir, figure_dir, table_dir = ensure_results_dirs(config)

    if config.task == "binary":
        original_train_df, test_df = load_binary_dataset(config.data_dir)
    else:
        original_train_df, test_df = load_sigma_dataset(config.data_dir)

    print_dataset_report(original_train_df, f"{config.task.title()} Original Train")
    print_dataset_report(test_df, f"{config.task.title()} Untouched Test")
    train_split, val_split, calibration_split = split_train_val_calibration(
        original_train_df
    )
    print(
        f"Split sizes: train={len(train_split)}, val={len(val_split)}, "
        f"calibration={len(calibration_split)}, test={len(test_df)}"
    )

    train_loader, val_loader, calibration_loader, test_loader = create_loaders(
        config, train_split, val_split, calibration_split, test_df
    )
    device = get_device()
    print(f"Using device: {device}")

    model = HybridCNNKmerClassifier(
        num_kmers=4 ** config.k,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )
    model, history_df = train_hybrid_model_select_by_f1(
        model,
        train_loader,
        val_loader,
        num_classes=config.num_classes,
        class_labels=train_split["label"],
        epochs=config.epochs,
        lr=config.learning_rate,
        patience=config.patience,
        device=device,
        model_save_path=model_dir / f"{config.prefix}_cnn_kmer.pt",
    )
    history_df.to_csv(table_dir / f"{config.prefix}_training_history.csv", index=False)
    plot_training_history(
        history_df, save_path=figure_dir / f"{config.prefix}_training_history.png"
    )
    plt.close("all")

    print(f"\nEvaluating {config.task} hybrid on untouched test set...")
    test_proba, test_labels = predict_hybrid_proba(model, test_loader, device)
    test_pred = np.argmax(test_proba, axis=1)
    metrics = evaluate_classification(
        test_labels, test_pred, test_proba, class_names=list(config.class_names)
    )
    save_metrics_table(metrics, table_dir / f"{config.prefix}_metrics.csv")
    report_df = classification_report_dataframe(
        test_labels, test_pred, list(config.class_names)
    )
    report_df.to_csv(table_dir / f"{config.prefix}_classification_report.csv")
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
    plt.close("all")

    run_calibration_and_conformal(
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
    run_mc_dropout(config, model, test_loader, figure_dir, table_dir, device)

    print(f"\nHybrid {config.task} metrics")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    return metrics


def read_metric_table(path):
    path = Path(path)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if {"metric", "value"}.issubset(df.columns):
        metrics = {}
        for _, row in df.iterrows():
            try:
                metrics[row["metric"]] = float(row["value"])
            except (TypeError, ValueError):
                continue
        return metrics
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def comparison_row(task, experiment, model, k, metrics):
    row = {"task": task, "experiment": experiment, "model": model, "k": k}
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column, np.nan)
    return row


def best_kmer_row(path):
    path = Path(path)
    if not path.is_file():
        return {}, np.nan, "unavailable"
    df = pd.read_csv(path)
    if df.empty:
        return {}, np.nan, "unavailable"
    best = df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    return best.to_dict(), best.get("k", np.nan), best.get("model", "kmer")


def create_final_comparison(table_dir):
    binary_cnn = read_metric_table(table_dir / "binary_metrics.csv")
    sigma_safe = read_metric_table(table_dir / "sigma_safe_metrics.csv")
    binary_hybrid = read_metric_table(table_dir / "hybrid_binary_metrics.csv")
    sigma_hybrid = read_metric_table(table_dir / "hybrid_sigma_metrics.csv")
    binary_kmer, binary_k, binary_kmer_model = best_kmer_row(
        table_dir / "kmer_binary_baseline_metrics.csv"
    )
    sigma_kmer, sigma_k, sigma_kmer_model = best_kmer_row(
        table_dir / "kmer_sigma_baseline_metrics.csv"
    )

    comparison_df = pd.DataFrame(
        [
            comparison_row("binary", "binary_cnn", "cnn", np.nan, binary_cnn),
            comparison_row(
                "binary", "binary_best_kmer", binary_kmer_model, binary_k, binary_kmer
            ),
            comparison_row(
                "binary",
                "binary_hybrid_cnn_kmer",
                "hybrid_cnn_kmer",
                BINARY_K,
                binary_hybrid,
            ),
            comparison_row("sigma", "sigma_safe_cnn", "cnn", np.nan, sigma_safe),
            comparison_row(
                "sigma", "sigma_best_kmer", sigma_kmer_model, sigma_k, sigma_kmer
            ),
            comparison_row(
                "sigma",
                "sigma_hybrid_cnn_kmer",
                "hybrid_cnn_kmer",
                SIGMA_K,
                sigma_hybrid,
            ),
        ]
    )
    comparison_df.to_csv(table_dir / "final_model_comparison_with_hybrid.csv", index=False)
    return comparison_df


def print_comparison_interpretation(comparison_df):
    print("\nBinary comparison: CNN vs best k-mer vs hybrid")
    print(comparison_df[comparison_df["task"] == "binary"].to_string(index=False))
    print("\nSigma comparison: safe CNN vs best k-mer vs hybrid")
    print(comparison_df[comparison_df["task"] == "sigma"].to_string(index=False))

    print("\nHonest interpretation")
    binary_rows = comparison_df[comparison_df["task"] == "binary"]
    sigma_rows = comparison_df[comparison_df["task"] == "sigma"]
    binary_hybrid_f1 = float(
        binary_rows.loc[
            binary_rows["experiment"] == "binary_hybrid_cnn_kmer", "f1_macro"
        ].iloc[0]
    )
    sigma_hybrid_f1 = float(
        sigma_rows.loc[
            sigma_rows["experiment"] == "sigma_hybrid_cnn_kmer", "f1_macro"
        ].iloc[0]
    )
    binary_non_hybrid_best = binary_rows[
        binary_rows["experiment"] != "binary_hybrid_cnn_kmer"
    ]["f1_macro"].max()
    sigma_non_hybrid_best = sigma_rows[
        sigma_rows["experiment"] != "sigma_hybrid_cnn_kmer"
    ]["f1_macro"].max()

    if binary_hybrid_f1 > binary_non_hybrid_best:
        print("- Binary: the hybrid improves over both the CNN and best k-mer baseline.")
    else:
        print("- Binary: the hybrid does not improve over the strongest existing baseline.")

    if sigma_hybrid_f1 > sigma_non_hybrid_best:
        print("- Sigma: the hybrid improves over both the safe CNN and best k-mer baseline.")
    else:
        print("- Sigma: the hybrid does not improve over the strongest existing baseline.")

    if max(sigma_hybrid_f1, sigma_non_hybrid_best) < 0.5:
        print("- Sigma classification remains weak/modest and should be reported cautiously.")


def main():
    binary_config = HybridExperimentConfig(
        task="binary",
        prefix="hybrid_binary",
        k=BINARY_K,
        num_classes=2,
        class_names=("Non-Promoter", "Promoter"),
        epochs=50,
    )
    sigma_config = HybridExperimentConfig(
        task="sigma",
        prefix="hybrid_sigma",
        k=SIGMA_K,
        num_classes=6,
        class_names=("Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"),
        epochs=100,
    )

    binary_metrics = run_hybrid_experiment(binary_config)
    sigma_metrics = run_hybrid_experiment(sigma_config)

    table_dir = PROJECT_ROOT / "results" / "tables"
    comparison_df = create_final_comparison(table_dir)
    print("\nHybrid binary metrics")
    print({key: value for key, value in binary_metrics.items() if not isinstance(value, dict)})
    print("\nHybrid sigma metrics")
    print({key: value for key, value in sigma_metrics.items() if not isinstance(value, dict)})
    print("\nFinal model comparison with hybrid")
    print(comparison_df.to_string(index=False))
    print_comparison_interpretation(comparison_df)
    print("\nHybrid CNN+k-mer experiment completed successfully.")


if __name__ == "__main__":
    main()
