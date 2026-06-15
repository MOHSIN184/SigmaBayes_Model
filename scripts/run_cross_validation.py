from pathlib import Path
import itertools
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_binary_dataset, load_sigma_dataset  # noqa: E402
from src.encoding import create_torch_dataset  # noqa: E402
from src.evaluate import evaluate_classification  # noqa: E402
from src.models import CNNPromoterClassifier  # noqa: E402


RUN_CNN_CV = True
CNN_EPOCHS_BINARY = 25
CNN_EPOCHS_SIGMA = 40

RANDOM_STATE = 42
N_SPLITS = 5
MAX_LEN = 81
BATCH_SIZE = 64
BINARY_K = 5
SIGMA_K = 3
BINARY_CLASS_NAMES = ["Non-Promoter", "Promoter"]
SIGMA_CLASS_NAMES = ["Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"]
SUMMARY_METRICS = [
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
WIDE_METRICS = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "auroc",
    "auprc",
    "auroc_macro_ovr",
    "auprc_macro_ovr",
]


def set_seed(seed):
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


def kmer_count_vector(sequence, kmers, normalize=True):
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

    denominator = valid_windows if valid_windows > 0 else total_windows
    if normalize and denominator > 0:
        counts /= float(denominator)
    return counts


def build_kmer_matrix(sequences, k, normalize=True):
    kmers = generate_kmers(k)
    return np.stack(
        [kmer_count_vector(sequence, kmers, normalize=normalize) for sequence in sequences],
        axis=0,
    ).astype(np.float32)


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def decision_function_to_proba(model, X):
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:
        scores = np.clip(scores, -500, 500)
        positive = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - positive, positive])
    return softmax(scores)


def metric_row(task, model_name, fold, y_true, y_pred, y_proba, class_names):
    metrics = evaluate_classification(y_true, y_pred, y_proba, class_names=class_names)
    row = {"task": task, "model": model_name, "fold": fold}
    for metric in SUMMARY_METRICS:
        row[metric] = metrics.get(metric, np.nan)
    if task == "sigma":
        for class_name in class_names:
            row[f"f1_{class_name}"] = metrics["per_class_f1"].get(class_name, np.nan)
    return row


def create_loader(df, shuffle):
    return DataLoader(
        create_torch_dataset(df.reset_index(drop=True), max_len=MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def compute_torch_class_weights(labels, num_classes):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.asarray(labels, dtype=int),
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
    return total_loss / total_samples if total_samples else 0.0


def evaluate_loader_for_f1(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            predictions.extend(pred.detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    return {
        "loss": total_loss / total_samples if total_samples else 0.0,
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0)
        if labels
        else 0.0,
    }


def train_cnn_select_by_f1(
    train_inner_df,
    val_inner_df,
    num_classes,
    epochs,
    patience,
    lr,
    dropout,
    seed,
):
    set_seed(seed)
    device = get_device()
    train_loader = create_loader(train_inner_df, shuffle=True)
    val_loader = create_loader(val_inner_df, shuffle=False)
    model = CNNPromoterClassifier(num_classes=num_classes, dropout=dropout).to(device)
    class_weights = compute_torch_class_weights(train_inner_df["label"], num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -np.inf
    best_state_dict = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_loader_for_f1(model, val_loader, criterion, device)
        print(
            f"    epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"    early stopping at epoch {epoch}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)
    return model, device


def predict_cnn(model, df, device):
    loader = create_loader(df, shuffle=False)
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            proba = torch.softmax(logits, dim=1)
            probabilities.append(proba.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    y_proba = np.concatenate(probabilities, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = np.argmax(y_proba, axis=1)
    return y_true, y_pred, y_proba


def run_binary_rf_cv(full_df):
    print("\nRunning combined-data CV: binary 5-mer Random Forest")
    X = build_kmer_matrix(full_df["sequence"], BINARY_K, normalize=True)
    y = full_df["label"].to_numpy()
    rows = []
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        fold_seed = RANDOM_STATE + fold
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=fold_seed,
            n_jobs=-1,
        )
        model.fit(X[train_index], y[train_index])
        y_proba = model.predict_proba(X[test_index])
        y_pred = np.argmax(y_proba, axis=1)
        row = metric_row(
            "binary",
            "5mer_random_forest",
            fold,
            y[test_index],
            y_pred,
            y_proba,
            BINARY_CLASS_NAMES,
        )
        rows.append(row)
        print(
            f"  fold {fold}: f1_macro={row['f1_macro']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, mcc={row['mcc']:.4f}"
        )
    return rows


def run_sigma_svm_cv(full_df):
    print("\nRunning combined-data CV: sigma 3-mer Linear SVM")
    X = build_kmer_matrix(full_df["sequence"], SIGMA_K, normalize=True)
    y = full_df["label"].to_numpy()
    rows = []
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        fold_seed = RANDOM_STATE + fold
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=fold_seed,
                dual="auto",
            ),
        )
        model.fit(X[train_index], y[train_index])
        y_pred = model.predict(X[test_index])
        y_proba = decision_function_to_proba(model, X[test_index])
        row = metric_row(
            "sigma",
            "3mer_linear_svm",
            fold,
            y[test_index],
            y_pred,
            y_proba,
            SIGMA_CLASS_NAMES,
        )
        rows.append(row)
        print(
            f"  fold {fold}: f1_macro={row['f1_macro']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, mcc={row['mcc']:.4f}"
        )
    return rows


def run_cnn_cv(full_df, task, model_name, class_names, epochs, patience, lr):
    print(f"\nRunning combined-data CV: {task} CNN")
    y = full_df["label"].to_numpy()
    rows = []
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (train_index, test_index) in enumerate(splitter.split(full_df, y), start=1):
        fold_seed = RANDOM_STATE + fold
        fold_train_df = full_df.iloc[train_index].reset_index(drop=True)
        fold_test_df = full_df.iloc[test_index].reset_index(drop=True)
        train_inner_df, val_inner_df = train_test_split(
            fold_train_df,
            test_size=0.15,
            random_state=fold_seed,
            stratify=fold_train_df["label"],
        )
        print(
            f"  fold {fold}: train_inner={len(train_inner_df)}, "
            f"val_inner={len(val_inner_df)}, fold_test={len(fold_test_df)}"
        )
        model, device = train_cnn_select_by_f1(
            train_inner_df,
            val_inner_df,
            num_classes=len(class_names),
            epochs=epochs,
            patience=patience,
            lr=lr,
            dropout=0.3,
            seed=fold_seed,
        )
        y_true, y_pred, y_proba = predict_cnn(model, fold_test_df, device)
        row = metric_row(task, model_name, fold, y_true, y_pred, y_proba, class_names)
        rows.append(row)
        print(
            f"  fold {fold} result: f1_macro={row['f1_macro']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, mcc={row['mcc']:.4f}"
        )
    return rows


def create_summary_tables(fold_df):
    metric_columns = [
        column
        for column in fold_df.columns
        if column not in {"task", "model", "fold"}
        and pd.api.types.is_numeric_dtype(fold_df[column])
    ]
    summary_rows = []
    for (task, model), group in fold_df.groupby(["task", "model"]):
        for metric in metric_columns:
            values = group[metric].dropna()
            if values.empty:
                continue
            summary_rows.append(
                {
                    "task": task,
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    wide_rows = []
    for (task, model), group in fold_df.groupby(["task", "model"]):
        row = {"task": task, "model": model}
        for metric in WIDE_METRICS:
            if metric in group.columns:
                values = group[metric].dropna()
            else:
                values = pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    return summary_df, wide_df


def plot_cv_comparison(wide_df, task, save_path):
    plot_df = wide_df[wide_df["task"] == task].copy()
    plot_df = plot_df.sort_values("model")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        plot_df["model"],
        plot_df["f1_macro_mean"],
        yerr=plot_df["f1_macro_std"],
        capsize=6,
        color=["#3B82A0", "#7C9A42"][: len(plot_df)],
    )
    ax.set_ylabel("Macro F1 mean +/- std")
    ax.set_xlabel("Model")
    ax.set_title(f"{task.title()} 5-fold CV macro F1")
    ax.set_ylim(0, max(1.0, float(plot_df["f1_macro_mean"].max()) + 0.1))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def get_mean(wide_df, task, model, metric):
    rows = wide_df[(wide_df["task"] == task) & (wide_df["model"] == model)]
    if rows.empty:
        return np.nan
    return rows.iloc[0].get(f"{metric}_mean", np.nan)


def get_std(wide_df, task, model, metric):
    rows = wide_df[(wide_df["task"] == task) & (wide_df["model"] == model)]
    if rows.empty:
        return np.nan
    return rows.iloc[0].get(f"{metric}_std", np.nan)


def print_interpretation(wide_df):
    binary_rf = get_mean(wide_df, "binary", "5mer_random_forest", "f1_macro")
    binary_cnn = get_mean(wide_df, "binary", "binary_cnn", "f1_macro")
    sigma_svm = get_mean(wide_df, "sigma", "3mer_linear_svm", "f1_macro")
    sigma_cnn = get_mean(wide_df, "sigma", "sigma_cnn", "f1_macro")

    print("\nInterpretation")
    if pd.notna(binary_rf) and pd.notna(binary_cnn):
        if binary_rf > binary_cnn:
            print("- Binary: 5-mer Random Forest is more stable/stronger by mean macro F1.")
        else:
            print("- Binary: CNN is stronger by mean macro F1.")
    if pd.notna(sigma_svm) and pd.notna(sigma_cnn):
        if sigma_cnn > sigma_svm:
            print("- Sigma: CNN is better for balanced sigma classification by mean macro F1.")
        else:
            print("- Sigma: 3-mer Linear SVM is better by mean macro F1.")
        if max(sigma_svm, sigma_cnn) < 0.50:
            print("- Sigma macro F1 remains below 0.50, so sigma classification remains weak for performance-focused publication.")

    high_std_rows = []
    for _, row in wide_df.iterrows():
        std_value = row.get("f1_macro_std", np.nan)
        if pd.notna(std_value) and std_value >= 0.05:
            high_std_rows.append(f"{row['task']} {row['model']} std={std_value:.4f}")
    if high_std_rows:
        print("- Some CV results are unstable by macro F1 standard deviation: " + "; ".join(high_std_rows))
    else:
        print("- Macro F1 standard deviations are not high under the >=0.05 heuristic.")


def main():
    table_dir = PROJECT_ROOT / "results" / "tables"
    figure_dir = PROJECT_ROOT / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("This is a combined-data cross-validation stability experiment.")
    print("Original train and test files are combined only for CV, not for held-out evaluation.")

    binary_train, binary_test = load_binary_dataset(PROJECT_ROOT / "data")
    sigma_train, sigma_test = load_sigma_dataset(PROJECT_ROOT / "data")
    binary_full = pd.concat([binary_train, binary_test], ignore_index=True)
    sigma_full = pd.concat([sigma_train, sigma_test], ignore_index=True)

    all_rows = []
    all_rows.extend(run_binary_rf_cv(binary_full))
    all_rows.extend(run_sigma_svm_cv(sigma_full))

    if RUN_CNN_CV:
        all_rows.extend(
            run_cnn_cv(
                binary_full,
                task="binary",
                model_name="binary_cnn",
                class_names=BINARY_CLASS_NAMES,
                epochs=CNN_EPOCHS_BINARY,
                patience=6,
                lr=1e-3,
            )
        )
        all_rows.extend(
            run_cnn_cv(
                sigma_full,
                task="sigma",
                model_name="sigma_cnn",
                class_names=SIGMA_CLASS_NAMES,
                epochs=CNN_EPOCHS_SIGMA,
                patience=8,
                lr=3e-4,
            )
        )

    fold_df = pd.DataFrame(all_rows)
    summary_df, wide_df = create_summary_tables(fold_df)

    fold_df.to_csv(table_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(table_dir / "cv_summary_metrics.csv", index=False)
    wide_df.to_csv(table_dir / "cv_summary_wide.csv", index=False)

    plot_cv_comparison(
        wide_df, "binary", figure_dir / "cv_binary_model_comparison.png"
    )
    plot_cv_comparison(wide_df, "sigma", figure_dir / "cv_sigma_model_comparison.png")

    print("\nFold-by-fold metrics")
    print(fold_df.to_string(index=False))
    print("\nSummary wide table")
    print(wide_df.to_string(index=False))
    print_interpretation(wide_df)
    print("\nCross-validation experiment completed successfully.")


if __name__ == "__main__":
    main()
