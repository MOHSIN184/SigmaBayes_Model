from pathlib import Path
import itertools
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_binary_dataset, load_sigma_dataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    classification_report_dataframe,
    evaluate_classification,
    plot_confusion_matrix,
)


RANDOM_STATE = 42
K_VALUES = [3, 4, 5]
BINARY_CLASS_NAMES = ["Non-Promoter", "Promoter"]
SIGMA_CLASS_NAMES = ["Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"]
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


def generate_kmers(k):
    """Return all possible DNA k-mers in lexicographic A/C/G/T order."""
    return ["".join(chars) for chars in itertools.product("ACGT", repeat=k)]


def kmer_count_vector(sequence, kmers, normalize=True):
    """Count sliding-window k-mers for one sequence."""
    sequence = str(sequence).upper()
    k = len(kmers[0]) if kmers else 0
    counts = np.zeros(len(kmers), dtype=np.float32)
    if k <= 0 or len(sequence) < k:
        return counts

    kmer_to_index = {kmer: index for index, kmer in enumerate(kmers)}
    total_windows = len(sequence) - k + 1
    for start in range(total_windows):
        kmer = sequence[start : start + k]
        index = kmer_to_index.get(kmer)
        if index is not None:
            counts[index] += 1.0

    if normalize and total_windows > 0:
        counts /= float(total_windows)
    return counts


def build_kmer_matrix(sequences, k, normalize=True):
    """Build a dense k-mer feature matrix for a collection of sequences."""
    kmers = generate_kmers(k)
    kmer_to_index = {kmer: index for index, kmer in enumerate(kmers)}
    sequence_list = list(sequences)
    X = np.zeros((len(sequence_list), len(kmers)), dtype=np.float32)

    for row_index, sequence in enumerate(sequence_list):
        sequence = str(sequence).upper()
        if len(sequence) < k:
            continue
        total_windows = len(sequence) - k + 1
        for start in range(total_windows):
            kmer = sequence[start : start + k]
            column_index = kmer_to_index.get(kmer)
            if column_index is not None:
                X[row_index, column_index] += 1.0
        if normalize and total_windows > 0:
            X[row_index, :] /= float(total_windows)

    feature_names = [f"{k}mer_{kmer}" for kmer in kmers]
    return X, feature_names


def get_models(task_type):
    """Return CPU-friendly classical models for a binary or multiclass task."""
    if task_type not in {"binary", "sigma"}:
        raise ValueError("task_type must be 'binary' or 'sigma'.")

    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE,
                dual="auto",
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _sigmoid(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, -500, 500)
    return 1.0 / (1.0 + np.exp(-scores))


def _scores_to_probabilities(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)

    if not hasattr(model, "decision_function"):
        return None

    scores = model.decision_function(X_test)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        positive_probability = _sigmoid(scores)
        return np.column_stack([1.0 - positive_probability, positive_probability])
    return _softmax(scores)


def _metric_row(metrics, task_name, model_name, k):
    row = {
        "task": task_name,
        "model": model_name,
        "k": k,
    }
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column, np.nan)
    return row


def _prediction_dataframe(test_df, y_pred, y_proba, class_names, task_name, model_name, k):
    y_true = test_df["label"].to_numpy()
    prediction_df = pd.DataFrame(
        {
            "task": task_name,
            "model": model_name,
            "k": k,
            "id": test_df["id"].to_numpy(),
            "sequence": test_df["sequence"].to_numpy(),
            "true_label": y_true,
            "true_label_name": [class_names[label] for label in y_true],
            "predicted_label": y_pred,
            "predicted_label_name": [class_names[label] for label in y_pred],
            "correct": y_true == y_pred,
        }
    )

    if y_proba is not None:
        for class_index, class_name in enumerate(class_names):
            prediction_df[f"prob_{class_name}"] = y_proba[:, class_index]
        prediction_df["confidence"] = np.max(y_proba, axis=1)

    return prediction_df


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    task_name,
    model_name,
    k,
    test_df=None,
):
    """Train one model and return metrics, report, and prediction tables."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = _scores_to_probabilities(model, X_test)

    metrics = evaluate_classification(y_test, y_pred, y_proba, class_names)
    metrics.update({"task": task_name, "model": model_name, "k": k})

    report_df = classification_report_dataframe(y_test, y_pred, class_names)
    report_df.insert(0, "class_or_average", report_df.index)
    report_df.insert(0, "k", k)
    report_df.insert(0, "model", model_name)
    report_df.insert(0, "task", task_name)
    report_df = report_df.reset_index(drop=True)

    if test_df is None:
        test_df = pd.DataFrame(
            {
                "id": np.arange(len(y_test)),
                "sequence": [""] * len(y_test),
                "label": y_test,
            }
        )
    predictions_df = _prediction_dataframe(
        test_df, y_pred, y_proba, class_names, task_name, model_name, k
    )
    return metrics, report_df, predictions_df


def run_baselines_for_task(task_name, train_df, test_df, class_names, k_values):
    task_type = "binary" if len(class_names) == 2 else "sigma"
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    metric_rows = []
    reports = {}
    predictions = {}

    for k in k_values:
        print(f"\nBuilding {task_name} {k}-mer features...")
        X_train, _ = build_kmer_matrix(train_df["sequence"], k, normalize=True)
        X_test, _ = build_kmer_matrix(test_df["sequence"], k, normalize=True)

        for model_name, model in get_models(task_type).items():
            print(f"Training {task_name} {model_name} with {k}-mers...")
            metrics, report_df, predictions_df = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                class_names,
                task_name,
                model_name,
                k,
                test_df=test_df,
            )
            metric_rows.append(_metric_row(metrics, task_name, model_name, k))
            reports[(model_name, k)] = report_df
            predictions[(model_name, k)] = predictions_df

    metrics_df = pd.DataFrame(metric_rows)
    best_row = metrics_df.sort_values(
        ["f1_macro", "accuracy"], ascending=False
    ).iloc[0]
    best_key = (best_row["model"], int(best_row["k"]))
    return metrics_df, best_row, reports[best_key], predictions[best_key]


def _save_model_comparison_plot(metrics_df, title, save_path):
    plot_df = metrics_df.copy()
    plot_df["model_k"] = plot_df["model"] + " k=" + plot_df["k"].astype(str)
    plot_df = plot_df.sort_values("f1_macro", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["model_k"], plot_df["f1_macro"], color="#3B82A0")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, float(plot_df["f1_macro"].max()) + 0.05))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _read_cnn_metrics(path):
    path = Path(path)
    if not path.is_file():
        return {}

    df = pd.read_csv(path)
    if {"metric", "value"}.issubset(df.columns):
        values = {}
        for _, row in df.iterrows():
            try:
                values[row["metric"]] = float(row["value"])
            except (TypeError, ValueError):
                continue
        return values

    numeric_row = df.select_dtypes(include=[np.number]).head(1)
    if numeric_row.empty:
        return {}
    return numeric_row.iloc[0].to_dict()


def _comparison_row(task, experiment, model, k, metrics):
    row = {
        "task": task,
        "experiment": experiment,
        "model": model,
        "k": k,
    }
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column, np.nan)
    return row


def create_comparison_with_cnn(binary_best, sigma_best, table_dir):
    binary_cnn_metrics = _read_cnn_metrics(table_dir / "binary_metrics.csv")
    sigma_cnn_metrics = _read_cnn_metrics(table_dir / "sigma_safe_metrics.csv")

    rows = [
        _comparison_row("binary", "binary_cnn", "binary_cnn", np.nan, binary_cnn_metrics),
        _comparison_row(
            "binary",
            "best_kmer_binary_model",
            binary_best["model"],
            int(binary_best["k"]),
            binary_best.to_dict(),
        ),
        _comparison_row(
            "sigma", "sigma_safe_cnn", "sigma_safe_cnn", np.nan, sigma_cnn_metrics
        ),
        _comparison_row(
            "sigma",
            "best_kmer_sigma_model",
            sigma_best["model"],
            int(sigma_best["k"]),
            sigma_best.to_dict(),
        ),
    ]
    return pd.DataFrame(rows)


def _print_metrics_table(title, metrics_df):
    display_columns = ["model", "k", "accuracy", "f1_macro", "f1_weighted", "mcc"]
    available_columns = [column for column in display_columns if column in metrics_df.columns]
    print(f"\n{title}")
    print(metrics_df[available_columns].sort_values("f1_macro", ascending=False).to_string(index=False))


def _print_best(title, best_row):
    print(
        f"\n{title}: {best_row['model']} with k={int(best_row['k'])} "
        f"(macro F1={best_row['f1_macro']:.4f}, accuracy={best_row['accuracy']:.4f})"
    )


def _print_interpretation(comparison_df):
    binary_cnn = comparison_df.loc[comparison_df["experiment"] == "binary_cnn"].iloc[0]
    binary_kmer = comparison_df.loc[
        comparison_df["experiment"] == "best_kmer_binary_model"
    ].iloc[0]
    sigma_cnn = comparison_df.loc[comparison_df["experiment"] == "sigma_safe_cnn"].iloc[0]
    sigma_kmer = comparison_df.loc[
        comparison_df["experiment"] == "best_kmer_sigma_model"
    ].iloc[0]

    print("\nHonest interpretation")
    if pd.notna(binary_cnn["f1_macro"]):
        if binary_kmer["f1_macro"] > binary_cnn["f1_macro"]:
            print(
                "- Binary: classical k-mer features beat the CNN on macro F1; "
                "the CNN needs improvement for this endpoint."
            )
        else:
            print("- Binary: the CNN remains stronger than the best k-mer baseline on macro F1.")
    else:
        print("- Binary: CNN metrics were not available, so only k-mer results can be judged.")

    if pd.notna(sigma_cnn["f1_macro"]):
        if sigma_kmer["f1_macro"] > sigma_cnn["f1_macro"]:
            print(
                "- Sigma: classical k-mer features beat the safe CNN on macro F1; "
                "the CNN needs improvement for sigma-factor classification."
            )
        else:
            print("- Sigma: the safe CNN remains stronger than the best k-mer baseline on macro F1.")
    else:
        print("- Sigma: CNN metrics were not available, so only k-mer results can be judged.")

    best_sigma_f1 = sigma_kmer["f1_macro"]
    if pd.notna(sigma_cnn["f1_macro"]):
        best_sigma_f1 = max(best_sigma_f1, sigma_cnn["f1_macro"])
    if best_sigma_f1 < 0.5:
        print("- Sigma classification is still challenging; current macro F1 remains modest.")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    table_dir = results_dir / "tables"
    figure_dir = results_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    binary_train_df, binary_test_df = load_binary_dataset(data_dir)
    sigma_train_df, sigma_test_df = load_sigma_dataset(data_dir)

    binary_metrics_df, binary_best, binary_best_report, binary_best_predictions = (
        run_baselines_for_task(
            "binary",
            binary_train_df,
            binary_test_df,
            BINARY_CLASS_NAMES,
            K_VALUES,
        )
    )
    sigma_metrics_df, sigma_best, sigma_best_report, sigma_best_predictions = (
        run_baselines_for_task(
            "sigma",
            sigma_train_df,
            sigma_test_df,
            SIGMA_CLASS_NAMES,
            K_VALUES,
        )
    )

    binary_metrics_df.to_csv(table_dir / "kmer_binary_baseline_metrics.csv", index=False)
    sigma_metrics_df.to_csv(table_dir / "kmer_sigma_baseline_metrics.csv", index=False)
    binary_best_report.to_csv(
        table_dir / "kmer_binary_best_classification_report.csv", index=False
    )
    sigma_best_report.to_csv(
        table_dir / "kmer_sigma_best_classification_report.csv", index=False
    )
    binary_best_predictions.to_csv(table_dir / "kmer_binary_predictions.csv", index=False)
    sigma_best_predictions.to_csv(table_dir / "kmer_sigma_predictions.csv", index=False)

    _save_model_comparison_plot(
        binary_metrics_df,
        "Binary k-mer baseline comparison",
        figure_dir / "kmer_binary_model_comparison.png",
    )
    _save_model_comparison_plot(
        sigma_metrics_df,
        "Sigma k-mer baseline comparison",
        figure_dir / "kmer_sigma_model_comparison.png",
    )

    plot_confusion_matrix(
        binary_best_predictions["true_label"],
        binary_best_predictions["predicted_label"],
        BINARY_CLASS_NAMES,
        save_path=figure_dir / "kmer_binary_best_confusion_matrix.png",
    )
    plt.close("all")
    plot_confusion_matrix(
        sigma_best_predictions["true_label"],
        sigma_best_predictions["predicted_label"],
        SIGMA_CLASS_NAMES,
        save_path=figure_dir / "kmer_sigma_best_confusion_matrix.png",
    )
    plt.close("all")

    comparison_df = create_comparison_with_cnn(binary_best, sigma_best, table_dir)
    comparison_df.to_csv(
        table_dir / "kmer_baseline_comparison_with_cnn.csv", index=False
    )

    _print_metrics_table("Binary k-mer baseline results table", binary_metrics_df)
    _print_best("Best binary k-mer model", binary_best)
    _print_metrics_table("Sigma k-mer baseline results table", sigma_metrics_df)
    _print_best("Best sigma k-mer model", sigma_best)

    print("\nComparison with CNN")
    print(comparison_df.to_string(index=False))
    _print_interpretation(comparison_df)
    print("\nK-mer baseline experiment completed successfully.")


if __name__ == "__main__":
    main()
