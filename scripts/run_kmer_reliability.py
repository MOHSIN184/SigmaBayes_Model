from pathlib import Path
import itertools
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - for older scikit-learn versions
    FrozenEstimator = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import evaluate_calibration, plot_reliability_diagram  # noqa: E402
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
from src.evaluate import (  # noqa: E402
    classification_report_dataframe,
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    save_metrics_table,
)


RANDOM_STATE = 42
BINARY_K = 5
SIGMA_K = 3
BINARY_CLASS_NAMES = ["Non-Promoter", "Promoter"]
SIGMA_CLASS_NAMES = ["Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"]
METRIC_COLUMNS = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "auroc",
    "auprc",
    "auroc_macro_ovr",
    "auprc_macro_ovr",
]


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


def sigmoid(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, -500, 500)
    return 1.0 / (1.0 + np.exp(-scores))


def decision_function_to_proba(model, X):
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:
        positive_probability = sigmoid(scores)
        return np.column_stack([1.0 - positive_probability, positive_probability])
    return softmax(scores)


def fit_prefit_calibrator(base_model, X_val, y_val, method):
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
        )
        return calibrator.fit(X_val, y_val)

    calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv="prefit",
    )
    return calibrator.fit(X_val, y_val)


def save_classification_outputs(
    task,
    class_names,
    y_test,
    y_proba,
    figure_dir,
    table_dir,
):
    prefix = f"kmer_reliability_{task}"
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate_classification(y_test, y_pred, y_proba, class_names=class_names)
    save_metrics_table(metrics, table_dir / f"{prefix}_metrics.csv")
    report_df = classification_report_dataframe(y_test, y_pred, class_names)
    report_df.to_csv(table_dir / f"{prefix}_classification_report.csv")

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        save_path=figure_dir / f"{prefix}_confusion_matrix.png",
    )
    plot_roc_pr_curves(
        y_test,
        y_proba,
        class_names,
        save_dir=figure_dir,
        prefix=prefix,
    )
    plt.close("all")
    return metrics, y_pred


def save_calibration_outputs(
    task,
    class_names,
    y_test,
    proba_before,
    proba_after,
    figure_dir,
    table_dir,
):
    prefix = f"kmer_reliability_{task}"
    before = evaluate_calibration(
        y_test, proba_before, num_classes=len(class_names), n_bins=15
    )
    after = evaluate_calibration(
        y_test, proba_after, num_classes=len(class_names), n_bins=15
    )
    calibration_df = pd.DataFrame(
        [
            {"stage": "before", **before},
            {"stage": "after", **after},
        ]
    )
    calibration_df.to_csv(table_dir / f"{prefix}_calibration_metrics.csv", index=False)

    plot_reliability_diagram(
        y_test,
        proba_before,
        save_path=figure_dir / f"{prefix}_reliability_before.png",
        n_bins=15,
        title=f"{task.title()} k-mer Reliability Before Calibration",
    )
    plot_reliability_diagram(
        y_test,
        proba_after,
        save_path=figure_dir / f"{prefix}_reliability_after.png",
        n_bins=15,
        title=f"{task.title()} k-mer Reliability After Calibration",
    )
    plt.close("all")
    return calibration_df


def create_conformal_classwise_summary(class_names, test_labels, result_90, result_95):
    rows = []
    metrics_90 = result_90["metrics"]
    metrics_95 = result_95["metrics"]
    for class_index, class_name in enumerate(class_names):
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


def run_conformal_outputs(
    task,
    class_names,
    y_cal,
    proba_cal,
    y_test,
    proba_test,
    table_dir,
    save_classwise=False,
):
    prefix = f"kmer_reliability_{task}"
    results_by_alpha = []
    outputs = {}
    for alpha, suffix in [(0.1, "90"), (0.05, "95")]:
        result = run_conformal_prediction(
            y_cal,
            proba_cal,
            y_test,
            proba_test,
            class_names=class_names,
            alpha=alpha,
        )
        save_conformal_results(
            result,
            y_test,
            proba_test,
            table_dir / f"{prefix}_conformal_predictions_{suffix}.csv",
            class_names=class_names,
        )
        results_by_alpha.append(result)
        outputs[alpha] = result

    conformal_df = save_conformal_metrics(
        results_by_alpha, table_dir / f"{prefix}_conformal_metrics.csv"
    )
    if save_classwise:
        classwise_df = create_conformal_classwise_summary(
            class_names, y_test, outputs[0.1], outputs[0.05]
        )
        classwise_df.to_csv(
            table_dir / f"{prefix}_conformal_classwise_summary.csv", index=False
        )
    return conformal_df


def print_metric_dict(title, metrics):
    print(f"\n{title}")
    for key, value in metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")


def print_calibration(title, calibration_df):
    print(f"\n{title}")
    print(
        calibration_df[
            ["stage", "ece", "brier_score", "nll", "mean_confidence", "accuracy"]
        ].to_string(index=False)
    )


def print_conformal(title, conformal_df):
    print(f"\n{title}")
    print(
        conformal_df[
            ["alpha", "confidence_level", "coverage", "average_set_size", "qhat"]
        ].to_string(index=False)
    )


def run_binary_reliability(table_dir, figure_dir):
    print("\n===== Binary k-mer reliability workflow =====")
    train_df, test_df = load_binary_dataset(PROJECT_ROOT / "data")
    print_dataset_report(train_df, "Binary Original Train")
    print_dataset_report(test_df, "Binary Untouched Test")
    train_split, val_split, calibration_split = split_train_val_calibration(train_df)

    X_train = build_kmer_matrix(train_split["sequence"], BINARY_K, normalize=True)
    X_val = build_kmer_matrix(val_split["sequence"], BINARY_K, normalize=True)
    X_cal = build_kmer_matrix(calibration_split["sequence"], BINARY_K, normalize=True)
    X_test = build_kmer_matrix(test_df["sequence"], BINARY_K, normalize=True)
    y_train = train_split["label"].to_numpy()
    y_val = val_split["label"].to_numpy()
    y_cal = calibration_split["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    base_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training binary 5-mer Random Forest on train split only...")
    base_model.fit(X_train, y_train)
    proba_before = base_model.predict_proba(X_test)

    print("Calibrating binary Random Forest on validation split...")
    calibrated_model = fit_prefit_calibrator(base_model, X_val, y_val, method="isotonic")
    proba_after = calibrated_model.predict_proba(X_test)
    proba_cal = calibrated_model.predict_proba(X_cal)

    metrics, _ = save_classification_outputs(
        "binary",
        BINARY_CLASS_NAMES,
        y_test,
        proba_after,
        figure_dir,
        table_dir,
    )
    calibration_df = save_calibration_outputs(
        "binary",
        BINARY_CLASS_NAMES,
        y_test,
        proba_before,
        proba_after,
        figure_dir,
        table_dir,
    )
    conformal_df = run_conformal_outputs(
        "binary",
        BINARY_CLASS_NAMES,
        y_cal,
        proba_cal,
        y_test,
        proba_after,
        table_dir,
    )
    return metrics, calibration_df, conformal_df


def run_sigma_reliability(table_dir, figure_dir):
    print("\n===== Sigma k-mer reliability workflow =====")
    train_df, test_df = load_sigma_dataset(PROJECT_ROOT / "data")
    print_dataset_report(train_df, "Sigma Original Train")
    print_dataset_report(test_df, "Sigma Untouched Test")
    train_split, val_split, calibration_split = split_train_val_calibration(train_df)

    X_train = build_kmer_matrix(train_split["sequence"], SIGMA_K, normalize=True)
    X_val = build_kmer_matrix(val_split["sequence"], SIGMA_K, normalize=True)
    X_cal = build_kmer_matrix(calibration_split["sequence"], SIGMA_K, normalize=True)
    X_test = build_kmer_matrix(test_df["sequence"], SIGMA_K, normalize=True)
    y_train = train_split["label"].to_numpy()
    y_val = val_split["label"].to_numpy()
    y_cal = calibration_split["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    base_model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=10000,
            dual="auto",
        ),
    )
    print("Training sigma 3-mer Linear SVM on train split only...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        base_model.fit(X_train, y_train)
    proba_before = decision_function_to_proba(base_model, X_test)

    print("Calibrating sigma Linear SVM on validation split...")
    calibrated_model = fit_prefit_calibrator(base_model, X_val, y_val, method="sigmoid")
    proba_after = calibrated_model.predict_proba(X_test)
    proba_cal = calibrated_model.predict_proba(X_cal)

    metrics, _ = save_classification_outputs(
        "sigma",
        SIGMA_CLASS_NAMES,
        y_test,
        proba_after,
        figure_dir,
        table_dir,
    )
    calibration_df = save_calibration_outputs(
        "sigma",
        SIGMA_CLASS_NAMES,
        y_test,
        proba_before,
        proba_after,
        figure_dir,
        table_dir,
    )
    conformal_df = run_conformal_outputs(
        "sigma",
        SIGMA_CLASS_NAMES,
        y_cal,
        proba_cal,
        y_test,
        proba_after,
        table_dir,
        save_classwise=True,
    )
    return metrics, calibration_df, conformal_df


def read_metrics(path):
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
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def read_calibration(path):
    path = Path(path)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    output = {}
    for stage in ["before", "after"]:
        stage_df = df[df["stage"] == stage] if "stage" in df.columns else pd.DataFrame()
        if stage_df.empty:
            continue
        row = stage_df.iloc[0]
        output[f"ece_{stage}"] = row.get("ece", np.nan)
        output[f"brier_{stage}"] = row.get("brier_score", np.nan)
    return output


def read_conformal(path):
    path = Path(path)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    output = {}
    for alpha, suffix in [(0.1, "90"), (0.05, "95")]:
        if "alpha" not in df.columns:
            continue
        rows = df[np.isclose(df["alpha"].astype(float), alpha)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        output[f"conformal_coverage_{suffix}"] = row.get("coverage", np.nan)
        output[f"conformal_avg_set_size_{suffix}"] = row.get(
            "average_set_size", np.nan
        )
    return output


def comparison_row(task, experiment, model, metrics, calibration, conformal):
    row = {
        "task": task,
        "experiment": experiment,
        "model": model,
    }
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column, np.nan)
    for column in [
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
        "conformal_coverage_90",
        "conformal_avg_set_size_90",
        "conformal_coverage_95",
        "conformal_avg_set_size_95",
    ]:
        row[column] = calibration.get(column, conformal.get(column, np.nan))
    return row


def create_final_reliability_comparison(table_dir):
    rows = [
        comparison_row(
            "binary",
            "binary_cnn",
            "cnn",
            read_metrics(table_dir / "binary_metrics.csv"),
            read_calibration(table_dir / "binary_calibration_metrics.csv"),
            read_conformal(table_dir / "binary_conformal_metrics.csv"),
        ),
        comparison_row(
            "binary",
            "binary_best_kmer_reliable",
            "5mer_random_forest_calibrated",
            read_metrics(table_dir / "kmer_reliability_binary_metrics.csv"),
            read_calibration(table_dir / "kmer_reliability_binary_calibration_metrics.csv"),
            read_conformal(table_dir / "kmer_reliability_binary_conformal_metrics.csv"),
        ),
        comparison_row(
            "sigma",
            "sigma_safe_cnn",
            "cnn",
            read_metrics(table_dir / "sigma_safe_metrics.csv"),
            read_calibration(table_dir / "sigma_safe_calibration_metrics.csv"),
            read_conformal(table_dir / "sigma_safe_conformal_metrics.csv"),
        ),
        comparison_row(
            "sigma",
            "sigma_best_kmer_reliable",
            "3mer_linear_svm_calibrated",
            read_metrics(table_dir / "kmer_reliability_sigma_metrics.csv"),
            read_calibration(table_dir / "kmer_reliability_sigma_calibration_metrics.csv"),
            read_conformal(table_dir / "kmer_reliability_sigma_conformal_metrics.csv"),
        ),
    ]
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(table_dir / "final_reliability_comparison.csv", index=False)
    return comparison_df


def print_honest_interpretation(comparison_df):
    binary = comparison_df[comparison_df["task"] == "binary"]
    sigma = comparison_df[comparison_df["task"] == "sigma"]
    binary_kmer = binary[binary["experiment"] == "binary_best_kmer_reliable"].iloc[0]
    binary_cnn = binary[binary["experiment"] == "binary_cnn"].iloc[0]
    sigma_kmer = sigma[sigma["experiment"] == "sigma_best_kmer_reliable"].iloc[0]
    sigma_cnn = sigma[sigma["experiment"] == "sigma_safe_cnn"].iloc[0]

    print("\nHonest interpretation")
    if binary_kmer["f1_macro"] > binary_cnn["f1_macro"]:
        print("- Binary: the final binary endpoint should use the calibrated 5-mer Random Forest.")
    else:
        print("- Binary: the CNN remains stronger after the reliability split workflow.")

    if sigma_cnn["f1_macro"] >= sigma_kmer["f1_macro"]:
        print("- Sigma: the final sigma endpoint should use the safe CNN by macro F1.")
    else:
        print("- Sigma: the calibrated 3-mer Linear SVM is stronger by macro F1 in this split workflow.")

    if max(sigma_cnn["f1_macro"], sigma_kmer["f1_macro"]) < 0.5:
        print(
            "- Sigma classification is still not strong enough for a performance-focused journal paper."
        )

    print(
        "- The project is better framed as a trustworthy benchmarking and reliability framework "
        "for promoter/sigma model selection, rather than only a deep-learning classifier."
    )


def main():
    table_dir = PROJECT_ROOT / "results" / "tables"
    figure_dir = PROJECT_ROOT / "results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    binary_metrics, binary_calibration_df, binary_conformal_df = run_binary_reliability(
        table_dir, figure_dir
    )
    sigma_metrics, sigma_calibration_df, sigma_conformal_df = run_sigma_reliability(
        table_dir, figure_dir
    )

    print_metric_dict("Binary k-mer reliability metrics", binary_metrics)
    print_calibration("Binary calibration before/after", binary_calibration_df)
    print_conformal("Binary conformal coverage and set size", binary_conformal_df)
    print_metric_dict("Sigma k-mer reliability metrics", sigma_metrics)
    print_calibration("Sigma calibration before/after", sigma_calibration_df)
    print_conformal("Sigma conformal coverage and set size", sigma_conformal_df)

    comparison_df = create_final_reliability_comparison(table_dir)
    print("\nFinal reliability comparison table")
    print(comparison_df.to_string(index=False))
    print_honest_interpretation(comparison_df)
    print("\nK-mer reliability experiment completed successfully.")


if __name__ == "__main__":
    main()
