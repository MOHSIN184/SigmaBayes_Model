"""Reproduce and export the manuscript's calibrated binary 5-mer Random Forest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

SERVER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_kmer_reliability import build_kmer_matrix
from src.conformal import compute_nonconformity_scores, conformal_quantile
from src.data_loader import load_binary_dataset, split_train_val_calibration

RANDOM_STATE = 42
KMER_SIZE = 5
LABELS = ["Non-Promoter", "Promoter"]
ARTIFACT_PATH = SERVER_ROOT / "artifacts" / "binary_calibrated_5mer_random_forest.joblib"
CONFIG_PATH = SERVER_ROOT / "artifacts" / "binary_5mer_config.json"
RECORDED_METRICS_PATH = PROJECT_ROOT / "results" / "tables" / "kmer_reliability_binary_metrics.csv"
RECORDED_CONFORMAL_PATH = PROJECT_ROOT / "results" / "tables" / "kmer_reliability_binary_conformal_metrics.csv"
TOLERANCE = 1e-10

def fit_calibrator(base_model, features, labels):
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method="isotonic").fit(features, labels)
    return CalibratedClassifierCV(estimator=base_model, method="isotonic", cv="prefit").fit(features, labels)

def recorded_metrics() -> dict[str, float]:
    frame = pd.read_csv(RECORDED_METRICS_PATH)
    return {str(row.metric): float(row.value) for row in frame.itertuples() if row.metric in {"accuracy", "f1_macro", "auroc"}}

def recorded_qhats() -> dict[str, float]:
    frame = pd.read_csv(RECORDED_CONFORMAL_PATH)
    return {str(round(float(row.confidence_level) * 100)): float(row.qhat) for row in frame.itertuples()}

def assert_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOLERANCE):
        raise RuntimeError(f"Reproduction check failed for {name}: {actual} != {expected}")

def main() -> None:
    train_df, test_df = load_binary_dataset(PROJECT_ROOT / "data")
    train_split, validation_split, calibration_split = split_train_val_calibration(train_df, val_size=0.15, cal_size=0.15, random_state=RANDOM_STATE)
    matrices = {
        "train": build_kmer_matrix(train_split["sequence"], KMER_SIZE, normalize=True),
        "validation": build_kmer_matrix(validation_split["sequence"], KMER_SIZE, normalize=True),
        "calibration": build_kmer_matrix(calibration_split["sequence"], KMER_SIZE, normalize=True),
        "test": build_kmer_matrix(test_df["sequence"], KMER_SIZE, normalize=True),
    }
    base_model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    base_model.fit(matrices["train"], train_split["label"].to_numpy())
    calibrated_model = fit_calibrator(base_model, matrices["validation"], validation_split["label"].to_numpy())
    test_probabilities = calibrated_model.predict_proba(matrices["test"])
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_labels = test_df["label"].to_numpy()
    actual_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "f1_macro": float(f1_score(test_labels, test_predictions, average="macro")),
        "auroc": float(roc_auc_score(test_labels, test_probabilities[:, 1])),
    }
    for name, expected in recorded_metrics().items():
        assert_close(name, actual_metrics[name], expected)
    calibration_probabilities = calibrated_model.predict_proba(matrices["calibration"])
    scores = compute_nonconformity_scores(calibration_split["label"].to_numpy(), calibration_probabilities)
    qhats = {"90": conformal_quantile(scores, alpha=0.1), "95": conformal_quantile(scores, alpha=0.05)}
    for level, expected in recorded_qhats().items():
        assert_close(f"qhat_{level}", qhats[level], expected)
    bundle = {
        "artifact_version": 1, "model_name": "calibrated_5mer_random_forest",
        "base_model": base_model, "calibrated_model": calibrated_model, "labels": LABELS,
        "kmer_size": KMER_SIZE, "normalize": True, "feature_order": "lexicographic_ACGT", "random_state": RANDOM_STATE,
    }
    config = {
        "model_name": bundle["model_name"], "input_length": 81, "labels": LABELS,
        "feature_type": "normalized_kmer_counts", "kmer_size": KMER_SIZE, "feature_count": 4 ** KMER_SIZE,
        "feature_order": bundle["feature_order"], "normalize": True, "random_state": RANDOM_STATE,
        "split": {"train": 0.70, "validation": 0.15, "calibration": 0.15, "stratified": True},
        "estimator": {"type": "RandomForestClassifier", "n_estimators": 500, "class_weight": "balanced"},
        "calibration": {"method": "isotonic", "split": "validation"},
        "conformal_thresholds": qhats, "verified_test_metrics": actual_metrics,
        "source_script": "scripts/run_kmer_reliability.py",
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(ARTIFACT_PATH), "config": str(CONFIG_PATH), "metrics": actual_metrics, "qhat": qhats}, indent=2))

if __name__ == "__main__":
    main()
