"""Package only inference state that can be traced to saved BayesSigma outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_ROOT.parent
ARTIFACT_DIR = SERVER_ROOT / "artifacts"

TASKS = {
    "binary": {
        "labels": ["Non-Promoter", "Promoter"],
        "selected_model": "binary_cnn",
        "checkpoint": "binary_cnn.pt",
        "calibration": "binary_calibration_metrics.csv",
        "conformal": "binary_conformal_metrics.csv",
        "dropout": 0.3,
    },
    "sigma": {
        "labels": ["Sigma24", "Sigma28", "Sigma32", "Sigma38", "Sigma54", "Sigma70"],
        "selected_model": "safe_f1_selected_sigma_cnn",
        "checkpoint": "sigma_safe_cnn.pt",
        "calibration": "sigma_safe_calibration_metrics.csv",
        "conformal": "sigma_safe_conformal_metrics.csv",
        "dropout": 0.3,
        "mc_dropout_passes": 30,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_temperature(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[0]["temperature"]) if rows else None


def read_qhats(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    thresholds = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            confidence = str(round(float(row["confidence_level"]) * 100))
            thresholds[confidence] = float(row["qhat"])
    return thresholds if set(thresholds) >= {"90", "95"} else None


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": "BayesSigma",
        "input_length": 81,
        "alphabet": ["A", "C", "G", "T"],
        "available_outputs": ["raw_probabilities", "gc_content"],
        "tasks": {},
        "warnings": ["No fitted classical k-mer model artifact (.pkl/.joblib) was found."],
    }
    for name, config in TASKS.items():
        source = PROJECT_ROOT / "results" / "models" / config["checkpoint"]
        destination = ARTIFACT_DIR / config["checkpoint"]
        calibration = PROJECT_ROOT / "results" / "tables" / config["calibration"]
        conformal = PROJECT_ROOT / "results" / "tables" / config["conformal"]
        if source.is_file():
            shutil.copy2(source, destination)
        temperature = read_temperature(calibration)
        qhats = read_qhats(conformal)
        task = {
            "labels": config["labels"],
            "selected_model": config["selected_model"],
            "artifact_path": f"artifacts/{config['checkpoint']}",
            "sha256": sha256(destination) if destination.is_file() else "",
            "architecture": "CNNPromoterClassifier",
            "feature_type": "one_hot_acgt",
            "kmer_size": None,
            "dropout": config["dropout"],
            "temperature": temperature,
            "calibration_artifact": f"results/tables/{config['calibration']}" if temperature else None,
            "conformal_thresholds": qhats,
            "conformal_artifact": f"results/tables/{config['conformal']}" if qhats else None,
            "status": "ready" if destination.is_file() else "missing_checkpoint",
        }
        if "mc_dropout_passes" in config:
            task["mc_dropout_passes"] = config["mc_dropout_passes"]
        manifest["tasks"][name] = task
        if temperature is None:
            manifest["warnings"].append(f"{name}: saved calibration temperature is missing.")
        if qhats is None:
            manifest["warnings"].append(f"{name}: saved 90/95 conformal qhat values are missing.")
    rf_path = ARTIFACT_DIR / "binary_calibrated_5mer_random_forest.joblib"
    rf_config_path = ARTIFACT_DIR / "binary_5mer_config.json"
    if rf_path.is_file() and rf_config_path.is_file():
        rf_config = json.loads(rf_config_path.read_text(encoding="utf-8"))
        cnn_fallback = manifest["tasks"]["binary"]
        manifest["tasks"]["binary"] = {
            "labels": rf_config["labels"],
            "selected_model": rf_config["model_name"],
            "artifact_path": "artifacts/binary_calibrated_5mer_random_forest.joblib",
            "preprocessing_config": "artifacts/binary_5mer_config.json",
            "preprocessing_config_sha256": sha256(rf_config_path),
            "sha256": sha256(rf_path),
            "architecture": "RandomForestClassifier + isotonic CalibratedClassifierCV",
            "feature_type": rf_config["feature_type"],
            "kmer_size": rf_config["kmer_size"],
            "random_state": rf_config["random_state"],
            "calibration_artifact": "embedded in joblib bundle",
            "conformal_thresholds": rf_config["conformal_thresholds"],
            "conformal_artifact": "results/tables/kmer_reliability_binary_conformal_metrics.csv",
            "status": "ready",
            "fallback_model": cnn_fallback,
        }
        manifest["warnings"] = [
            warning for warning in manifest["warnings"]
            if "classical k-mer model" not in warning
        ]
    if all(
        task.get("temperature") or task.get("calibration_artifact")
        for task in manifest["tasks"].values()
    ):
        manifest["available_outputs"].append("temperature_scaled_probabilities")
    if all(task.get("conformal_thresholds") for task in manifest["tasks"].values()):
        manifest["available_outputs"].extend(["conformal_sets_90", "conformal_sets_95"])
    manifest["available_outputs"].append("sigma_mc_dropout_uncertainty")
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if all(t["status"] == "ready" for t in manifest["tasks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
