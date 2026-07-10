from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import torch
from torch import nn

from .utils import ARTIFACT_DIR, load_manifest


class CNNPromoterClassifier(nn.Module):
    """Exact architecture implemented in the research repository's src/models.py."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass
class ModelRegistry:
    manifest: dict = field(default_factory=dict)
    binary_model: Any = None
    sigma_model: nn.Module | None = None
    binary_model_name: str = ""
    sigma_model_name: str = ""
    warnings: list[str] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def models_loaded(self) -> bool:
        return self.binary_model is not None and self.sigma_model is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_task_model(task: dict, device: torch.device) -> nn.Module:
    artifact_path = ARTIFACT_DIR.parent / task["artifact_path"]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {artifact_path}")
    expected_hash = task.get("sha256", "").lower()
    if expected_hash and _sha256(artifact_path) != expected_hash:
        raise ValueError(f"Checkpoint checksum mismatch: {artifact_path.name}")
    model = CNNPromoterClassifier(
        num_classes=len(task["labels"]),
        dropout=float(task["dropout"]),
    )
    try:
        state_dict = torch.load(artifact_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(artifact_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _load_binary_random_forest(task: dict) -> dict:
    artifact_path = ARTIFACT_DIR.parent / task["artifact_path"]
    config_path = ARTIFACT_DIR.parent / task["preprocessing_config"]
    if not artifact_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Random Forest model or preprocessing config is missing.")
    if task.get("sha256") and _sha256(artifact_path) != task["sha256"].lower():
        raise ValueError("Random Forest artifact checksum mismatch.")
    if (
        task.get("preprocessing_config_sha256")
        and _sha256(config_path) != task["preprocessing_config_sha256"].lower()
    ):
        raise ValueError("Random Forest preprocessing config checksum mismatch.")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    bundle = joblib.load(artifact_path)
    required = {"base_model", "calibrated_model", "kmer_size", "labels"}
    if not required.issubset(bundle):
        raise ValueError("Random Forest bundle is missing required fields.")
    if bundle["labels"] != task["labels"] or config["labels"] != task["labels"]:
        raise ValueError("Random Forest label order does not match the manifest.")
    if int(bundle["kmer_size"]) != int(task["kmer_size"]):
        raise ValueError("Random Forest k-mer size does not match the manifest.")
    return bundle


def load_models() -> ModelRegistry:
    registry = ModelRegistry(manifest=load_manifest())
    registry.warnings.extend(registry.manifest.get("warnings", []))
    binary = registry.manifest["tasks"]["binary"]
    try:
        registry.binary_model = _load_binary_random_forest(binary)
        registry.binary_model_name = binary["selected_model"]
    except Exception as error:
        registry.warnings.append(f"Primary binary model failed to load: {error}")
        fallback = binary.get("fallback_model")
        if fallback:
            try:
                registry.binary_model = _load_task_model(fallback, registry.device)
                registry.binary_model_name = fallback["selected_model"]
                registry.warnings.append("Binary CNN fallback is active.")
            except Exception as fallback_error:
                registry.warnings.append(
                    f"Binary CNN fallback failed to load: {fallback_error}"
                )
    sigma = registry.manifest["tasks"]["sigma"]
    try:
        registry.sigma_model = _load_task_model(sigma, registry.device)
        registry.sigma_model_name = sigma["selected_model"]
    except Exception as error:
        registry.warnings.append(f"Sigma model failed to load: {error}")
    return registry
