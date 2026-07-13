from __future__ import annotations

import numpy as np
import torch

from .conformal import conformal_sets
from .model_loader import ModelRegistry
from .preprocessing import gc_content, normalized_kmer_vector, one_hot_encode
from .schemas import (
    PredictionResponse,
    SigmaPrediction,
    TaskPrediction,
    UncertaintyOutput,
)
from .uncertainty import mc_dropout


def _softmax(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]


def _unavailable_sigma(note: str) -> SigmaPrediction:
    return SigmaPrediction(
        available=False,
        uncertainty=UncertaintyOutput(available=False, note=note),
    )


def _predict_task(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    config: dict,
) -> tuple[TaskPrediction, np.ndarray]:
    with torch.inference_mode():
        logits = model(inputs)
    raw = _softmax(logits)
    labels = config["labels"]
    temperature = config.get("temperature")
    calibrated = _softmax(logits, float(temperature)) if temperature else None
    conformal_probabilities = calibrated if calibrated is not None else raw
    set_90, set_95 = conformal_sets(
        conformal_probabilities,
        labels,
        config.get("conformal_thresholds"),
    )
    prediction_index = int(np.argmax(raw))
    output = TaskPrediction(
        available=True,
        model_name=config["selected_model"],
        predicted_label=labels[prediction_index],
        confidence=float(raw[prediction_index]),
        probabilities=dict(zip(labels, map(float, raw))),
        calibrated_probabilities=(
            dict(zip(labels, map(float, calibrated))) if calibrated is not None else {}
        ),
        calibration_available=calibrated is not None,
        conformal_set_90=set_90,
        conformal_set_95=set_95,
        conformal_available=bool(config.get("conformal_thresholds")),
    )
    return output, raw


def _predict_binary_random_forest(
    bundle: dict,
    sequence: str,
    config: dict,
) -> TaskPrediction:
    features = normalized_kmer_vector(sequence, int(bundle["kmer_size"])).reshape(1, -1)
    raw = bundle["base_model"].predict_proba(features)[0]
    calibrated = bundle["calibrated_model"].predict_proba(features)[0]
    labels = config["labels"]
    set_90, set_95 = conformal_sets(
        calibrated, labels, config["conformal_thresholds"]
    )
    prediction_index = int(np.argmax(calibrated))
    return TaskPrediction(
        available=True,
        model_name=config["selected_model"],
        predicted_label=labels[prediction_index],
        confidence=float(calibrated[prediction_index]),
        probabilities=dict(zip(labels, map(float, raw))),
        calibrated_probabilities=dict(zip(labels, map(float, calibrated))),
        calibration_available=True,
        conformal_set_90=set_90,
        conformal_set_95=set_95,
        conformal_available=True,
    )


def predict(
    registry: ModelRegistry,
    sequence: str,
    run_binary: bool,
    run_sigma: bool,
) -> PredictionResponse:
    needs_tensor = (
        run_sigma and registry.sigma_model is not None
    ) or (
        run_binary
        and registry.binary_model is not None
        and not isinstance(registry.binary_model, dict)
    )
    inputs = (
        torch.from_numpy(one_hot_encode(sequence)).unsqueeze(0).to(registry.device)
        if needs_tensor
        else None
    )
    warnings = list(registry.warnings)
    binary = TaskPrediction(available=False)
    sigma = _unavailable_sigma("Sigma inference was not requested.")

    if run_binary:
        if registry.binary_model is None:
            warnings.append("Binary prediction requested but its model is unavailable.")
        elif isinstance(registry.binary_model, dict):
            binary = _predict_binary_random_forest(
                registry.binary_model,
                sequence,
                registry.manifest["tasks"]["binary"],
            )
        else:
            assert inputs is not None
            binary, _ = _predict_task(
                registry.binary_model,
                inputs,
                registry.manifest["tasks"]["binary"]["fallback_model"],
            )

    if run_sigma:
        if registry.sigma_model is None:
            sigma = _unavailable_sigma("Sigma model is unavailable.")
            warnings.append("Sigma prediction requested but its model is unavailable.")
        else:
            assert inputs is not None
            task_output, _ = _predict_task(
                registry.sigma_model,
                inputs,
                registry.manifest["tasks"]["sigma"],
            )
            uncertainty = mc_dropout(
                registry.sigma_model,
                inputs,
                passes=int(registry.manifest["tasks"]["sigma"]["mc_dropout_passes"]),
            )
            sigma = SigmaPrediction(
                **task_output.model_dump(),
                uncertainty=UncertaintyOutput(
                    available=True,
                    entropy=uncertainty["entropy"],
                    mutual_information=uncertainty["mutual_information"],
                    note=(
                        f"Estimated from {uncertainty['passes']} stochastic "
                        "dropout forward passes."
                    ),
                ),
            )
            if binary.available and binary.predicted_label == "Non-Promoter":
                warnings.append(
                    "Sigma-factor prediction is biologically meaningful primarily "
                    "for promoter-like sequences."
                )

    return PredictionResponse(
        valid=True,
        sequence=sequence,
        sequence_length=len(sequence),
        gc_content=gc_content(sequence),
        binary=binary,
        sigma=sigma,
        warnings=warnings,
    )
