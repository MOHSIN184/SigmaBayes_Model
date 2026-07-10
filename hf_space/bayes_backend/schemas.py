from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .preprocessing import validate_sequence


class PredictionRequest(BaseModel):
    sequence: str = Field(..., description="Exactly 81 DNA bases.")
    run_binary: bool = True
    run_sigma: bool = True

    @field_validator("sequence")
    @classmethod
    def validate_dna(cls, value: str) -> str:
        return validate_sequence(value)


class TaskPrediction(BaseModel):
    available: bool
    model_name: str = ""
    predicted_label: str = ""
    confidence: Optional[float] = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    calibrated_probabilities: dict[str, float] = Field(default_factory=dict)
    calibration_available: bool = False
    conformal_set_90: list[str] = Field(default_factory=list)
    conformal_set_95: list[str] = Field(default_factory=list)
    conformal_available: bool = False


class UncertaintyOutput(BaseModel):
    available: bool
    entropy: Optional[float] = None
    mutual_information: Optional[float] = None
    note: str = ""


class SigmaPrediction(TaskPrediction):
    uncertainty: UncertaintyOutput


class PredictionResponse(BaseModel):
    valid: bool
    sequence: str
    sequence_length: int
    gc_content: float
    binary: TaskPrediction
    sigma: SigmaPrediction
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    binary_model_loaded: bool
    sigma_model_loaded: bool
    binary_model_name: str
    sigma_model_name: str
    calibration_available: bool
    conformal_available: bool
    warnings: list[str]


class MetadataResponse(BaseModel):
    project: str
    input_length: int
    alphabet: list[str]
    binary_labels: list[str]
    sigma_labels: list[str]
    available_outputs: list[str]
    loaded_models: dict[str, str]
    model_manifest: dict[str, Any]
