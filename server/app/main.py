from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict
from .model_loader import ModelRegistry, load_models
from .schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from .utils import cors_origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = load_models()
    yield


app = FastAPI(
    title=os.getenv("API_TITLE", "BayesSigma API"),
    description="Artifact-backed promoter and sigma-factor inference.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry = _registry(request)
    tasks = registry.manifest["tasks"]
    calibration = all(
        bool(tasks[name].get("temperature") or tasks[name].get("calibration_artifact"))
        for name in ("binary", "sigma")
    )
    conformal = all(
        bool(tasks[name].get("conformal_thresholds"))
        for name in ("binary", "sigma")
    )
    return HealthResponse(
        status="ok" if registry.models_loaded else "degraded",
        models_loaded=registry.models_loaded,
        binary_model_loaded=registry.binary_model is not None,
        sigma_model_loaded=registry.sigma_model is not None,
        binary_model_name=registry.binary_model_name,
        sigma_model_name=registry.sigma_model_name,
        calibration_available=calibration,
        conformal_available=conformal,
        warnings=registry.warnings,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request) -> MetadataResponse:
    registry = _registry(request)
    manifest = registry.manifest
    return MetadataResponse(
        project=manifest["project"],
        input_length=manifest["input_length"],
        alphabet=manifest["alphabet"],
        binary_labels=manifest["tasks"]["binary"]["labels"],
        sigma_labels=manifest["tasks"]["sigma"]["labels"],
        available_outputs=manifest["available_outputs"],
        loaded_models={
            "binary": registry.binary_model_name,
            "sigma": registry.sigma_model_name,
        },
        model_manifest=manifest,
    )


@app.post("/predict", response_model=PredictionResponse)
def prediction(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    return predict(
        _registry(request),
        payload.sequence,
        payload.run_binary,
        payload.run_sigma,
    )
