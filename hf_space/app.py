"""Lightweight production web service for BayesSigma inference."""
from __future__ import annotations

import os
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bayes_backend.inference import predict
from bayes_backend.model_loader import ModelRegistry, load_models
from bayes_backend.preprocessing import normalized_kmer_vector
from bayes_backend.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from bayes_backend.utils import cors_origins
from web_utils import MAX_RECORDS, validate_web_sequence

APP_VERSION = "2.2.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_SIGMA_BATCH = 25
MAX_REQUEST_BYTES = 2 * 1024 * 1024

# A single CPU worker prevents PyTorch and BLAS from oversubscribing Render's free CPU.
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

REGISTRY: ModelRegistry = load_models()
INFERENCE_LOCK = threading.Lock()
LEGACY_LOCK = threading.Lock()
LEGACY_RESULTS: OrderedDict[str, dict] = OrderedDict()


class BatchRecord(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    sequence: str


class BatchRequest(BaseModel):
    records: list[BatchRecord] = Field(min_length=1, max_length=MAX_RECORDS)
    run_binary: bool = True
    run_sigma: bool = False


class BatchResult(BaseModel):
    id: str
    length: int
    prediction: str = ""
    confidence: Optional[float] = None
    sigma_factor: str = ""
    processing_time_ms: float
    status: str
    error: str = ""


class BatchResponse(BaseModel):
    results: list[BatchResult]
    processed: int
    failed: int
    total_time_ms: float


app = FastAPI(
    title="BayesSigma API",
    description="Resource-efficient promoter and sigma-factor inference.",
    version=APP_VERSION,
)
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def cache_static_assets(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if (
        request.method == "POST"
        and content_length
        and content_length.isdigit()
        and int(content_length) > MAX_REQUEST_BYTES
    ):
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body exceeds the 2 MB limit."},
        )
    response = await call_next(request)
    if request.url.path.startswith("/static/") and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=604800, immutable"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    tasks = REGISTRY.manifest["tasks"]
    calibration = all(
        bool(tasks[name].get("temperature") or tasks[name].get("calibration_artifact"))
        for name in ("binary", "sigma")
    )
    conformal = all(
        bool(tasks[name].get("conformal_thresholds")) for name in ("binary", "sigma")
    )
    return HealthResponse(
        status="ok" if REGISTRY.models_loaded else "degraded",
        models_loaded=REGISTRY.models_loaded,
        binary_model_loaded=REGISTRY.binary_model is not None,
        sigma_model_loaded=REGISTRY.sigma_model is not None,
        binary_model_name=REGISTRY.binary_model_name,
        sigma_model_name=REGISTRY.sigma_model_name,
        calibration_available=calibration,
        conformal_available=conformal,
        warnings=REGISTRY.warnings,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    manifest = REGISTRY.manifest
    return MetadataResponse(
        project=manifest["project"],
        input_length=manifest["input_length"],
        alphabet=manifest["alphabet"],
        binary_labels=manifest["tasks"]["binary"]["labels"],
        sigma_labels=manifest["tasks"]["sigma"]["labels"],
        available_outputs=manifest["available_outputs"],
        loaded_models={
            "binary": REGISTRY.binary_model_name,
            "sigma": REGISTRY.sigma_model_name,
        },
        model_manifest=manifest,
    )


@app.post("/predict", response_model=PredictionResponse)
def prediction(payload: PredictionRequest) -> PredictionResponse:
    with INFERENCE_LOCK:
        return predict(
            REGISTRY,
            payload.sequence,
            payload.run_binary,
            payload.run_sigma,
        )


@app.post("/gradio_api/call/v2/predict", include_in_schema=False)
def legacy_prediction_start(payload: PredictionRequest) -> dict[str, str]:
    """Keep the former Gradio call URL available for existing direct clients."""
    result = prediction(payload).model_dump(mode="json")
    event_id = uuid.uuid4().hex
    with LEGACY_LOCK:
        LEGACY_RESULTS[event_id] = result
        while len(LEGACY_RESULTS) > 32:
            LEGACY_RESULTS.popitem(last=False)
    return {"event_id": event_id}


@app.get("/gradio_api/call/predict/{event_id}", include_in_schema=False)
def legacy_prediction_result(event_id: str) -> Response:
    with LEGACY_LOCK:
        result = LEGACY_RESULTS.pop(event_id, None)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction event not found.")
    payload = json.dumps([result], separators=(",", ":"))
    return Response(
        content=f"event: complete\ndata: {payload}\n\n",
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def _predict_record(record: BatchRecord, run_binary: bool, run_sigma: bool) -> BatchResult:
    started = time.perf_counter()
    try:
        sequence = validate_web_sequence(record.sequence)
        output = predict(REGISTRY, sequence, run_binary, run_sigma)
        return BatchResult(
            id=record.id,
            length=len(sequence),
            prediction=output.binary.predicted_label,
            confidence=output.binary.confidence,
            sigma_factor=output.sigma.predicted_label,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            status="Success",
        )
    except (RuntimeError, ValueError) as error:
        return BatchResult(
            id=record.id,
            length=len(record.sequence.strip()),
            processing_time_ms=(time.perf_counter() - started) * 1000,
            status="Invalid",
            error=str(error),
        )


def _binary_only_batch(records: list[BatchRecord]) -> list[BatchResult]:
    """Vectorize Random Forest inference in small chunks to cap peak memory."""
    bundle = REGISTRY.binary_model
    if not isinstance(bundle, dict):
        return [_predict_record(record, True, False) for record in records]

    results: list[BatchResult | None] = [None] * len(records)
    valid: list[tuple[int, BatchRecord, str, float]] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        try:
            sequence = validate_web_sequence(record.sequence)
            valid.append(
                (index, record, sequence, (time.perf_counter() - started) * 1000)
            )
        except ValueError as error:
            results[index] = BatchResult(
                id=record.id,
                length=len(record.sequence.strip()),
                processing_time_ms=(time.perf_counter() - started) * 1000,
                status="Invalid",
                error=str(error),
            )

    labels = REGISTRY.manifest["tasks"]["binary"]["labels"]
    kmer_size = int(bundle["kmer_size"])
    chunk_size = 128
    for offset in range(0, len(valid), chunk_size):
        chunk = valid[offset : offset + chunk_size]
        inference_started = time.perf_counter()
        features = np.stack(
            [normalized_kmer_vector(item[2], kmer_size) for item in chunk]
        )
        probabilities = bundle["calibrated_model"].predict_proba(features)
        per_record_ms = (time.perf_counter() - inference_started) * 1000 / len(chunk)
        for item, probability in zip(chunk, probabilities):
            index, record, sequence, validation_ms = item
            prediction_index = int(np.argmax(probability))
            results[index] = BatchResult(
                id=record.id,
                length=len(sequence),
                prediction=labels[prediction_index],
                confidence=float(probability[prediction_index]),
                processing_time_ms=validation_ms + per_record_ms,
                status="Success",
            )
        del features, probabilities
    return [result for result in results if result is not None]


@app.post("/predict-batch", response_model=BatchResponse)
def batch_prediction(payload: BatchRequest) -> BatchResponse:
    if not payload.run_binary and not payload.run_sigma:
        raise HTTPException(status_code=400, detail="Select at least one prediction task.")
    if payload.run_sigma and len(payload.records) > MAX_SIGMA_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sigma-factor batches are limited to {MAX_SIGMA_BATCH} sequences; "
                "disable sigma-factor prediction for larger promoter batches."
            ),
        )

    started = time.perf_counter()
    with INFERENCE_LOCK:
        if payload.run_binary and not payload.run_sigma:
            results = _binary_only_batch(payload.records)
        else:
            results = [
                _predict_record(record, payload.run_binary, payload.run_sigma)
                for record in payload.records
            ]
    failed = sum(result.status != "Success" for result in results)
    return BatchResponse(
        results=results,
        processed=len(results) - failed,
        failed=failed,
        total_time_ms=(time.perf_counter() - started) * 1000,
    )
