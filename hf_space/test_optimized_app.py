"""Regression tests for optimized inference and HTTP behavior."""
from __future__ import annotations

import itertools

import numpy as np
import torch
from fastapi.testclient import TestClient

from app import app
from bayes_backend.preprocessing import normalized_kmer_vector
from bayes_backend.uncertainty import mc_dropout

PROMOTER = "TAGATGTCCTTGATTAACACCAAAATTAAACCTTTTAAAAACCAGGCATTCAAAAACGGCGAATTCATCGAAATCACCGAA"
NON_PROMOTER = "TTCAAAGGTGAAACTCAGGTTACTGACCAGCTGACCGGTTACGGCCAGTGGGAATATCAGATCCAGGGCAACAGCGCTGAA"


def _reference_vector(sequence: str, k: int) -> np.ndarray:
    kmers = ["".join(chars) for chars in itertools.product("ACGT", repeat=k)]
    lookup = {kmer: index for index, kmer in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    for start in range(len(sequence) - k + 1):
        counts[lookup[sequence[start : start + k]]] += 1
    return counts / (len(sequence) - k + 1)


def test_kmer_optimization_preserves_feature_order() -> None:
    assert np.array_equal(
        normalized_kmer_vector(PROMOTER, 5), _reference_vector(PROMOTER, 5)
    )


def test_health_and_lightweight_homepage() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["binary_model_name"] == "calibrated_5mer_random_forest"
    page = client.get("/")
    assert page.status_code == 200
    assert "Promoter Prediction Web Server" in page.text
    for feature in ("Use Example", "Reset", "Upload FASTA", "Citation", "Useful Resources"):
        assert feature in page.text
    assert len(page.content) < 10_000


def test_single_prediction_contract_is_preserved() -> None:
    response = TestClient(app).post(
        "/predict",
        json={"sequence": PROMOTER, "run_binary": True, "run_sigma": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["binary"]["predicted_label"] == "Promoter"
    assert payload["warnings"] == []


def test_former_gradio_call_url_remains_compatible() -> None:
    client = TestClient(app)
    started = client.post(
        "/gradio_api/call/v2/predict",
        json={"sequence": PROMOTER, "run_binary": True, "run_sigma": False},
    )
    assert started.status_code == 200
    completed = client.get(
        f"/gradio_api/call/predict/{started.json()['event_id']}"
    )
    assert completed.status_code == 200
    assert "event: complete" in completed.text
    assert "calibrated_5mer_random_forest" in completed.text


def test_vectorized_batch_handles_valid_and_invalid_records() -> None:
    response = TestClient(app).post(
        "/predict-batch",
        json={
            "records": [
                {"id": "promoter", "sequence": PROMOTER},
                {"id": "non_promoter", "sequence": NON_PROMOTER},
                {"id": "short", "sequence": PROMOTER[:-1]},
            ],
            "run_binary": True,
            "run_sigma": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed"] == 2
    assert payload["failed"] == 1
    assert [item["prediction"] for item in payload["results"][:2]] == [
        "Promoter",
        "Non-Promoter",
    ]
    assert "shorter than 81 bp" in payload["results"][2]["error"]


def test_mc_dropout_chunking_preserves_requested_sample_count() -> None:
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(4 * 81, 6),
    )
    output = mc_dropout(model, torch.ones((1, 4, 81)), passes=30)
    assert output["passes"] == 30
    assert output["mean_probabilities"].shape == (6,)
    assert np.isfinite(output["entropy"])


def test_sigma_batch_limit_protects_free_instance() -> None:
    response = TestClient(app).post(
        "/predict-batch",
        json={
            "records": [
                {"id": f"sequence_{index}", "sequence": PROMOTER}
                for index in range(26)
            ],
            "run_binary": True,
            "run_sigma": True,
        },
    )
    assert response.status_code == 400
    assert "limited to 25 sequences" in response.json()["detail"]
