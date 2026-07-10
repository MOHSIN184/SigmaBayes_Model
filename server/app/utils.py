from __future__ import annotations

import json
import os
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = SERVER_ROOT / "artifacts"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def cors_origins() -> list[str]:
    defaults = [
        "http://localhost:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://mohsin184.github.io",
        "https://mohsin184.github.io/SigmaBayes_Model",
        "https://mohsin184.github.io/SigmaBayes_Model/",
    ]
    configured = os.getenv("CORS_ORIGINS")
    if configured:
        defaults.extend(item.strip().rstrip("/") for item in configured.split(","))
    return sorted(set(defaults))
