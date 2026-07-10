# BayesSigma inference API

This FastAPI service performs real inference from packaged BayesSigma PyTorch state dictionaries. It is self-contained for Docker deployment on Hugging Face Spaces: the exact CNN architecture and one-hot preprocessing are included under `app/`, and selected checkpoints are under `artifacts/`.

## Models and outputs

| Task | Served model | Selection |
|---|---|---|
| Binary | `calibrated_5mer_random_forest` | Reproduces the manuscript workflow: 500 balanced trees, normalized 5-mers, seed 42, and isotonic validation-split calibration. |
| Sigma | `safe_f1_selected_sigma_cnn` | Recorded final sigma CNN selected by validation macro F1. |

The binary task returns base Random Forest probabilities plus isotonic-calibrated probabilities. Its 90%/95% conformal `qhat` values are reproduced from the independent calibration split. The sigma CNN returns raw and temperature-scaled probabilities. Sets use `probability >= 1 - qhat`. Sigma inference also returns predictive entropy and mutual information from 30 MC-dropout passes.

Limitations: models are dataset-specific, lack cross-species and wet-lab validation, sigma classes are imbalanced, and sigma prediction is meaningful primarily for promoter-like sequences. The binary CNN remains packaged strictly as an automatic fallback.

## Endpoints

- `GET /health`: model and reliability-artifact status
- `GET /metadata`: input contract, labels, availability, and manifest
- `POST /predict`: binary and/or sigma inference
- `GET /docs`: OpenAPI interface

Sequences are stripped of whitespace, uppercased, and must contain exactly 81 A/C/G/T bases.

## Run locally

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Windows Command Prompt test:

```bat
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"sequence\":\"AAGTCATGAAACGATTCAAACATGGCGCGAATATTTATGTGATGCCTCCTTTACCGTCGCTCTCTGGTTAACACCCCATGC\",\"run_binary\":true,\"run_sigma\":true}"
```

Open `http://localhost:8000/health` and `http://localhost:8000/docs`.

## Re-export artifacts

From the project root, reproduce and verify the binary artifact first:

```powershell
python server/export_binary_random_forest.py
python server/export_inference_artifacts.py
```

The Random Forest exporter refuses to save unless held-out accuracy, macro F1, AUROC, and both conformal quantiles reproduce the recorded result. The general exporter then refreshes checksums and the manifest.

## Hugging Face Spaces

1. Create a new Hugging Face Space and select **Docker** as SDK.
2. Copy the contents of `server/` to the Space repository root.
3. Ensure `Dockerfile` is at the Space root, then commit and push.
4. Wait for the build and test `https://YOUR-SPACE.hf.space/health` and `/docs`.
5. Enter `https://YOUR-SPACE.hf.space` in the GitHub Pages demo API URL field.

The image exposes port 7860 and starts `uvicorn app.main:app --host 0.0.0.0 --port 7860`. Additional origins may be supplied through `CORS_ORIGINS`.

Exact frontend API placeholder: `https://YOUR-SPACE.hf.space`.
