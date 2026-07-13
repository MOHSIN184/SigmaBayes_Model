# BayesSigma lightweight web service

This directory contains the Render free-tier deployment. A native FastAPI service
serves a dependency-free HTML/CSS/JavaScript interface and the verified BayesSigma
models. The calibrated 5-mer Random Forest handles promoter prediction; the selected
safe CNN provides optional sigma-factor prediction and MC-dropout uncertainty.

## Resource controls

- one Uvicorn worker and one inference lock
- one PyTorch/BLAS thread to avoid CPU oversubscription
- CPU-only PyTorch wheel in Docker
- models checksum-validated and loaded once at startup
- binary batches vectorized in bounded chunks of 128 records
- MC-dropout samples evaluated in bounded chunks of 10
- browser-streamed FASTA parsing with 10 MB / 5,000-record limits
- optional sigma-factor batches limited to 25 records on free CPU
- gzip responses and versioned static assets cached for seven days

## Endpoints

- `GET /`: responsive browser interface
- `GET /health`: loaded-model and artifact status
- `GET /metadata`: model and input metadata
- `POST /predict`: detailed single-sequence contract
- `POST /predict-batch`: compact batch results used by the browser
- `GET /docs`: OpenAPI documentation

## Local run

```powershell
cd hf_space
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1 --no-access-log
```

Open `http://127.0.0.1:7860`.

## Scientific scope

Predictions are dataset-specific research outputs. They are not cross-species,
clinical, or wet-lab validated. Sigma-factor prediction is biologically most
meaningful for promoter-like sequences.
