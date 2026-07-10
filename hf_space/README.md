---
title: BayesSigma API
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: 3.11
pinned: false
---

# BayesSigma Gradio Space

This is the free CPU Gradio deployment of BayesSigma. The `bayes_backend/` package is a verbatim vendored copy of the tested `server/app` inference modules; preprocessing and model logic are unchanged.

Served outputs:

- calibrated 5-mer Random Forest binary prediction
- binary base and isotonic-calibrated probabilities
- selected safe sigma CNN prediction
- sigma raw and temperature-scaled probabilities
- 90% and 95% split-conformal sets
- sigma MC-dropout predictive entropy and mutual information
- input length and GC content

The model and preprocessing artifacts are checksum-validated at startup. The binary CNN remains a fallback if the Random Forest cannot load.

## Local test

```powershell
cd hf_space
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

## Deploy on Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose **Gradio** as the SDK.
3. Choose the **Blank** template.
4. Keep free CPU hardware; no paid hardware or Docker SDK is required.
5. Copy every file and directory inside `hf_space/` to the Space repository root.
6. Commit and push the Space repository.
7. Wait for the build, then test an 81 bp A/C/G/T sequence.
8. Click **Use via API** and confirm that `/predict` is listed.
9. In GitHub Pages, enter `YOUR_USERNAME/YOUR_SPACE_NAME` or the Space URL.

No `Dockerfile` is used or required.

## JavaScript client

The named endpoint is `/predict`:

```javascript
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const client = await Client.connect("YOUR_USERNAME/YOUR_SPACE_NAME");
const result = await client.predict("/predict", [
  "AAGTCATGAAACGATTCAAACATGGCGCGAATATTTATGTGATGCCTCCTTTACCGTCGCTCTCTGGTTAACACCCCATGC",
  true,
  true
]);
const prediction = typeof result.data[0] === "string" ? JSON.parse(result.data[0]) : result.data[0];
console.log(prediction);
```

Public Spaces are callable from GitHub Pages through the Gradio client. Private Spaces require an access token; never expose it in client-side code.

## Scientific scope

These are dataset-specific research predictions. They are not cross-species, clinical, or wet-lab validated. Sigma prediction is biologically most meaningful for promoter-like sequences.
