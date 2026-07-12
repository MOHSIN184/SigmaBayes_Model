"""Free CPU Gradio Space adapter for verified BayesSigma inference."""
from __future__ import annotations

import os
from typing import Any

import gradio as gr

from bayes_backend.inference import predict
from bayes_backend.model_loader import load_models
from bayes_backend.preprocessing import normalize_sequence, validate_sequence

REGISTRY = load_models()

def unavailable_task() -> dict[str, Any]:
    return {"available": False, "model_name": "", "predicted_label": "", "confidence": None, "probabilities": {}, "calibrated_probabilities": {}, "calibration_available": False, "conformal_set_90": [], "conformal_set_95": [], "conformal_available": False}

def predict_sequence(sequence: str, run_binary: bool, run_sigma: bool) -> dict[str, Any]:
    """Validate input and return the same JSON contract as FastAPI."""
    normalized = normalize_sequence(sequence or "")
    try:
        normalized = validate_sequence(normalized)
    except ValueError as error:
        sigma = unavailable_task()
        sigma["uncertainty"] = {"available": False, "entropy": None, "mutual_information": None, "note": "Prediction was not run because input validation failed."}
        return {"valid": False, "sequence": normalized, "sequence_length": len(normalized), "gc_content": None, "binary": unavailable_task(), "sigma": sigma, "warnings": [str(error)]}
    if not run_binary and not run_sigma:
        sigma = unavailable_task()
        sigma["uncertainty"] = {"available": False, "entropy": None, "mutual_information": None, "note": "Sigma inference was not requested."}
        return {"valid": True, "sequence": normalized, "sequence_length": 81, "gc_content": (normalized.count("G") + normalized.count("C")) / 81, "binary": unavailable_task(), "sigma": sigma, "warnings": ["Select at least one prediction task."]}
    return predict(REGISTRY, normalized, bool(run_binary), bool(run_sigma)).model_dump(mode="json")

with gr.Blocks(title="BayesSigma") as demo:
    gr.Markdown(
        "# BayesSigma\n"
        "Reliability-aware promoter and sigma-factor inference from verified artifacts."
    )
    sequence_input = gr.Textbox(
        label="81 bp DNA sequence",
        lines=4,
        placeholder="Use only A, C, G, and T",
    )
    with gr.Row():
        binary_input = gr.Checkbox(
            value=True, label="Run binary promoter prediction"
        )
        sigma_input = gr.Checkbox(
            value=True, label="Run sigma-factor prediction"
        )
    submit = gr.Button("Run BayesSigma", variant="primary")
    output = gr.JSON(label="BayesSigma prediction")
    submit.click(
        fn=predict_sequence,
        inputs=[sequence_input, binary_input, sigma_input],
        outputs=output,
        api_name="predict",
    )
    gr.Examples(
        examples=[[
            "AAGTCATGAAACGATTCAAACATGGCGCGAATATTTATGTGATGCCTCCTTTACCGTCGCTCTCTGGTTAACACCCCATGC",
            True,
            True,
        ]],
        inputs=[sequence_input, binary_input, sigma_input],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
