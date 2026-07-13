"""Publication-quality Gradio interface for verified BayesSigma inference."""
from __future__ import annotations

import html
import os
import time
from typing import Any

import gradio as gr

from bayes_backend.inference import predict
from bayes_backend.model_loader import load_models
from bayes_backend.preprocessing import normalize_sequence, validate_sequence
from web_utils import SequenceRecord, collect_records, validate_web_sequence

APP_VERSION = "2.0.0"
EXAMPLE_SEQUENCE = (
    "AAGTCATGAAACGATTCAAACATGGCGCGAATATTTATGTGATGCCTCCTTTACCGTCGCTCTCTGGTTAACACCCCATGC"
)
RESULT_HEADERS = [
    "Sequence ID",
    "Length",
    "Prediction",
    "Confidence Score",
    "Sigma Factor",
    "Processing Time",
    "Status",
]

REGISTRY = load_models()


def unavailable_task() -> dict[str, Any]:
    return {
        "available": False,
        "model_name": "",
        "predicted_label": "",
        "confidence": None,
        "probabilities": {},
        "calibrated_probabilities": {},
        "calibration_available": False,
        "conformal_set_90": [],
        "conformal_set_95": [],
        "conformal_available": False,
    }


def predict_sequence(sequence: str, run_binary: bool, run_sigma: bool) -> dict[str, Any]:
    """Preserve the original public API contract for single-sequence clients."""
    normalized = normalize_sequence(sequence or "")
    try:
        normalized = validate_sequence(normalized)
    except ValueError as error:
        sigma = unavailable_task()
        sigma["uncertainty"] = {
            "available": False,
            "entropy": None,
            "mutual_information": None,
            "note": "Prediction was not run because input validation failed.",
        }
        return {
            "valid": False,
            "sequence": normalized,
            "sequence_length": len(normalized),
            "gc_content": None,
            "binary": unavailable_task(),
            "sigma": sigma,
            "warnings": [str(error)],
        }
    if not run_binary and not run_sigma:
        sigma = unavailable_task()
        sigma["uncertainty"] = {
            "available": False,
            "entropy": None,
            "mutual_information": None,
            "note": "Sigma inference was not requested.",
        }
        return {
            "valid": True,
            "sequence": normalized,
            "sequence_length": 81,
            "gc_content": (normalized.count("G") + normalized.count("C")) / 81,
            "binary": unavailable_task(),
            "sigma": sigma,
            "warnings": ["Select at least one prediction task."],
        }
    return predict(REGISTRY, normalized, bool(run_binary), bool(run_sigma)).model_dump(
        mode="json"
    )


def _alert(message: str, kind: str = "info") -> str:
    icons = {"success": "✓", "error": "!", "info": "i"}
    safe_message = html.escape(message)
    return (
        f'<div class="alert alert-{kind}" role="status" aria-live="polite">'
        f'<span class="alert-icon" aria-hidden="true">{icons[kind]}</span>'
        f"<span>{safe_message}</span></div>"
    )


def _empty_results() -> list[list[Any]]:
    return []


def _single_result_card(
    record: SequenceRecord, result: dict[str, Any], elapsed: float
) -> str:
    binary = result["binary"]
    sigma = result["sigma"]
    prediction = binary.get("predicted_label") or "Not requested"
    confidence = binary.get("confidence")
    confidence_text = f"{confidence * 100:.2f}%" if confidence is not None else "—"
    badge_class = "promoter" if prediction == "Promoter" else "non-promoter"
    sigma_label = sigma.get("predicted_label") or "Not requested"
    return f"""
    <section class="result-card" aria-label="Prediction summary">
      <div class="result-card-heading">
        <div><span class="eyebrow">Prediction complete</span><h3>{html.escape(record.identifier)}</h3></div>
        <span class="prediction-badge {badge_class}">{html.escape(prediction)}</span>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>Confidence score</span><strong>{confidence_text}</strong></div>
        <div class="metric"><span>Sequence length</span><strong>{len(record.sequence)} bp</strong></div>
        <div class="metric"><span>Sigma factor</span><strong>{html.escape(sigma_label)}</strong></div>
        <div class="metric"><span>Processing time</span><strong>{elapsed:.3f} s</strong></div>
      </div>
    </section>
    """


def run_batch_prediction(
    manual_text: str,
    upload: Any,
    run_binary: bool,
    run_sigma: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, list[list[Any]], dict[str, Any]]:
    """Validate and predict records sequentially to keep memory use bounded."""
    if not run_binary and not run_sigma:
        message = "Select at least one prediction task before submitting."
        return _alert(message, "error"), "", _empty_results(), {}

    try:
        records = collect_records(manual_text, upload)
    except (OSError, ValueError) as error:
        return _alert(str(error), "error"), "", _empty_results(), {}

    rows: list[list[Any]] = []
    details: dict[str, Any] = {}
    valid_count = 0
    batch_started = time.perf_counter()
    single_card = ""

    for index, record in enumerate(records):
        progress((index, len(records)), desc=f"Analyzing sequence {index + 1} of {len(records)}")
        sequence_started = time.perf_counter()
        try:
            normalized = validate_web_sequence(record.sequence)
            result = predict_sequence(normalized, run_binary, run_sigma)
            elapsed = time.perf_counter() - sequence_started
            binary = result["binary"]
            sigma = result["sigma"]
            prediction = binary.get("predicted_label") or "Not requested"
            confidence = binary.get("confidence")
            confidence_text = (
                f"{confidence * 100:.2f}%" if confidence is not None else "—"
            )
            rows.append(
                [
                    record.identifier,
                    len(normalized),
                    prediction,
                    confidence_text,
                    sigma.get("predicted_label") or "Not requested",
                    f"{elapsed:.3f} s",
                    "Success",
                ]
            )
            details[record.identifier] = result
            valid_count += 1
            if len(records) == 1:
                single_card = _single_result_card(record, result, elapsed)
        except (RuntimeError, ValueError) as error:
            elapsed = time.perf_counter() - sequence_started
            rows.append(
                [
                    record.identifier,
                    len(record.sequence.strip()),
                    "—",
                    "—",
                    "—",
                    f"{elapsed:.3f} s",
                    str(error),
                ]
            )
            details[record.identifier] = {"valid": False, "error": str(error)}

    total_elapsed = time.perf_counter() - batch_started
    invalid_count = len(records) - valid_count
    if valid_count == len(records):
        status = _alert(
            f"Successfully processed {valid_count:,} sequence(s) in {total_elapsed:.2f} seconds.",
            "success",
        )
    elif valid_count:
        status = _alert(
            f"Processed {valid_count:,} valid sequence(s); {invalid_count:,} record(s) need attention.",
            "info",
        )
    else:
        status = _alert("No valid sequences were available for prediction.", "error")
    return status, single_card, rows, details


def use_example() -> tuple[str, None, str, str, list[list[Any]], dict[str, Any]]:
    return (
        EXAMPLE_SEQUENCE,
        None,
        _alert("Verified 81 bp example loaded. Select Predict to analyze it.", "info"),
        "",
        _empty_results(),
        {},
    )


def reset_interface() -> tuple[str, None, str, str, list[list[Any]], dict[str, Any]]:
    return (
        "",
        None,
        _alert("Inputs are ready for a new analysis.", "info"),
        "",
        _empty_results(),
        {},
    )


CSS = """
:root { --navy:#102a43; --blue:#176b87; --teal:#2a9d8f; --ink:#243b53; --muted:#627d98; --line:#d9e2ec; --surface:#ffffff; --bg:#f4f8fb; }
body, .gradio-container { background:var(--bg) !important; color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }
.gradio-container { max-width:none !important; padding:0 !important; }
.page-shell { max-width:1180px; margin:0 auto; padding:0 24px; }
.hero { background:linear-gradient(135deg,#0b2942 0%,#155e75 62%,#2a9d8f 100%); color:white; padding:56px 0 74px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; width:420px; height:420px; border:1px solid rgba(255,255,255,.16); border-radius:50%; right:-100px; top:-190px; box-shadow:0 0 0 55px rgba(255,255,255,.035),0 0 0 110px rgba(255,255,255,.025); }
.hero-inner { max-width:1180px; margin:auto; padding:0 24px; position:relative; z-index:1; }
.brand { display:flex; align-items:center; gap:12px; font-weight:750; letter-spacing:.02em; margin-bottom:48px; }
.brand-mark { width:40px; height:40px; border-radius:12px; background:rgba(255,255,255,.16); display:grid; place-items:center; border:1px solid rgba(255,255,255,.25); }
.hero h1 { color:white !important; font-size:clamp(2.35rem,5vw,4.4rem) !important; line-height:1.02 !important; letter-spacing:-.045em !important; max-width:780px; margin:0 0 22px !important; }
.hero p { max-width:690px; font-size:1.12rem; line-height:1.7; color:#e6f6f4; margin:0; }
.hero-tags { display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }
.hero-tag { border:1px solid rgba(255,255,255,.22); background:rgba(255,255,255,.1); border-radius:999px; padding:8px 13px; font-size:.82rem; font-weight:650; }
.workspace { margin-top:-38px; position:relative; z-index:2; padding-bottom:36px; }
.app-card, .content-card { background:var(--surface); border:1px solid rgba(188,204,220,.75); border-radius:20px; box-shadow:0 14px 40px rgba(16,42,67,.09); padding:26px !important; }
.section-title h2 { color:var(--navy) !important; margin:0 0 5px !important; font-size:1.4rem !important; }
.section-title p { color:var(--muted); margin:0 0 20px; font-size:.94rem; }
.input-panel { border:none !important; background:transparent !important; padding:0 !important; }
.input-panel textarea { font-family:"SFMono-Regular",Consolas,monospace !important; letter-spacing:.025em; line-height:1.65 !important; border-radius:12px !important; }
.input-panel textarea:focus, input:focus, button:focus-visible, a:focus-visible { outline:3px solid rgba(42,157,143,.28) !important; outline-offset:2px; }
.upload-panel { border:1px dashed #9fb3c8 !important; border-radius:14px !important; background:#f8fbfd !important; }
.task-options { background:#f5f9fc; border:1px solid #e2eaf1; border-radius:14px; padding:11px 15px !important; }
.primary-button { background:linear-gradient(135deg,#176b87,#21867a) !important; color:white !important; border:none !important; min-height:48px; font-weight:750 !important; box-shadow:0 7px 18px rgba(23,107,135,.22); }
.secondary-button { min-height:48px; border:1px solid #bcccdc !important; background:white !important; color:var(--navy) !important; font-weight:700 !important; }
.example-box { background:#eff8f7; border:1px solid #c6e8e2; border-radius:14px; padding:18px 20px; margin-top:10px; }
.example-box code { display:block; overflow-wrap:anywhere; color:#155e75; font-size:.84rem; line-height:1.6; margin:8px 0 5px; }
.example-meta { color:#52756f; font-size:.8rem; font-weight:650; }
.alert { display:flex; align-items:flex-start; gap:11px; border-radius:12px; padding:13px 15px; margin:12px 0 4px; font-size:.92rem; line-height:1.5; }
.alert-icon { display:grid; place-items:center; width:22px; height:22px; flex:0 0 22px; border-radius:50%; font-weight:800; }
.alert-info { background:#edf6ff; border:1px solid #c8e1f5; color:#174e73; }.alert-info .alert-icon{background:#cce5f7;}
.alert-success { background:#ecf9f2; border:1px solid #bde7cf; color:#17633a; }.alert-success .alert-icon{background:#bfe8d1;}
.alert-error { background:#fff2f1; border:1px solid #f4c7c3; color:#9b2c2c; }.alert-error .alert-icon{background:#f5ccc8;}
.results-wrap { margin-top:22px; }
.result-card { border:1px solid #d7e6e3; background:linear-gradient(145deg,#fff,#f4fbf9); border-radius:16px; padding:22px; margin:12px 0 18px; }
.result-card-heading { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
.result-card h3 { margin:3px 0 0; color:var(--navy); font-size:1.2rem; }.eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; font-weight:800; }
.prediction-badge { border-radius:999px; padding:8px 13px; font-weight:800; font-size:.82rem; }.prediction-badge.promoter{background:#d9f5e5;color:#126b3a}.prediction-badge.non-promoter{background:#fde2e0;color:#a42929}
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }.metric{background:white;border:1px solid #e2eaf1;border-radius:12px;padding:14px}.metric span{display:block;color:var(--muted);font-size:.74rem;margin-bottom:5px}.metric strong{color:var(--navy);font-size:.98rem}
.dataframe { border-radius:14px !important; overflow:hidden; }
.info-grid { display:grid; grid-template-columns:1.15fr .85fr; gap:22px; margin-top:22px; }
.content-card h2 { color:var(--navy) !important; font-size:1.3rem !important; margin-top:0 !important; }.content-card p{color:var(--muted);line-height:1.7}.citation{border-left:4px solid var(--teal);padding-left:16px;margin-top:15px}.citation strong{color:var(--navy)}
.resource-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:16px; }.resource-link{display:flex;justify-content:space-between;align-items:center;padding:12px 13px;border:1px solid var(--line);border-radius:10px;text-decoration:none!important;color:#176b87!important;font-weight:700;background:#fbfdff}.resource-link:hover{background:#edf8f6;border-color:#8ed3c8}
.footer { background:#102a43; color:#cbd8e5; padding:34px 24px; margin-top:24px; }.footer-inner{max-width:1180px;margin:auto;display:flex;justify-content:space-between;gap:25px;align-items:center}.footer strong{color:white}.footer a{color:#82d7cb!important}.footer-meta{text-align:right;font-size:.82rem;line-height:1.7}
@media (max-width:800px){.hero{padding:38px 0 62px}.brand{margin-bottom:32px}.page-shell{padding:0 14px}.app-card,.content-card{padding:18px!important;border-radius:16px}.metric-grid,.info-grid{grid-template-columns:1fr}.resource-grid{grid-template-columns:1fr}.footer-inner{align-items:flex-start;flex-direction:column}.footer-meta{text-align:left}}
"""


HEADER_HTML = """
<header class="hero">
  <div class="hero-inner">
    <div class="brand"><span class="brand-mark" aria-hidden="true">βΣ</span><span>BayesSigma Research Server</span></div>
    <h1>Promoter Prediction Web Server</h1>
    <p>An AI-powered web server for predicting bacterial promoter sequences and associated sigma factors from DNA using calibrated machine learning and deep learning.</p>
    <div class="hero-tags" aria-label="Server capabilities"><span class="hero-tag">81 bp input</span><span class="hero-tag">Batch FASTA analysis</span><span class="hero-tag">Calibrated confidence</span><span class="hero-tag">Uncertainty-aware</span></div>
  </div>
</header>
"""

FOOTER_HTML = f"""
<footer class="footer">
  <div class="footer-inner">
    <div><strong>BayesSigma</strong><br><span>Computational Biology Research Web Server</span></div>
    <div class="footer-meta">Developer: MMohsin · Institution: Independent Computational Biology Research Project<br>Contact: <a href="mailto:mmohsinfarooqhussain@gmail.com">mmohsinfarooqhussain@gmail.com</a> · <a href="https://github.com/MOHSIN184/SigmaBayes_Model" target="_blank" rel="noopener noreferrer">GitHub</a> · Version {APP_VERSION}<br>© 2026 BayesSigma. All rights reserved.</div>
  </div>
</footer>
"""


with gr.Blocks(title="BayesSigma | Promoter Prediction", css=CSS) as demo:
    gr.HTML(HEADER_HTML)
    with gr.Column(elem_classes="page-shell workspace"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="app-card"):
                gr.HTML(
                    '<div class="section-title"><h2>Sequence analysis</h2><p>Paste one sequence per line or upload a multi-record FASTA file. Each sequence must contain exactly 81 DNA bases.</p></div>'
                )
                sequence_input = gr.Textbox(
                    label="DNA sequence input",
                    lines=7,
                    placeholder="Paste an 81 bp DNA sequence, multiple sequences (one per line), or FASTA-formatted text…",
                    elem_classes="input-panel",
                )
                file_input = gr.File(
                    label="Upload FASTA or text file",
                    file_types=[".fasta", ".fa", ".txt"],
                    type="filepath",
                    elem_classes="upload-panel",
                )
                with gr.Row(elem_classes="task-options"):
                    binary_input = gr.Checkbox(
                        value=True, label="Promoter prediction"
                    )
                    sigma_input = gr.Checkbox(
                        value=True, label="Sigma-factor prediction"
                    )
                with gr.Row():
                    submit = gr.Button(
                        "Predict sequences", variant="primary", elem_classes="primary-button"
                    )
                    reset = gr.Button("Reset", elem_classes="secondary-button")
                status_output = gr.HTML(
                    _alert("Ready for sequence input.", "info")
                )
            with gr.Column(scale=2, elem_classes="app-card"):
                gr.HTML(
                    f"""<div class="section-title"><h2>Example input</h2><p>Load a verified input to explore the server workflow.</p></div>
                    <div class="example-box"><strong>Example Input</strong><code>{EXAMPLE_SEQUENCE}</code><span class="example-meta">Exactly 81 bp · A/C/G/T alphabet</span></div>"""
                )
                example_button = gr.Button(
                    "Use Example", elem_classes="secondary-button"
                )
                gr.HTML(
                    """<div class="example-box"><strong>FASTA format</strong><code>&gt;Sequence_1<br>ACGT…<br>&gt;Sequence_2<br>TGCA…</code><span class="example-meta">Headers are ignored during validation. Up to 5,000 records / 10 MB.</span></div>"""
                )

        with gr.Column(elem_classes="app-card results-wrap"):
            gr.HTML(
                '<div class="section-title"><h2>Prediction results</h2><p>Review class assignments, confidence, sigma-factor output, and per-record validation status.</p></div>'
            )
            summary_output = gr.HTML("")
            results_table = gr.Dataframe(
                headers=RESULT_HEADERS,
                value=_empty_results(),
                datatype=["str", "number", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Batch prediction results",
                elem_classes="dataframe",
            )
            with gr.Accordion("Detailed model output", open=False):
                details_output = gr.JSON(label="Probabilities and uncertainty")

        gr.HTML(
            """<div class="info-grid">
              <section class="content-card"><h2>Citation</h2><p>If you use this web server in your research, please cite our publication.</p><div class="citation"><strong>BayesSigma Research Team.</strong> BayesSigma: reliability-aware promoter and sigma-factor prediction using calibrated machine learning. <em>Journal reference pending.</em><br><strong>DOI:</strong> 10.0000/bayessigma.placeholder</div></section>
              <section class="content-card"><h2>Useful Resources</h2><p>Trusted genomics and regulatory-sequence databases.</p><div class="resource-grid">
                <a class="resource-link" href="https://www.ncbi.nlm.nih.gov/" target="_blank" rel="noopener noreferrer">NCBI <span>↗</span></a>
                <a class="resource-link" href="https://www.ensembl.org/" target="_blank" rel="noopener noreferrer">Ensembl <span>↗</span></a>
                <a class="resource-link" href="http://www.softberry.com/berry.phtml?topic=plantprom&amp;group=data&amp;subgroup=plantprom" target="_blank" rel="noopener noreferrer">PlantProm DB <span>↗</span></a>
                <a class="resource-link" href="https://jaspar.elixir.no/" target="_blank" rel="noopener noreferrer">JASPAR <span>↗</span></a>
                <a class="resource-link" href="https://epd.expasy.org/epd/" target="_blank" rel="noopener noreferrer">Promoter Database <span>↗</span></a>
              </div></section>
            </div>"""
        )

    gr.HTML(FOOTER_HTML)

    submit.click(
        fn=run_batch_prediction,
        inputs=[sequence_input, file_input, binary_input, sigma_input],
        outputs=[status_output, summary_output, results_table, details_output],
        api_name=False,
        scroll_to_output=True,
        show_progress="full",
        concurrency_limit=1,
    )
    example_button.click(
        fn=use_example,
        outputs=[
            sequence_input,
            file_input,
            status_output,
            summary_output,
            results_table,
            details_output,
        ],
        api_name=False,
    )
    reset.click(
        fn=reset_interface,
        outputs=[
            sequence_input,
            file_input,
            status_output,
            summary_output,
            results_table,
            details_output,
        ],
        api_name=False,
    )

    # Hidden compatibility endpoint: existing API clients retain the original inputs/output.
    with gr.Group(visible=False):
        legacy_sequence = gr.Textbox()
        legacy_binary = gr.Checkbox(value=True)
        legacy_sigma = gr.Checkbox(value=True)
        legacy_output = gr.JSON()
        legacy_submit = gr.Button()
    legacy_submit.click(
        fn=predict_sequence,
        inputs=[legacy_sequence, legacy_binary, legacy_sigma],
        outputs=legacy_output,
        api_name="predict",
    )

demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
