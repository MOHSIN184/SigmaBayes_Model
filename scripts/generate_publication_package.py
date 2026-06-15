from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
PUBLICATION_DIR = RESULTS_DIR / "publication"


INPUT_FILES = {
    "final_reliability_comparison": TABLE_DIR / "final_reliability_comparison.csv",
    "final_model_comparison_with_hybrid": TABLE_DIR
    / "final_model_comparison_with_hybrid.csv",
    "cv_summary_wide": TABLE_DIR / "cv_summary_wide.csv",
    "analysis_sigma_class_diagnostic": TABLE_DIR
    / "analysis_sigma_class_diagnostic.csv",
    "analysis_sigma_confusion_pairs": TABLE_DIR / "analysis_sigma_confusion_pairs.csv",
    "analysis_sigma_kmer_cosine_similarity": TABLE_DIR
    / "analysis_sigma_kmer_cosine_similarity.csv",
    "analysis_class_imbalance_summary": TABLE_DIR
    / "analysis_class_imbalance_summary.csv",
    "analysis_publication_decision_report": RESULTS_DIR
    / "analysis_publication_decision_report.md",
    "final_research_summary": RESULTS_DIR / "final_research_summary.md",
}


FINAL_VERDICT = (
    "Current results are not ready for a well-known performance-focused journal, "
    "but the project can be reframed as a reliability-aware benchmarking study."
)


def warn(message):
    print(f"Warning: {message}")


def read_inputs():
    loaded = {}
    for name, path in INPUT_FILES.items():
        if not path.exists():
            warn(f"Missing input file, skipping: {path}")
            loaded[name] = None
            continue
        if path.suffix.lower() == ".csv":
            loaded[name] = pd.read_csv(path)
        else:
            loaded[name] = path.read_text(encoding="utf-8")
    return loaded


def get_metric(df, experiment, metric):
    if df is None or "experiment" not in df.columns or metric not in df.columns:
        return None
    rows = df[df["experiment"] == experiment]
    if rows.empty:
        return None
    value = rows.iloc[0][metric]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_cv_metric(df, task, model, metric):
    column = f"{metric}_mean"
    if df is None or column not in df.columns:
        return None
    rows = df[(df["task"] == task) & (df["model"] == model)]
    if rows.empty:
        return None
    try:
        return float(rows.iloc[0][column])
    except (TypeError, ValueError):
        return None


def format_metric(value):
    return "not available" if value is None else f"{value:.3f}"


def write_readiness_report(inputs):
    reliability = inputs["final_reliability_comparison"]
    cv = inputs["cv_summary_wide"]
    binary_rf_f1 = get_metric(reliability, "binary_best_kmer_reliable", "f1_macro")
    binary_rf_auroc = get_metric(reliability, "binary_best_kmer_reliable", "auroc")
    binary_rf_auprc = get_metric(reliability, "binary_best_kmer_reliable", "auprc")
    sigma_cnn_f1 = get_metric(reliability, "sigma_safe_cnn", "f1_macro")
    sigma_kmer_f1 = get_metric(reliability, "sigma_best_kmer_reliable", "f1_macro")
    binary_rf_cv = get_cv_metric(cv, "binary", "5mer_random_forest", "f1_macro")
    sigma_svm_cv = get_cv_metric(cv, "sigma", "3mer_linear_svm", "f1_macro")
    sigma_cnn_cv = get_cv_metric(cv, "sigma", "sigma_cnn", "f1_macro")

    content = f"""# Publication Readiness Report for BayesSigma

## 1. Executive Verdict

- Current work is not ready for a well-known performance-focused journal.
- It may be developed into a benchmark/reliability paper.
- The final submission should avoid claiming state-of-the-art performance.

## 2. Final Recommended Title

BayesSigma: A Reliability-Aware Benchmark for Bacterial Promoter and Sigma-Factor Prediction Using Calibrated Machine Learning and Conformal Prediction

## 3. Current Strongest Results

- Binary best model: calibrated 5-mer Random Forest.
- Binary calibrated 5-mer Random Forest macro F1: {format_metric(binary_rf_f1)}.
- Binary calibrated 5-mer Random Forest AUROC/AUPRC: {format_metric(binary_rf_auroc)} / {format_metric(binary_rf_auprc)}.
- Sigma best macro-F1 model is split-dependent: safe CNN on the held-out reliability comparison, while 3-mer Linear SVM is slightly stronger in combined-data CV.
- Sigma safe CNN held-out macro F1: {format_metric(sigma_cnn_f1)}.
- Sigma calibrated 3-mer SVM held-out macro F1: {format_metric(sigma_kmer_f1)}.
- Cross-validation result: binary RF is stable with mean macro F1 {format_metric(binary_rf_cv)}.
- Cross-validation result: sigma remains weak, with 3-mer SVM mean macro F1 {format_metric(sigma_svm_cv)} and sigma CNN mean macro F1 {format_metric(sigma_cnn_cv)}.

## 4. Why Binary Result Is Acceptable

- Binary performance is stable under 5-fold cross-validation.
- The calibrated 5-mer Random Forest has useful AUROC/AUPRC performance.
- Calibration improved ECE for the k-mer Random Forest.
- Conformal prediction achieved target coverage on the binary endpoint.
- The strongest binary model is simple, interpretable, and CPU-friendly.

## 5. Why Sigma Result Is Weak

- Sigma macro F1 remains below 0.50.
- Sigma38 is a high-difficulty class with very low F1.
- Sigma54 has a small sample size.
- Conformal prediction sets are large for sigma classes.
- Sigma classes are imbalanced, with Sigma70 dominant.
- Several sigma classes show overlapping or confused sequence patterns.

## 6. What Makes the Project Still Valuable

- Reliability analysis is unusually explicit for this project type.
- The work compares calibration behavior across model families.
- Conformal prediction provides coverage-aware outputs instead of only point predictions.
- The project honestly compares CNN, k-mer, and hybrid models.
- The benchmark-style contribution identifies failure modes in sigma-factor prediction.
- The analysis supports a transparent model-selection framework rather than a single overclaimed classifier.

## 7. Publication Recommendation

Recommended action: Do not submit yet to a well-known journal as a high-performance classifier paper. Continue only if the manuscript is reframed as a reliability-aware benchmarking study or if additional external datasets and biological validation are added.
"""
    path = PUBLICATION_DIR / "final_publication_readiness_report.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_manuscript_outline():
    content = """# Final Manuscript Outline

## 1. Abstract

- State the promoter and sigma-factor prediction problem.
- Summarize the comparison between CNN, k-mer, hybrid, calibrated, and conformal models.
- Report that binary promoter classification is stable, while sigma-factor classification remains difficult.
- Emphasize the reliability-aware benchmarking contribution.

## 2. Introduction

- Introduce bacterial promoters and sigma-factor specificity.
- Explain why reliable prediction matters.
- Describe the risk of overclaiming from a single train/test split.
- Motivate calibration, uncertainty, conformal prediction, and cross-validation.

## 3. Related Work

- Summarize promoter prediction methods.
- Summarize sigma-factor classification methods.
- Discuss k-mer machine learning baselines.
- Discuss CNN/deep learning approaches.
- Discuss reliability, calibration, and conformal prediction in bioinformatics.

## 4. Materials and Methods

### Dataset

- Describe binary promoter/non-promoter data.
- Describe sigma-factor class labels and class imbalance.
- Explain original held-out test evaluation and separate combined-data CV.

### Models

- Explain evaluated model families and why each was included.

### k-mer Baselines

- Describe normalized 3-mer, 4-mer, and 5-mer features.
- Describe Logistic Regression, Linear SVM, and Random Forest.

### CNN Models

- Describe one-hot encoding and CNN architecture.
- Explain training, early stopping, and macro-F1 model selection.

### Hybrid Model

- Describe CNN sequence branch plus k-mer MLP branch.
- Report that it was tested but did not improve performance.

### Calibration

- Explain temperature scaling for CNN models.
- Explain probability calibration for classical models.
- Define ECE, Brier score, and NLL.

### Conformal Prediction

- Explain calibration split, qhat, coverage, and set size.
- Clarify that conformal prediction estimates reliability, not accuracy improvement.

### Cross-validation

- Describe combined-data 5-fold stratified CV.
- Explain that CV is separate from held-out test evaluation.

### Biological/Error Analysis

- Describe position-frequency summaries.
- Describe k-mer cosine similarity.
- Describe class diagnostics, confusion pairs, and uncertainty summaries.

## 5. Results

### Dataset Characteristics

- Show class counts and imbalance.
- Highlight Sigma70 dominance and small Sigma54/Sigma28/Sigma38 classes.

### Binary Classification Results

- Report that calibrated 5-mer Random Forest is strongest.
- Compare against CNN and hybrid model.

### Sigma Classification Results

- Report weak macro F1.
- Highlight difficult classes including Sigma38 and Sigma54.

### Model Comparison

- Compare CNN, k-mer, and hybrid models.
- Emphasize that CNN is not consistently superior.

### Calibration and Conformal Prediction

- Report reliability metrics and coverage.
- Discuss large sigma conformal sets.

### Cross-validation Stability

- Report mean +/- standard deviation across 5 folds.
- Explain stability of binary RF and weakness of sigma models.

### Biological/Error Analysis

- Discuss motif/k-mer overlap.
- Discuss confusion pairs and uncertainty by class.

## 6. Discussion

### Main Findings

- Binary endpoint is usable and stable.
- Sigma endpoint remains biologically and statistically difficult.

### Why k-mer RF Beats CNN in Binary Task

- Short motifs and local composition may be sufficient.
- Dataset size may favor classical features over deep models.

### Why Sigma Classification Is Hard

- Class imbalance.
- Overlapping promoter patterns.
- Small minority classes.
- Ambiguity in sigma-factor sequence signatures.

### Trustworthiness Contribution

- Honest benchmarking.
- Calibration.
- Conformal prediction.
- Error analysis.
- Stability analysis.

## 7. Limitations

- Dataset-specific results.
- No external or wet-lab validation.
- Weak sigma performance.
- Large conformal sets for sigma.

## 8. Future Work

- Add external datasets.
- Add more sigma promoter examples.
- Add biological motif interpretation.
- Validate high-confidence predictions experimentally.

## 9. Conclusion

- Present BayesSigma as a reliability-aware benchmark.
- Avoid state-of-the-art claims.
- Emphasize transparent model selection and failure-mode analysis.
"""
    path = PUBLICATION_DIR / "final_manuscript_outline.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_titles_and_claims():
    content = """# Final Recommended Title and Claims

## A. Recommended Title

BayesSigma: A Reliability-Aware Benchmark for Bacterial Promoter and Sigma-Factor Prediction Using Calibrated Machine Learning and Conformal Prediction

## B. Alternative Titles

- BayesSigma: Reliability-Aware Benchmarking of Promoter and Sigma-Factor Prediction Models
- Calibrated Machine Learning for Trustworthy Bacterial Promoter Prediction
- Reliability and Uncertainty Analysis of Bacterial Promoter and Sigma-Factor Classifiers

## C. Safe Claims

- We benchmark deep learning and k-mer machine learning models for bacterial promoter and sigma-factor classification.
- We show that calibrated k-mer Random Forest provides the strongest binary promoter performance.
- We show that sigma-factor classification remains challenging under class imbalance.
- We evaluate model reliability using calibration metrics and conformal prediction.
- We identify sigma classes with high uncertainty and poor separability.

## D. Claims to Avoid

- Do not claim state-of-the-art.
- Do not claim cross-species generalization.
- Do not claim wet-lab validation.
- Do not claim deep learning is superior.
- Do not claim sigma-factor classification is solved.
- Do not claim conformal prediction improves accuracy.
"""
    path = PUBLICATION_DIR / "final_recommended_title_and_claims.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_limitations_and_future_work():
    content = """# Final Limitations and Future Work

## Limitations

- Dataset-specific results.
- No external validation.
- No cross-species validation.
- No wet-lab validation.
- Sigma class imbalance.
- Small Sigma54 sample size.
- Weak Sigma38 performance.
- Models may learn dataset-specific k-mer patterns.
- Conformal sets are large for sigma task.

## Future Work

- Add external promoter datasets.
- Add species-level validation.
- Add more sigma-factor promoter samples.
- Add motif/logo analysis with biological interpretation.
- Add transformer embeddings only after stronger data validation.
- Add experimental validation for high-confidence predictions.
- Improve class imbalance handling carefully.
- Evaluate reliability on independent datasets.
"""
    path = PUBLICATION_DIR / "final_limitations_and_future_work.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_reviewer_risk_assessment():
    rows = [
        {
            "reviewer_concern": "Sigma performance is weak",
            "severity": "High",
            "why_it_matters": "Macro F1 remains below a strong publication threshold.",
            "mitigation": "Frame sigma as a failure-mode and reliability benchmark; add more sigma data.",
        },
        {
            "reviewer_concern": "No external validation",
            "severity": "High",
            "why_it_matters": "Generalization beyond the current dataset is unknown.",
            "mitigation": "Add independent datasets before journal submission.",
        },
        {
            "reviewer_concern": "No wet-lab validation",
            "severity": "Medium",
            "why_it_matters": "Biological utility is not experimentally confirmed.",
            "mitigation": "Avoid functional claims; propose wet-lab validation as future work.",
        },
        {
            "reviewer_concern": "CNN is not better than k-mer RF",
            "severity": "Medium",
            "why_it_matters": "Deep learning novelty may be questioned.",
            "mitigation": "Present honest model selection rather than CNN superiority.",
        },
        {
            "reviewer_concern": "Hybrid model failed",
            "severity": "Medium",
            "why_it_matters": "Combining features did not improve results.",
            "mitigation": "Report as a negative result that supports benchmark transparency.",
        },
        {
            "reviewer_concern": "Dataset may be too narrow",
            "severity": "High",
            "why_it_matters": "Models may learn dataset-specific signals.",
            "mitigation": "Add broader species and promoter sources.",
        },
        {
            "reviewer_concern": "Conformal prediction sets are large",
            "severity": "Medium",
            "why_it_matters": "Large sets limit practical sigma-factor specificity.",
            "mitigation": "Discuss this as evidence of uncertainty and sigma difficulty.",
        },
        {
            "reviewer_concern": "Calibration does not always improve every metric",
            "severity": "Low",
            "why_it_matters": "Reliability metrics can trade off differently.",
            "mitigation": "Report all calibration metrics without overclaiming.",
        },
        {
            "reviewer_concern": "Class imbalance affects conclusions",
            "severity": "High",
            "why_it_matters": "Accuracy can hide minority-class failure.",
            "mitigation": "Prioritize macro F1, per-class diagnostics, and imbalance analysis.",
        },
        {
            "reviewer_concern": "Novelty may be questioned",
            "severity": "Medium",
            "why_it_matters": "Benchmarking alone may be seen as incremental.",
            "mitigation": "Emphasize reliability, conformal prediction, CV, and error analysis package.",
        },
    ]
    df = pd.DataFrame(rows)
    headers = ["reviewer_concern", "severity", "why_it_matters", "mitigation"]
    markdown_lines = [
        "# Final Reviewer Risk Assessment",
        "",
        "| reviewer_concern | severity | why_it_matters | mitigation |",
        "|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        markdown_lines.append(
            "| "
            + " | ".join(str(row[header]).replace("|", "/") for header in headers)
            + " |"
        )
    markdown = "\n".join(markdown_lines) + "\n"
    path = PUBLICATION_DIR / "final_reviewer_risk_assessment.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def write_decision_summary():
    rows = [
        {
            "item": "Binary endpoint",
            "decision": "Usable",
            "evidence": "Calibrated 5-mer Random Forest is strongest and stable in CV.",
            "recommendation": "Use k-mer RF as the binary endpoint.",
        },
        {
            "item": "Sigma endpoint",
            "decision": "Weak/modest",
            "evidence": "Macro F1 remains below 0.50 and minority classes are difficult.",
            "recommendation": "Do not present as a solved classifier.",
        },
        {
            "item": "Deep learning novelty",
            "decision": "Limited",
            "evidence": "CNN is not consistently better than classical k-mer models.",
            "recommendation": "Avoid deep-learning superiority claims.",
        },
        {
            "item": "Calibration contribution",
            "decision": "Valuable",
            "evidence": "Reliability metrics quantify probability quality across models.",
            "recommendation": "Keep as a central benchmark component.",
        },
        {
            "item": "Conformal contribution",
            "decision": "Valuable but cautious",
            "evidence": "Coverage is measurable, but sigma prediction sets are large.",
            "recommendation": "Frame as uncertainty quantification, not accuracy improvement.",
        },
        {
            "item": "Journal readiness",
            "decision": "Not ready for performance-focused journal",
            "evidence": "Sigma performance and validation are insufficient.",
            "recommendation": "Add external/biological validation before submission.",
        },
        {
            "item": "Best framing",
            "decision": "Reliability-aware benchmark",
            "evidence": "Strongest contribution is calibrated comparison and failure analysis.",
            "recommendation": "Reframe manuscript around trustworthy benchmarking.",
        },
        {
            "item": "Required next improvement",
            "decision": "External validation and more sigma data",
            "evidence": "Current sigma classes are imbalanced and overlapping.",
            "recommendation": "Prioritize additional datasets and biological interpretation.",
        },
    ]
    df = pd.DataFrame(rows)
    path = PUBLICATION_DIR / "final_decision_summary.csv"
    df.to_csv(path, index=False)
    return path, df


def main():
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()

    created_paths = [
        write_readiness_report(inputs),
        write_manuscript_outline(),
        write_titles_and_claims(),
        write_limitations_and_future_work(),
        write_reviewer_risk_assessment(),
    ]
    decision_path, decision_df = write_decision_summary()
    created_paths.append(decision_path)

    print("\nCreated publication package file paths")
    for path in created_paths:
        print(f"  {path}")

    print("\nFinal decision summary")
    print(decision_df.to_string(index=False))

    print("\nOne-line final verdict")
    print(FINAL_VERDICT)
    print("\nPublication package generated successfully.")


if __name__ == "__main__":
    sys.exit(main())
