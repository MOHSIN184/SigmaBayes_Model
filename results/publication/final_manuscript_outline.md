# Final Manuscript Outline

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
