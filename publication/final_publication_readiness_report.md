# Publication Readiness Report for BayesSigma

## 1. Executive Verdict

- Current work is not ready for a well-known performance-focused journal.
- It may be developed into a benchmark/reliability paper.
- The final submission should avoid claiming state-of-the-art performance.

## 2. Final Recommended Title

BayesSigma: A Reliability-Aware Benchmark for Bacterial Promoter and Sigma-Factor Prediction Using Calibrated Machine Learning and Conformal Prediction

## 3. Current Strongest Results

- Binary best model: calibrated 5-mer Random Forest.
- Binary calibrated 5-mer Random Forest macro F1: 0.808.
- Binary calibrated 5-mer Random Forest AUROC/AUPRC: 0.872 / 0.852.
- Sigma best macro-F1 model is split-dependent: safe CNN on the held-out reliability comparison, while 3-mer Linear SVM is slightly stronger in combined-data CV.
- Sigma safe CNN held-out macro F1: 0.323.
- Sigma calibrated 3-mer SVM held-out macro F1: 0.143.
- Cross-validation result: binary RF is stable with mean macro F1 0.813.
- Cross-validation result: sigma remains weak, with 3-mer SVM mean macro F1 0.288 and sigma CNN mean macro F1 0.262.

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
