# BayesSigma: Trustworthy Promoter Classification via Calibrated Deep Learning and Conformal Prediction

BayesSigma classifies bacterial promoters and their sigma-factor type — but its main contribution isn't the classifier itself, it's making the classifier's confidence trustworthy. Alongside a predicted label, it reports a **calibrated probability**, an **uncertainty estimate**, and a **conformal prediction set**, so the output can be safely used in risk-aware genomics workflows instead of being treated as a single overconfident guess.

## Why This Matters

A confident wrong prediction is worse than an honest "I'm not sure." Most promoter classifiers report only a predicted label and an accuracy/F1 score. BayesSigma instead asks, for every prediction:

- How confident is the model, really?
- Is that confidence calibrated (does 90% confidence mean ~90% correct)?
- How uncertain is this specific prediction?
- Given a target reliability level, which labels should still be considered plausible?

## Results at a Glance

| Task | Accuracy | Macro F1 | MCC | AUROC |
|---|---:|---:|---:|---:|
| Binary (Promoter / Non-Promoter) | 0.799 | 0.798 | 0.597 | 0.864 |
| Sigma-Factor (6-class) | 0.453 | 0.323 | 0.191 | 0.708 (macro OVR) |

**Bottom line:** binary promoter detection is reliable and well-calibrated. Sigma-factor classification is genuinely hard — confirmed not just by a low macro F1, but by large conformal prediction sets and high uncertainty estimates across folds. This contrast, and the diagnostics behind it, is the actual finding of this project.

## Scope

**In scope:**
- Binary promoter classification (Non-Promoter vs. Promoter)
- Sigma-factor multi-class classification (Sigma24, Sigma28, Sigma32, Sigma38, Sigma54, Sigma70)
- MC Dropout uncertainty estimation
- Temperature-scaling probability calibration
- Split conformal prediction

**Out of scope:**
- Cross-species promoter classification
- Graph neural network modeling
- Wet-lab / experimental validation

## Dataset

All sequences are fixed-length 81 bp DNA segments. No invalid characters were detected in any split.

**Binary dataset**

| Split | Total | Non-Promoter | Promoter |
|---|---:|---:|---:|
| Train | 5258 | 2695 | 2563 |
| Test | 1315 | 674 | 641 |

**Sigma-factor dataset**

| Split | Total | Sigma70 | Sigma24 | Sigma32 | Sigma38 | Sigma28 | Sigma54 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2315 | 1358 | 388 | 234 | 150 | 108 | 77 |
| Test | 583 | 340 | 98 | 59 | 38 | 28 | 20 |

Note the sigma dataset is heavily imbalanced — Sigma70 has ~18x more training examples than Sigma54.

## Methodology

- One-hot DNA encoding (A, C, G, T channels)
- 1D CNN classifier for both binary and sigma-factor tasks
- Class-weighted cross-entropy to address sigma-factor class imbalance
- **MC Dropout** — repeated stochastic forward passes to estimate prediction uncertainty
- **Temperature scaling** — post-hoc calibration so predicted confidence matches empirical accuracy
- **Split conformal prediction** — produces a set of plausible labels guaranteed to contain the true label at a chosen confidence level (e.g., 90%), instead of one possibly-overconfident label
- Evaluated with Expected Calibration Error (ECE), Brier Score, and reliability diagrams

**Metric glossary:**

| Term | Meaning |
|---|---|
| MCC | Matthews Correlation Coefficient — balanced accuracy measure, robust to class imbalance |
| AUROC | Area under the ROC curve — ranking quality across thresholds |
| AUPRC | Area under the precision-recall curve — more informative than AUROC under class imbalance |
| OVR | One-vs-Rest — each class scored against all others combined |
| ECE | Expected Calibration Error — gap between predicted confidence and actual correctness |
| Conformal coverage | Fraction of test cases where the true label was inside the predicted set |

## Results

### Binary Classification

- Accuracy: 0.7985
- Macro F1: 0.7980
- MCC: 0.5969
- AUROC: 0.8640
- AUPRC: 0.8558
- Conformal coverage: 0.8905 (target 90%) · 0.9521 (target 95%)

### Sigma-Factor Classification — Final Model (`safe_f1_selected_sigma_cnn`)

- Accuracy: 0.4528
- Macro F1: 0.3228
- Weighted F1: 0.4692
- MCC: 0.1909
- AUROC (macro OVR): 0.7077
- AUPRC (macro OVR): 0.2973

### Sigma Model Comparison

| Experiment | Accuracy | Macro F1 | Weighted F1 | MCC |
|---|---:|---:|---:|---:|
| original_sigma_cnn | 0.4219 | 0.2688 | 0.4168 | 0.1359 |
| improved_sampler_sigma_cnn | 0.1406 | 0.1762 | 0.0656 | 0.1003 |
| **safe_f1_selected_sigma_cnn (final)** | **0.4528** | **0.3228** | **0.4692** | **0.1909** |

`improved_sampler_sigma_cnn` combined `WeightedRandomSampler` with class weights — this overcorrected the imbalance, damaged Sigma70 performance, and is kept here as a documented negative result rather than discarded.

## Key Findings

- Binary promoter classification is stable, reasonably accurate, and — after calibration — well-aligned between confidence and correctness.
- Sigma-factor classification is fundamentally harder: classes are imbalanced, sequence patterns overlap across sigma types, and this difficulty shows up consistently across accuracy, calibration, and conformal set size — not just one metric.
- Large conformal prediction sets for sigma-factor classification aren't a failure of the method; they're an honest signal that the model genuinely can't narrow down the label with confidence for many sequences.

## Limitations

- Single dataset family; no cross-species generalization tested
- No wet-lab / experimental validation
- Sigma-factor classes are imbalanced, with Sigma54 especially data-poor
- Calibration improves probability reliability but does not by itself fix class separability
- No motif-level or saliency-based interpretability analysis (planned future work)

## Repository Structure

```text
BAYESSIGMA/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_dataset_check.ipynb       # validates sequence length/integrity, class counts
│   ├── 02_binary_cnn.ipynb          # trains/evaluates the binary CNN classifier
│   ├── 03_mc_dropout.ipynb          # MC Dropout uncertainty estimation
│   ├── 04_calibration.ipynb         # temperature scaling, ECE, Brier Score, reliability diagrams
│   ├── 05_conformal_prediction.ipynb # split conformal prediction sets and coverage
│   └── 06_sigma_multiclass.ipynb    # sigma-factor multi-class experiments
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
├── scripts/
│   ├── run_binary_experiment.py
│   ├── run_dataset_check.py
│   ├── run_sigma_experiment.py
│   ├── run_sigma_improved_experiment.py
│   └── run_sigma_safe_experiment.py
├── src/
│   ├── calibration.py
│   ├── conformal.py
│   ├── data_loader.py
│   ├── encoding.py
│   ├── evaluate.py
│   ├── models.py
│   └── train.py
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Requires: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`, `tqdm`, `jupyter`

## Usage

Run from the project root, in order:

```bash
python scripts/run_dataset_check.py
python scripts/run_binary_experiment.py
python scripts/run_sigma_experiment.py
python scripts/run_sigma_improved_experiment.py
python scripts/run_sigma_safe_experiment.py
```

Each script writes its outputs to `results/` (figures, trained models, and metric tables).

## Summary

BayesSigma is not presented as a state-of-the-art promoter classifier. It is a reliability-focused evaluation framework: it delivers predicted labels alongside calibrated confidence, uncertainty estimates, and conformal prediction sets, so that promoter and sigma-factor predictions can be judged not just by whether they're right, but by how much they should be trusted.
