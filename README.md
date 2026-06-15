# BayesSigma: Trustworthy Promoter Classification via Calibrated Deep Learning and Conformal Prediction

BayesSigma is a promoter classification project focused on trustworthy prediction rather than accuracy alone. Many promoter classifiers report a predicted class and standard performance scores; BayesSigma additionally reports calibrated probabilities, uncertainty estimates, and conformal prediction sets so model outputs are easier to interpret in risk-aware biological workflows.

## Research Motivation

Promoter recognition is a core task in computational genomics, but prediction confidence matters as much as the predicted label. A high-confidence false prediction can be more damaging than an uncertain one. BayesSigma addresses this by combining deep learning classification with uncertainty quantification, probability calibration, and split conformal prediction.

The project asks not only "what class is this sequence?" but also:

- How confident is the model?
- Are the probabilities calibrated?
- Is the prediction uncertain?
- Which labels should remain plausible under a target coverage level?

## Scope

- Binary promoter classification: Non-Promoter vs Promoter
- Sigma-factor multi-class classification: Sigma24, Sigma28, Sigma32, Sigma38, Sigma54, Sigma70
- MC Dropout uncertainty estimation
- Temperature scaling calibration
- Split conformal prediction

## Explicit Non-Scope

- Not cross-species classification
- Not graph neural network modeling
- Not wet-lab validated

## Dataset Description

All sequences are 81 bp. No invalid DNA characters were detected.

### Binary Dataset

| Split | Total | Non-Promoter | Promoter |
|---|---:|---:|---:|
| Train | 5258 | 2695 | 2563 |
| Test | 1315 | 674 | 641 |

### Sigma Dataset

| Split | Total | Sigma70 | Sigma24 | Sigma32 | Sigma38 | Sigma28 | Sigma54 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2315 | 1358 | 388 | 234 | 150 | 108 | 77 |
| Test | 583 | 340 | 98 | 59 | 38 | 28 | 20 |

Sequence length:

- All sequences are 81 bp
- Invalid sequence count: 0

## Methodology

- One-hot DNA encoding with channels for A, C, G, and T
- 1D CNN classifier for binary and sigma-factor classification
- Class-weighted cross-entropy for imbalanced sigma-factor learning
- MC Dropout for epistemic-style uncertainty estimation
- Temperature scaling for post-hoc probability calibration
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Split conformal prediction for prediction sets at target confidence levels

## Folder Structure

```text
BAYESSIGMA/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_dataset_check.ipynb
│   ├── 02_binary_cnn.ipynb
│   ├── 03_mc_dropout.ipynb
│   ├── 04_calibration.ipynb
│   ├── 05_conformal_prediction.ipynb
│   └── 06_sigma_multiclass.ipynb
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

Create and activate a Python environment, then install:

```bash
pip install -r requirements.txt
```

Required packages:

- numpy
- pandas
- scikit-learn
- matplotlib
- torch
- tqdm
- jupyter

## How To Run

Run commands from the project root:

```bash
python scripts/run_dataset_check.py
python scripts/run_binary_experiment.py
python scripts/run_sigma_experiment.py
python scripts/run_sigma_improved_experiment.py
python scripts/run_sigma_safe_experiment.py
```

## Current Final Results

### Binary Classification

- Accuracy: 0.7985
- Macro F1: 0.7980
- MCC: 0.5969
- AUROC: 0.8640
- AUPRC: 0.8558
- 90% conformal coverage: 0.8905
- 95% conformal coverage: 0.9521

### Sigma Final Chosen Model

Final selected model: `safe_f1_selected_sigma_cnn`

- Accuracy: 0.4528
- Macro F1: 0.3228
- Weighted F1: 0.4692
- MCC: 0.1909
- AUROC macro OVR: 0.7077
- AUPRC macro OVR: 0.2973

### Sigma Comparison

| Experiment | Accuracy | Macro F1 | Weighted F1 | MCC |
|---|---:|---:|---:|---:|
| original_sigma_cnn | 0.4219 | 0.2688 | 0.4168 | 0.1359 |
| improved_sampler_sigma_cnn | 0.1406 | 0.1762 | 0.0656 | 0.1003 |
| safe_f1_selected_sigma_cnn | 0.4528 | 0.3228 | 0.4692 | 0.1909 |

## Interpretation

Binary promoter classification is acceptable for this project stage, with balanced accuracy and macro F1 near 0.80 and strong AUROC/AUPRC values.

Sigma-factor classification remains challenging. The classes are imbalanced, and promoter patterns across sigma factors can be biologically similar. The safe F1-selected CNN is the best sigma model among the tested variants. It improves macro F1 and overall accuracy compared with the original sigma CNN.

The WeightedRandomSampler experiment showed an important failure mode: combining aggressive sampling with class weights overcorrected the imbalance, harmed Sigma70, and reduced overall performance. This result is kept as evidence rather than hidden.

Conformal prediction provides useful coverage guarantees, but sigma-factor prediction sets are often large. This indicates real uncertainty in multi-class sigma assignment, especially for smaller or harder classes.

## Limitations

- Single dataset family
- No cross-species validation
- No wet-lab validation
- Sigma classes are imbalanced
- Small classes like Sigma54 have limited samples
- Calibration may not always improve all metrics

## Final Research Claim

BayesSigma is not just an accuracy-driven promoter classifier. It provides predicted labels, calibrated confidence, uncertainty estimates, and conformal prediction sets to support more trustworthy promoter and sigma-factor prediction.
