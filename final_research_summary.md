# BayesSigma: Trustworthy Promoter Classification via Calibrated Deep Learning and Conformal Prediction

## Dataset Summary

Binary:

- Train: 5258
- Test: 1315
- Sequence length: 81 bp
- Invalid DNA characters: 0

Sigma:

- Train: 2315
- Test: 583
- Sequence length: 81 bp
- Invalid DNA characters: 0

## Final Binary Result

Final binary model: `binary_cnn`

- Accuracy: 0.7985
- Precision macro: 0.7992
- Recall macro: 0.7977
- Macro F1: 0.7980
- Weighted F1: 0.7982
- MCC: 0.5970
- AUROC: 0.8640
- AUPRC: 0.8558

## Final Sigma Result

Final sigma model: `safe_f1_selected_sigma_cnn`

- Accuracy: 0.4528
- Precision macro: 0.3585
- Recall macro: 0.3338
- Macro F1: 0.3228
- Weighted F1: 0.4692
- MCC: 0.1909
- AUROC macro OVR: 0.7077
- AUPRC macro OVR: 0.2973

## Sigma Experiment Comparison

- `original_sigma_cnn`: baseline class-weighted CNN.
- `improved_sampler_sigma_cnn`: weighted sampler plus class weights; this over-corrected imbalance and harmed Sigma70.
- `safe_f1_selected_sigma_cnn`: normal shuffled batches, class weights, lower learning rate, and model selection by validation macro F1.

The safe F1-selected sigma CNN is the final sigma model because it gives the best balanced sigma result among the completed sigma experiments.

## Main Research Interpretation

Binary promoter classification is acceptable. Sigma-factor classification is more difficult because classes are imbalanced and sigma-factor promoter patterns can be biologically similar. Calibration and conformal prediction provide reliability information beyond ordinary accuracy scores. The safe sigma model gives the best balanced sigma result. WeightedRandomSampler with class weights over-corrected imbalance and harmed Sigma70.

## Final Claim

BayesSigma is not only a classifier. It is a trustworthy prediction framework that outputs:

- class prediction
- calibrated confidence
- uncertainty estimates
- conformal prediction sets

## Limitations

- No cross-species validation
- No wet-lab validation
- Dataset-specific
- Sigma classes are imbalanced
- Small classes like Sigma54 remain difficult
