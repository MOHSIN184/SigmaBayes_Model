# Final Reviewer Risk Assessment

| reviewer_concern | severity | why_it_matters | mitigation |
|---|---|---|---|
| Sigma performance is weak | High | Macro F1 remains below a strong publication threshold. | Frame sigma as a failure-mode and reliability benchmark; add more sigma data. |
| No external validation | High | Generalization beyond the current dataset is unknown. | Add independent datasets before journal submission. |
| No wet-lab validation | Medium | Biological utility is not experimentally confirmed. | Avoid functional claims; propose wet-lab validation as future work. |
| CNN is not better than k-mer RF | Medium | Deep learning novelty may be questioned. | Present honest model selection rather than CNN superiority. |
| Hybrid model failed | Medium | Combining features did not improve results. | Report as a negative result that supports benchmark transparency. |
| Dataset may be too narrow | High | Models may learn dataset-specific signals. | Add broader species and promoter sources. |
| Conformal prediction sets are large | Medium | Large sets limit practical sigma-factor specificity. | Discuss this as evidence of uncertainty and sigma difficulty. |
| Calibration does not always improve every metric | Low | Reliability metrics can trade off differently. | Report all calibration metrics without overclaiming. |
| Class imbalance affects conclusions | High | Accuracy can hide minority-class failure. | Prioritize macro F1, per-class diagnostics, and imbalance analysis. |
| Novelty may be questioned | Medium | Benchmarking alone may be seen as incremental. | Emphasize reliability, conformal prediction, CV, and error analysis package. |
