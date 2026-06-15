# BayesSigma Biological and Error Analysis Report

## Summary of Current Evidence

- Binary classification is stable and strongest with the calibrated 5-mer Random Forest.
- Sigma classification remains weak.
- Cross-validation confirms sigma macro F1 remains below 0.50.
- CNN models are not consistently superior to classical k-mer models.
- The hybrid CNN+k-mer model did not improve performance.

## Biological Interpretation

Position-frequency and consensus summaries were generated for binary and sigma classes. The binary promoter/non-promoter comparison has a mean 5-mer cosine similarity of approximately 0.838. Sigma classes show high 3-mer profile overlap, with the maximum off-diagonal cosine similarity approximately 0.993. This supports the interpretation that sigma-factor classes are not cleanly separable using short local composition alone.

Class imbalance is substantial in the sigma dataset. Sigma70 is dominant, while Sigma54, Sigma28, and Sigma38 are small classes. This imbalance makes macro F1 a more honest endpoint than accuracy and helps explain why models can achieve acceptable raw accuracy while failing minority sigma classes.

The sigma diagnostic table identifies the hardest classes as: Sigma38 (F1=0.080), Sigma54 (F1=0.270), Sigma32 (F1=0.276). Low F1, high uncertainty, and large conformal set sizes indicate that reliability outputs are meaningful: the system tends to express uncertainty where sigma labels are biologically or statistically difficult.

## Confusion Patterns

Top sigma confusion pairs:

- Sigma70 -> Sigma70: 167 (49.1%)
- Sigma70 -> Sigma32: 86 (25.3%)
- Sigma70 -> Sigma24: 62 (18.2%)
- Sigma24 -> Sigma24: 37 (37.8%)
- Sigma32 -> Sigma32: 30 (50.8%)
- Sigma24 -> Sigma70: 29 (29.6%)
- Sigma24 -> Sigma32: 27 (27.6%)
- Sigma38 -> Sigma70: 19 (50.0%)

These errors suggest overlapping sequence patterns and minority-class fragility rather than a simple neural-network architecture problem.

## Publication Decision

BayesSigma is not recommended yet for a well-known performance-focused journal as a high-performance sigma-factor classifier. Binary promoter classification is comparatively stable, but sigma-factor classification remains too weak for strong performance claims.

The project may be suitable after reframing as a trustworthy benchmarking and reliability framework. The current evidence is strongest when presented as a comparative study of calibrated classical and neural models, uncertainty estimation, conformal prediction, and stability analysis.

## Minimum Improvements Before Submission

- Add external datasets if possible.
- Add more sigma promoter samples, especially Sigma54, Sigma28, and Sigma38.
- Add motif/logo interpretation with biological discussion.
- Add calibrated/conformal reliability comparison across best models.
- Add cross-validation results.
- Discuss class imbalance and overlapping sigma motifs.
- Avoid claiming a state-of-the-art classifier.

## Recommended Final Framing

BayesSigma should be presented as:

"A reliability-aware benchmark for bacterial promoter and sigma-factor prediction using calibrated machine learning, uncertainty estimation, and conformal prediction."

## Final Decision

Proceed only as a benchmark/reliability paper, not as a high-performance classifier paper.

Do not submit to a well-known journal until sigma performance or biological validation improves.
