# BayesSigma website

This directory is a dependency-free static research site compatible with GitHub Pages and direct local opening.

- `index.html`: overview and headline results
- `methodology.html`: data, models, and reliability workflow
- `results.html`: figures, metrics, calibration, conformal prediction, CV, and errors
- `reproducibility.html`: commands mapped to actual repository scripts
- `demo.html`: 81 bp client validation and a configurable API hook
- `assets/`: theme, JavaScript, selected figure copies, and source CSV copies

Run `python -m http.server 8000 --directory docs`, then visit `http://localhost:8000`, or open `docs/index.html`. `.nojekyll` disables Jekyll processing.

When experiments change, intentionally refresh copied assets from `results/`; the website never generates scientific values.
