# BayesSigma

BayesSigma is a reliability-aware benchmark for bacterial promoter recognition and six-class sigma-factor classification on 81 bp DNA sequences. It compares one-hot CNN, k-mer, and hybrid CNN+k-mer models, then evaluates calibration, Monte Carlo dropout uncertainty, split conformal prediction, five-fold stability, and biological error patterns.

## Recorded results

The held-out binary CNN reaches accuracy 0.7985, macro F1 0.7980, and AUROC 0.8640. The selected safe sigma CNN reaches accuracy 0.4528, macro F1 0.3228, and macro one-vs-rest AUROC 0.7077. The project reports weaker sigma performance and large conformal sets openly.

## Repository structure

- `data/`: raw and CD-HIT-processed FASTA datasets
- `src/`: loading, encoding, models, training, evaluation, and reliability
- `scripts/`: experiment, CV, analysis, and publication workflows
- `notebooks/`: staged experimental notebooks
- `results/`: model state dictionaries, figures, tables, and summaries
- `docs/`: static GitHub Pages research website

## Set up and run

```bash
python -m venv .venv
# activate the environment for your shell
pip install -r requirements.txt
python scripts/run_dataset_check.py
python scripts/run_binary_experiment.py
python scripts/run_sigma_safe_experiment.py
```

Additional workflows include `run_kmer_baselines.py`, `run_kmer_reliability.py`, `run_hybrid_experiment.py`, `run_cross_validation.py`, `run_biological_error_analysis.py`, `generate_publication_package.py`, and `final_project_check.py`. See [the reproducibility page](docs/reproducibility.html) for commands.

## Website and deployment

Open `docs/index.html`, or run `python -m http.server 8000 --directory docs` and visit `http://localhost:8000`. To deploy, push the repository, choose **GitHub Actions** under **Settings → Pages**, and run the included workflow. The static demo validates sequences but makes no fake predictions.

## Scope and limitations

The study is dataset-specific, has no cross-species or wet-lab validation, and contains severe sigma-class imbalance. Requirements are unpinned. PyTorch model state dictionaries exist, but calibration and conformal outputs are not bundled as versioned serving artifacts, so no inference server is enabled by default.

## Citation

> TODO: Add authors, manuscript title, venue or preprint, year, DOI, and BibTeX after publication metadata are finalized.
