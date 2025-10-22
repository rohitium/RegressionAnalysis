# Agents Guide

This project performs cross-validated regression analyses on HIV drug resistance mutation (DRM) data using the Stanford HIVDB GenoPheno datasets. The codebase is organised into focused modules so agents (or new contributors) can quickly locate functionality.

## Module Map

| Module | Responsibilities |
|--------|------------------|
| `src/constants.py` | Drug class definitions, dataset URLs, DRM keyword metadata. |
| `src/data_loading.py` | Downloads/reads HIVDB tab files and DRM CSV lists. |
| `src/data_processing.py` | Normalises DRM strings, validates tokens, and builds binary feature matrices from sequence columns. |
| `src/models.py` | Fits OLS and LASSO models and computes repeated-k-fold CV MSE scores (with coefficient dispersion estimates for LASSO). |
| `src/results.py` | Assembles per-class coefficient tables and writes CSV reports (including combined performance metrics). |
| `src/plots.py` | Produces coefficient bar plots and benchmark visualisations (MSE heatmaps, best-model coefficient panels). |
| `src/DRMcv.py` | Core workflows: primary `DRMcv` runner and the benchmarking driver that iterate through drug classes, manage filtering, aggregate outputs, and invoke plotting/saving helpers. |
| `src.main` | Command-line interface. Parses arguments, dispatches to `DRMcv` or the benchmark routine. |

## CLI Usage (`python -m src.main`)

```
python -m src.main [OPTIONS]
```

### Common options

- `--min_muts N` &mdash; Keep only mutations observed in more than `N` samples (default 5).
- `--nfold K` / `--nrep R` &mdash; Repeated k-fold CV configuration (defaults 5/10).
- `--no-lars` &mdash; Disable LASSO (OLS always runs).
- `--drm_lists PATH [PATH ...]` &mdash; Supply custom DRM CSVs or a directory containing `DRM_*.csv`.
- `--drugs DRUG ...` &mdash; Restrict analysis to specific drug codes.
- `--no-legend` &mdash; Omit legends from coefficient plots.
- `--benchmark` &mdash; Run the benchmark workflow (see below); ignores `--drm_lists`.

When no DRM lists are provided the pipeline loads the curated defaults from `inputs/default/`.

### Outputs

- `results/Regression_results_{CLASS}.csv` &mdash; Coefficient tables (OLS + optional LASSO).
- `results/Regression_results_performance.csv` &mdash; Aggregate CV metrics for each drug.
- `plots/coefficients_{DRUG}.png` &mdash; Coefficient bar plots.

If `--benchmark` is used, additional artefacts are written:

- `results/Regression_results_benchmarks.csv` &mdash; Long-format MSE summary of all scenarios.
- `plots/benchmarks_mse.png` &mdash; Heatmap comparisons across min-muts, DRM lists, and models.
- `plots/benchmarks_coefficients.png` &mdash; Coefficients from the best-performing combination.

## Benchmark Workflow (`--benchmark`)

Benchmarks sweep across:

- DRM sources: `inputs/default/` and `inputs/drm_with_scores/`
- `min_muts` values: 1, 5, 10, 20
- Models: OLS (always) and LASSO (unless `--no-lars`)

Each scenario runs the core `DRMcv` routine with plot/CSV generation disabled, captures CV metrics and coefficients, then aggregates/visualises the comparisons.

## Custom DRM Lists

CSV files must expose `Pos` (integer) and `Mut` (string) columns. Example:

```
Pos,Mut
41,L
184,V
```

File names should follow `DRM_{drug_or_class}.csv` (e.g., `DRM_3TC.csv`, `DRM_NRTI.csv`), allowing drug-specific lists to override class-level definitions when both exist.

## Implementation Notes

- Mutation strings support mixtures and insertion/deletion markers. Lowercase `ins`/`del` sequences become the canonical `#`/`~` tokens, whereas uppercase `INS`/`DEL` are treated as literal residue mixtures. Mixture handling can be capped via the `--mixtures` option (number of residues allowed in a mixture).
- LASSO standard errors are derived via repeated k-fold resampling, aligning the error bars with the CV evaluation scheme.
- The core workflow guards against insufficient samples/features and logs informative messages when drugs are skipped.
- Paper mode harmonises mutation coverage per class (filling absent coefficients with zeros) and enforces shared y-axis limits across drugs for consistent visuals.

## Suggested Entry Points

1. `src.main` &mdash; For extending CLI options or integrating with other tooling.
2. `DRMcv.py::DRMcv` &mdash; To embed the analysis inside another driver or notebook.
3. `DRMcv.py::run_benchmark` &mdash; For automated comparisons between DRM list sets.
