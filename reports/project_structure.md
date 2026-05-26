# Project structure after refactor

The project uses `scripts/` only as thin CLI entrypoints. Reusable logic lives under `src/fedrecon/`.

## Main areas

```text
src/fedrecon/
  data/          Dataset loading, client datasets, support/query splitting, sampling.
  models/        Matrix factorization models.
  algorithms/    FEDRECON round logic, reconstruction, client update, aggregation, compression.
  baselines/     Baseline-specific reusable utilities.
  simulation/    Training loops and sweep execution.
  analysis/      Result summaries, plots, heatmaps, trade-off visualizations.
  utils/         Configs, seeding, paths, device helpers.
```

## CLI scripts

The files in `scripts/` should not contain experiment logic. They should parse command-line arguments and call functions from `src/fedrecon/`.

Examples:

```bash
python scripts/run_experiment.py --config configs/quick_colab_cpu.yaml
python scripts/run_sweep.py --config configs/sweep_quantization_eval10_strong_update_v1.yaml
python scripts/run_centralized_mf.py --config configs/centralized_mf_baseline.yaml
```

## Centralized baseline placement

Centralized MF is split across:

```text
src/fedrecon/models/centralized_matrix_factorization.py
src/fedrecon/baselines/centralized_mf.py
src/fedrecon/simulation/centralized_trainer.py
scripts/run_centralized_mf.py
```

This keeps the model, baseline utilities, training loop, and CLI entrypoint separate.

## Analysis placement

Reusable analysis logic is under `src/fedrecon/analysis/`:

```text
summarize_runs.py
summarize_client_groups.py
sweep_plot.py
sweep_heatmap.py
tradeoff.py
summary_tradeoff.py
```

The corresponding scripts in `scripts/` are wrappers for Colab/terminal convenience.
