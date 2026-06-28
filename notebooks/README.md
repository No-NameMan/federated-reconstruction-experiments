# Notebooks

This project keeps the main implementation in `src/`, `scripts/` and `configs/`. Notebooks are optional and are intended only for exploration, Colab orchestration or compact result overviews.

Recommended notebook usage:

```text
notebooks/
  01_results_overview.ipynb   # optional: reads final CSV files and reproduces selected plots
```

The repository is fully usable without notebooks. Experiments are launched from CLI scripts, and final public artifacts are stored in `results/final/`.

## Typical CLI workflow

From the project root:

```powershell
python scripts/run_experiment.py --config configs/debug.yaml
python scripts/run_sweep.py --config configs/sweep_fedrecon_strong_100r_v1.yaml
python scripts/run_fedavg_sweep.py --config configs/sweep_fedavg_global_100r_v1.yaml
python scripts/run_centralized_mf.py --config configs/centralized_mf_baseline.yaml
```

Summary tables and figures are generated with scripts from `scripts/` and selected final artifacts are copied to `results/final/`.
