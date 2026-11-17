# Notebooks

The CLI scripts in `scripts/` run the experiments from YAML configurations in `configs/`. The published CSV tables and figures are available in [results/final](../results/final/README.md).

## Typical CLI workflow

From the project root:

```powershell
python scripts/run_experiment.py --config configs/debug.yaml
python scripts/run_sweep.py --config configs/sweep_fedrecon_strong_100r_v1.yaml
python scripts/run_fedavg_sweep.py --config configs/sweep_fedavg_global_100r_v1.yaml
python scripts/run_centralized_mf.py --config configs/centralized_mf_baseline.yaml
```

Summary tables and figures are generated with scripts from `scripts/` and selected final artifacts are copied to `results/final/`.
