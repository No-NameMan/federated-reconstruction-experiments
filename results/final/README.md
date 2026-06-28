# Final results

This directory contains selected final artifacts from the FedRecon course project. Raw experiment runs are not committed; only compact tables and figures intended for public inspection are stored here.

## Figures

| File | Description |
|---|---|
| `figures/fig_02_eval_reconstruction_budget.png` | Validation RMSE as a function of the number of local reconstruction steps used during evaluation. |
| `figures/fig_03_client_update_heatmap.png` | Validation RMSE heatmap for different client update steps and client learning rates. |
| `figures/fig_04_quantization_tradeoff.png` | Communication-quality trade-off for none, fp16, int8, int4 and sign compression. |
| `figures/fig_05_client_groups_reconstruction.png` | Test RMSE for user groups with different rating counts under different reconstruction budgets. |
| `figures/fig_06_dropout_stress.png` | Test RMSE under simulated client dropout. |
| `figures/fig_07_paperlike_instability_heatmap.png` | No-bias paper-like sanity check with different reconstruction and client update steps. |

## Tables

| File | Description |
|---|---|
| `tables/tab_01_baseline_comparison_100r_seed42.csv` | FedRecon, FedAvg-like and centralized MF comparison for seed 42. |
| `tables/tab_02_fedrecon_100r_aggregate.csv` | Multi-seed aggregate metrics for FedRecon. |
| `tables/tab_03_fedavg_100r_aggregate.csv` | Multi-seed aggregate metrics for FedAvg-like baseline. |
| `tables/tab_04_eval_reconstruction_budget.csv` | Evaluation reconstruction budget sweep. |
| `tables/tab_05_client_update_dynamics.csv` | Client update steps and learning rate sweep. |
| `tables/tab_06_quantization_tradeoff.csv` | Quantization and communication trade-off sweep. |
| `tables/tab_07_client_groups_strong.csv` | Client group metrics for the strong setting. |
| `tables/tab_08_client_group_reconstruction.csv` | Client group metrics under different reconstruction budgets. |
| `tables/tab_09_dropout_stress.csv` | Client dropout stress-test aggregate metrics. |
| `tables/tab_10_paperlike_safelr_steps.csv` | Paper-like no-bias 3x3 reconstruction/update step sweep. |

## Notes

The artifacts were selected for compact public presentation. To reproduce or regenerate them, run the corresponding sweeps and summary scripts from the project root.
