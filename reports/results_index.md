# Results Index

Этот документ перечисляет итоговые таблицы и графики, которые следует использовать в курсовой и презентации.

## 1. Baseline comparison

**Таблицы:**

```text
results/tables/final_baseline_comparison_100r_seed42.csv
results/tables/final_baseline_comparison_with_fedavg.csv
results/tables/sweep_fedrecon_strong_100r_v1_final_metrics_aggregate.csv
results/tables/sweep_fedavg_global_100r_v1_final_metrics_aggregate.csv
```

**Смысл:**

- centralized MF + RECONEVAL — reference/upper baseline;
- FEDRECON strong и FedAvg-like — federated baselines;
- на 100 rounds FEDRECON и FedAvg-like дают почти одинаковое качество.

## 2. Evaluation reconstruction budget

**Таблицы:**

```text
results/tables/sweep_eval_reconstruction_steps_quick_v1_summary.csv
results/tables/sweep_eval_reconstruction_steps_strong_update_v1_summary.csv
```

**Графики:**

```text
results/figures/sweep_eval_reconstruction_steps_quick_v1_rmse.png
results/figures/sweep_eval_reconstruction_steps_strong_update_v1_final_val_rmse.png
```

**Вывод:**

Evaluation/inference reconstruction является ключевым фактором качества FEDRECON. Несколько локальных шагов дают значительный прирост, после чего наблюдается насыщение.

## 3. Client update budget

**Таблицы:**

```text
results/tables/sweep_client_update_steps_lr_eval10_medium_v1_dynamics.csv
```

**График:**

```text
results/figures/sweep_client_update_steps_lr_eval10_medium_v1_heatmap.png
```

**Вывод:**

Большее число client update steps и больший client update learning rate улучшали качество в исследованном диапазоне. Это влияет на локальный compute, но не на размер одной клиентской дельты.

## 4. Quantization / communication efficiency

**Таблицы:**

```text
results/tables/sweep_quantization_eval10_strong_update_v1_summary.csv
```

**Графики:**

```text
results/figures/sweep_quantization_eval10_strong_update_v1_aggregate_tradeoff.png
results/figures/sweep_quantization_eval10_strong_update_v1_rmse_vs_mb.png
```

**Вывод:**

`fp16`, `int8`, `int4` почти не ухудшают качество. `int4` даёт сильное снижение communication cost при почти неизменном RMSE. `sign` существенно агрессивнее и заметно деградирует качество.

## 5. Client sparsity / heterogeneity

**Таблицы:**

```text
results/tables/strong_update_test_client_groups_pooled.csv
results/tables/sweep_eval_reconstruction_steps_strong_update_v1_test_groups.csv
```

**График:**

```text
results/figures/sweep_eval_reconstruction_steps_strong_update_v1_test_groups_pooled_rmse.png
```

**Вывод:**

Оптимальный reconstruction budget зависит от плотности локальных данных. Sparse users быстро достигают насыщения, dense users выигрывают от большего числа reconstruction steps, но остаются сложной группой.

## 6. Dropout robustness

**Таблицы:**

```text
results/tables/sweep_dropout_strong_update_v1_final_metrics_aggregate.csv
results/tables/sweep_dropout_clients_stress_v1_final_metrics_aggregate.csv
results/tables/sweep_dropout_clients_stress_v1_dynamics.csv
```

**Графики:**

```text
results/figures/sweep_dropout_strong_update_v1_rmse.png
results/figures/sweep_dropout_clients_stress_v1_test_rmse.png
```

**Вывод:**

FEDRECON устойчив к client dropout в MovieLens-симуляции. Существенная деградация появляется только при очень малом effective number of clients per round.

## 7. Paper-like sanity check

**Таблицы:**

```text
results/tables/sweep_fedrecon_paperlike_100r_v1_final_metrics.csv
results/tables/sweep_fedrecon_paperlike_100r_v1_dynamics.csv
results/tables/sweep_fedrecon_paperlike_steps_100r_3x3_v1_final_metrics.csv
results/tables/sweep_fedrecon_paperlike_steps_100r_3x3_safelr_v1_final_metrics.csv
results/tables/sweep_fedrecon_paperlike_steps_100r_3x3_safelr_v1_dynamics.csv
```

**Графики:**

```text
results/figures/sweep_fedrecon_paperlike_steps_100r_3x3_v1_val_rmse_heatmap.png
```

**Нужно построить дополнительно:**

```text
results/figures/sweep_fedrecon_paperlike_steps_100r_3x3_safelr_v1_val_rmse_heatmap.png
```

**Вывод:**

Paper-like no-bias FEDRECON обучается при достаточном local-step budget и безопасных learning rates. Агрессивные learning rates приводят к divergence/NaN.

## 8. Кандидаты на финальные рисунки

Минимальный набор для курсовой:

```text
fig_01_fedrecon_algorithm_diagram.png                 # нужно нарисовать
fig_02_eval_reconstruction_budget.png
fig_03_client_update_heatmap.png
fig_04_quantization_tradeoff.png
fig_05_client_groups_reconstruction.png
fig_06_dropout_stress.png
fig_07_paperlike_sanity_heatmap.png
```

## 9. Кандидаты на финальные таблицы

```text
tab_01_baseline_comparison.csv
tab_02_eval_reconstruction_budget.csv
tab_03_quantization_tradeoff.csv
tab_04_client_group_reconstruction.csv
tab_05_dropout_stress.csv
tab_06_paperlike_sanity_check.csv
```
