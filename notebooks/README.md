# Notebooks

Папка `notebooks/` используется для Colab-оркестрации, а не для реализации алгоритмов.

Основная логика проекта должна оставаться в:

```text
src/fedrecon/
scripts/
configs/
```

Рекомендуемая структура ноутбуков:

```text
00_colab_setup.ipynb
01_run_experiments.ipynb
02_analyze_results.ipynb
03_export_final_artifacts.ipynb
```

## 00_colab_setup.ipynb

Назначение:

- mount Google Drive;
- clone/pull GitHub repository;
- install requirements;
- restore results from Google Drive backup.

Ключевые команды:

```bash
git pull
pip install -r requirements.txt
rsync -av "/content/drive/MyDrive/fedrecon-course-backups/results_latest/" "/content/fedrecon-course/results/"
```

## 01_run_experiments.ipynb

Назначение:

- запуск ключевых экспериментов через CLI scripts.

Примеры:

```bash
python scripts/run_sweep.py --config configs/sweep_fedrecon_strong_100r_v1.yaml
python scripts/run_fedavg_sweep.py --config configs/sweep_fedavg_global_100r_v1.yaml
python scripts/run_centralized_mf.py --config configs/centralized_mf_baseline.yaml
```

## 02_analyze_results.ipynb

Назначение:

- пересборка summary-таблиц;
- построение финальных графиков.

Примеры:

```bash
python scripts/summarize_final_metrics.py ...
python scripts/make_summary_tradeoff_plot.py ...
python scripts/make_group_metric_plot.py ...
```

## 03_export_final_artifacts.ipynb

Назначение:

- сбор `results/final/`;
- backup final artifacts to Google Drive.
