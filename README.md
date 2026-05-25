# FEDRECON Course Project

PyTorch-проект для исследования Federated Reconstruction в задаче matrix factorization на MovieLens 1M.

## Цель

Реализовать лёгкий, воспроизводимый симулятор частично локального федеративного обучения:

- local parameters: user embedding, optional user bias;
- global parameters: item embeddings, item bias, global bias;
- training: FEDRECON;
- primary dataset: MovieLens 1M;
- primary metrics: RMSE, MAE, rounded rating accuracy, communication cost.

## Быстрый старт в Colab

```bash
!pip install -r requirements.txt
!python scripts/run_experiment.py --config configs/debug.yaml
```

## Текущий статус

Это стартовый MVP-шаблон. В нём уже есть:

- структура проекта;
- конфиги;
- загрузчик MovieLens 1M;
- клиентские датасеты;
- matrix factorization;
- базовые метрики;
- базовый FEDRECON round;
- CLI для запуска эксперимента.

Следующие шаги:

1. Проверить baseline run в Colab.
2. Ускорить client update, если deepcopy окажется bottleneck.
3. Добавить полноценные baselines.
4. Добавить sweeps и графики.
