# Final research thesis draft

## 1. Working title

Коммуникационно-эффективное частично локальное федеративное обучение для рекомендательных систем.

## 2. Object and subject

**Объект исследования:** федеративное обучение рекомендательных моделей.

**Предмет исследования:** частично локальное федеративное обучение на основе FEDRECON для matrix factorization, включая reconstruction budget, client update budget, communication compression, client heterogeneity and dropout.

## 3. Goal

Цель работы — реализовать воспроизводимый PyTorch-симулятор FEDRECON для MovieLens 1M и экспериментально исследовать, как качество модели зависит от локальной реконструкции, клиентских обновлений и коммуникационных ограничений.

## 4. Tasks

1. Изучить оригинальный алгоритм FEDRECON и постановку partially local federated learning.
2. Реализовать FEDRECON для matrix factorization на MovieLens 1M.
3. Провести baseline-сравнение с centralized MF и FedAvg-like federated baseline.
4. Исследовать влияние evaluation reconstruction budget.
5. Исследовать влияние client update budget.
6. Оценить влияние квантования глобальных дельт на качество и communication cost.
7. Проанализировать качество по группам клиентов с разным числом рейтингов.
8. Проверить устойчивость к client dropout.
9. Провести paper-like sanity check относительно no-bias постановки оригинальной статьи.

## 5. Novelty / contribution

Практическая новизна проекта состоит не в новом алгоритме вместо FEDRECON, а в систематическом экспериментальном анализе его поведения в расширенной PyTorch-симуляции:

- раздельный анализ train-time и eval-time reconstruction budget;
- исследование `client_update.steps × client_update.lr`;
- квантование глобальных дельт;
- анализ sparse/dense client groups;
- dropout stress-test;
- comparison with centralized and FedAvg-like baselines;
- paper-like sanity check.

## 6. Main findings

1. Evaluation-time local reconstruction является ключевым фактором качества.
2. Увеличение client update budget улучшает качество без увеличения communication payload, но повышает локальные вычисления.
3. Int4-квантование глобальных дельт почти не ухудшает качество при значительном снижении коммуникации.
4. FEDRECON и FedAvg-like global-only в нашей MF-постановке дают практически одинаковое качество на 100 раундах.
5. Centralized MF значительно лучше, но это reference baseline, а не равный federated competitor.
6. Оптимальный reconstruction budget зависит от плотности клиентских данных.
7. FEDRECON устойчив к client dropout в MovieLens-симуляции.
8. Paper-like no-bias режим чувствителен к learning rates and local-step budget; при безопасной настройке он обучается ожидаемо.

## 7. Main limitation

Работа не является полной репликацией Table 1 из оригинальной статьи. Основные отличия:

- bias-augmented model in main experiments;
- smaller computational budget;
- mostly ReconEval-style evaluation;
- no full StandardEval branch;
- no exhaustive 500-round hyperparameter grid.

## 8. Possible future work

- Реализовать full StandardEval/ReconstructionEval protocol from the paper.
- Добавить adaptive reconstruction steps based on local client data size.
- Проверить более крупные или более sparse datasets.
- Реализовать server optimizers such as Adagrad/Yogi.
- Оптимизировать симуляцию без deepcopy per client.
