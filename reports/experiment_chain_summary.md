# FEDRECON — цепочка экспериментов

Этот документ фиксирует, зачем проводился каждый эксперимент, какой вопрос он закрывал и какой вывод дал.

## 1. Исходная цель

Цель проекта — реализовать и исследовать Federated Reconstruction (FEDRECON) для matrix factorization на MovieLens 1M в частично локальной федеративной постановке.

Изначальная идея включала:

- проверку reconstruction budget;
- проверку client update budget;
- квантование глобальных дельт;
- сравнение с baseline-ами;
- анализ sparse/dense clients;
- dropout / partial participation;
- sanity check относительно оригинальной статьи.

## 2. Debug run

**Зачем:** проверить, что PyTorch-реализация запускается end-to-end.

**Конфиг:** `configs/debug.yaml`.

**Результат:** validation RMSE снижался примерно с `2.43` до `2.00` за 5 раундов.

**Вывод:** базовый цикл FEDRECON работает: загрузка данных, support/query split, reconstruction, client update, aggregation, CSV-логирование.

## 3. Quick baseline

**Зачем:** проверить, что модель обучается на более осмысленном горизонте.

**Конфиг:** `configs/quick_colab_cpu.yaml`.

**Результат:** validation RMSE снижался примерно с `1.45` до `1.10` за 30 раундов.

**Вывод:** реализация обучается устойчиво; quick-режим подходит для диагностических sweep-ов.

## 4. Первый reconstruction sweep

**Зачем:** проверить влияние числа reconstruction steps.

**Постановка:** `reconstruction.steps ∈ {0, 1, 2, 5, 10}`.

**Проблема:** один и тот же параметр использовался и в training, и в evaluation.

**Вывод:** эксперимент показал важность reconstruction, но смешал train-time и eval-time эффекты. После этого параметры были разделены.

## 5. Разделение train-time и eval-time reconstruction

В проекте были разделены:

```text
reconstruction.steps             # budget во время training
evaluation.reconstruction_steps  # budget во время evaluation/inference
```

**Вывод:** это важная методологическая правка, потому что FEDRECON нужно анализировать отдельно как метод обучения глобальных параметров и как метод локальной адаптации на новых клиентах.

## 6. Training reconstruction steps при фиксированном eval budget

**Зачем:** проверить влияние именно training reconstruction budget.

**Постановка:**

```text
training reconstruction.steps ∈ {0, 1, 2, 5, 10}
evaluation.reconstruction_steps = 10
```

**Результат:** большее число train-time steps не улучшало качество; малые значения иногда были лучше.

**Интерпретация:** большой evaluation budget может компенсировать различия между глобальными моделями; bias terms дают сильный baseline.

## 7. Strict no-bias diagnostic

**Зачем:** проверить, не объясняется ли поведение dominance bias terms.

**Постановка:**

```yaml
use_user_bias: false
use_item_bias: false
use_global_bias: false
```

**Результат:** в quick-режиме no-bias модель почти не обучалась, RMSE оставался около `3.8`.

**Вывод:** для MovieLens 1M bias terms важны; strict no-bias без достаточного бюджета и настройки learning rates слишком жёсткий.

## 8. Evaluation reconstruction budget sweep

**Зачем:** проверить, сколько локальной адаптации нужно клиенту на evaluation/inference.

**Постановка:**

```text
training reconstruction.steps = 5
evaluation.reconstruction.steps ∈ {0, 1, 2, 5, 10}
```

**Результат:** увеличение eval reconstruction steps монотонно улучшало RMSE.

**Ключевой вывод:** основной практический выигрыш FEDRECON проявляется через локальную reconstruction на стороне клиента при evaluation/inference.

## 9. Client update steps / learning rate sweep

**Зачем:** исследовать `k_u`, число локальных шагов клиента по глобальным параметрам.

**Постановка:**

```text
client_update.steps ∈ {1, 2, 5, 10}
client_update.lr ∈ {0.01, 0.03, 0.05, 0.1, 0.2}
seeds = {42, 43, 44}
```

**Результат:** в исследованном диапазоне большее `k_u` и больший `client_update.lr` улучшали качество.

**Вывод:** `k_u` управляет trade-off между качеством/скоростью обучения и локальной вычислительной нагрузкой, но не увеличивает communication payload.

## 10. Quantization sweep

**Зачем:** проверить communication-quality trade-off.

**Постановка:**

```text
compression.method ∈ {none, fp16, int8, int4, sign}
seeds = {42, 43, 44}
```

**Результат:** `fp16`, `int8`, `int4` почти не ухудшали качество; `sign` давал заметную деградацию.

**Ключевой вывод:** int4-квантование глобальных дельт даёт сильное снижение communication cost почти без потери качества.

## 11. Centralized MF + RECONEVAL baseline

**Зачем:** получить внешний reference baseline.

**Результат:** centralized baseline существенно лучше FEDRECON/FedAvg-like.

**Интерпретация:** centralized MF не является равным federated competitor; это upper/reference baseline, потому что он имеет централизованный доступ к train ratings и обучается 20 эпох.

## 12. FedAvg-like baseline

**Зачем:** добавить федеративный baseline без явного support/query separation.

**Постановка:** local user parameters and global item parameters jointly updated on client data, only global delta sent to server.

**Результат:** на 100 раундах FEDRECON strong и FedAvg-like global-only оказались практически эквивалентны.

**Вывод:** в текущей MovieLens MF-постановке основную роль играет качество global item parameters и RECONEVAL reconstruction; explicit support/query separation не дала заметного выигрыша над FedAvg-like.

## 13. Client sparsity analysis

**Зачем:** проверить качество по клиентским группам с разным числом рейтингов.

**Группы:**

```text
<=50, 51–100, 101–200, 201–500, >500
```

**Результат:** качество улучшалось до группы `201–500`, но `>500` оказалось сложнее.

**Вывод:** dense users могут иметь более разнообразные предпочтения; фиксированный reconstruction budget не одинаково оптимален для всех групп.

## 14. Group-wise evaluation reconstruction budget

**Зачем:** проверить, кому сильнее помогает увеличение evaluation reconstruction budget.

**Результат:** всем группам помогает рост `eval_k` от 0 до 5/10; sparse users быстро насыщаются, dense users выигрывают от большего числа шагов, но остаются сложнее.

**Вывод:** оптимальный reconstruction budget зависит от плотности локальных данных клиента.

## 15. Dropout / partial participation

**Зачем:** проверить устойчивость к client dropout.

**Результат:** dropout до 50% почти не влиял на качество; stress-test показал небольшую деградацию только при очень малом effective number of clients per round.

**Вывод:** в MovieLens-симуляции FEDRECON устойчив к сильному dropout, особенно при достаточном числе sampled clients per round.

## 16. Paper-like sanity check

**Зачем:** проверить, что реализация ведёт себя разумно в условиях, более близких к оригинальной статье.

**Постановка:**

```text
no-bias MF
embedding_dim = 50
batch_size = 5
100 clients per round
ReconEval-style evaluation
```

**Результаты:**

- при агрессивных learning rates и увеличенных local steps модель уходила в NaN;
- при safer learning rates все 9 конфигураций 3×3 обучились;
- увеличение local steps улучшало качество;
- лучшая safer точка `reconstruction.steps=50`, `client_update.steps=50` дала test RMSE около `1.265`.

**Вывод:** paper-like no-bias режим требует совместной настройки local steps и learning rates. Это не полная репликация Table 1, но качественная динамика FEDRECON воспроизводится.
