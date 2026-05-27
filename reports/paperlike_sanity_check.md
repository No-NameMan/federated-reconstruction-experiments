# Paper-like sanity check

## 1. Зачем нужен этот блок

Основная экспериментальная часть проекта использует расширенную MF-модель с bias terms:

```text
rating_hat = dot(user_embedding, item_embedding) + user_bias + item_bias + global_bias
```

Оригинальная статья и TFF tutorial для MovieLens используют более простую no-bias matrix factorization постановку, где локальный user embedding и глобальный item embedding взаимодействуют через dot product.

Поэтому результаты основной постановки нельзя напрямую сравнивать с Table 1 из статьи. Нужен отдельный sanity check в условиях, более близких к paper-like no-bias regime.

## 2. Отличия наших основных экспериментов от статьи

Основная постановка проекта отличается от оригинальной статьи:

| Aspect | Original paper-like setup | Main project setup |
|---|---|---|
| Model | no-bias dot product MF | MF with user/item/global bias |
| Embedding dim | 50 | 32 in main runs |
| Rounds | up to 500 | mostly 30/100 |
| Clients per round | 100 | often 25 |
| Local batch size | 5 | full-batch originally; minibatch added later |
| Evaluation | StandardEval and ReconEval variants | mostly ReconEval-style held-out users |
| Goal | reproduce benchmark | analyze practical constraints and modifications |

## 3. Conducted sanity checks

### 3.1 Initial no-bias diagnostic

Strict no-bias model in quick regime did not learn well: RMSE stayed around `3.8`.

This indicated that no-bias MovieLens MF is much more sensitive to optimization budget and scale than the bias-augmented model.

### 3.2 Paper-like LR sweep

The model was changed toward paper-like conditions:

```text
embedding_dim = 50
use_user_bias = false
use_item_bias = false
use_global_bias = false
clients_per_round = 100
batch_size = 5
rounds = 100
```

A learning-rate sweep showed that the no-bias model can learn when learning rates are chosen appropriately. The best 10×10 setting achieved test RMSE around `1.25`, substantially better than the failed no-bias diagnostic.

### 3.3 Aggressive local-step 3×3 sweep

A 3×3 sweep over local steps with aggressive learning rates diverged for most configurations. This showed that local-step budget and learning rates must be tuned jointly.

### 3.4 Safe local-step 3×3 sweep

A safer 3×3 sweep with lower learning rates trained all configurations without NaNs. Increasing local steps improved quality. The best point was:

```text
reconstruction.steps = 50
client_update.steps = 50
test RMSE ≈ 1.265
```

## 4. Interpretation

This does not reproduce the exact Table 1 values from the paper. That would require:

- 500 rounds;
- full paper hyperparameter grid;
- exact StandardEval and ReconEval variants;
- exact TFF data pipeline and local batching;
- averaging over multiple runs.

However, the sanity check confirms the qualitative behavior:

- no-bias FEDRECON can learn;
- RMSE decreases with sufficient local-step budget;
- too aggressive learning rates cause numerical instability;
- differences from the paper are plausibly due to setup/budget differences, not an obvious implementation bug.

## 5. How to describe in the course paper

Suggested wording:

> To verify the implementation against the original FEDRECON setting, we conducted a paper-like sanity check using no-bias matrix factorization with embedding dimension 50, minibatch size 5, and 100 clients per round. Unlike the initial no-bias quick diagnostic, the paper-like setup trained successfully when sufficient local steps and stable learning rates were used. Although the experiment does not reproduce the exact benchmark numbers from the original paper, it confirms the expected qualitative behavior of FEDRECON: increasing the local reconstruction/update budget improves the no-bias model, while overly aggressive learning rates lead to divergence.
