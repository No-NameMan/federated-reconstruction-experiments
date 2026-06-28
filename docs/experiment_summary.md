# Experiment summary

This document summarizes the main experiments in the FedRecon course project.

## Dataset and evaluation protocol

Experiments use MovieLens 1M. Users are treated as clients in a federated learning setup. Validation and test users are evaluated with a reconstruction protocol: local user parameters are reconstructed on a support split, and RMSE, MAE and rounded rating accuracy are computed on a query split.

The main model is matrix factorization with item embeddings, user embeddings and bias terms. User embeddings and user biases are local parameters; item embeddings, item biases and the global bias are global parameters.

## Main methods

### FedRecon

FedRecon keeps user-specific parameters local. In each round, the server sends global parameters to sampled clients. A client reconstructs its local profile, computes an update for the global parameters and sends only the global delta back to the server.

### FedAvg-like baseline

The FedAvg-like baseline also trains client-local user profiles and global item parameters, but it is used as a simpler federated comparison point. Only global deltas are aggregated.

### Centralized MF + ReconEval

The centralized baseline is trained without federated constraints on all train ratings. It is not directly comparable in privacy or deployment setting, but gives a useful quality reference.

## Key experiments

### 1. Baseline comparison

The 100-round comparison showed that FedRecon and FedAvg-like reach almost identical test quality in the main setting:

| Method | Test RMSE | Test MAE | Accuracy |
|---|---:|---:|---:|
| FedRecon | 1.0035 ± 0.0007 | 0.8088 ± 0.0003 | 0.3759 ± 0.0001 |
| FedAvg-like | 1.0032 ± 0.0005 | 0.8085 ± 0.0006 | 0.3760 ± 0.0008 |
| Centralized MF + ReconEval | ≈ 0.881 | ≈ 0.700 | ≈ 0.431 |

Interpretation: in this configuration, the learned global item parameters and subsequent local reconstruction dominate the final quality. The centralized baseline performs better, but it is trained under a different and less constrained setup.

### 2. Evaluation reconstruction budget

The evaluation sweep varied the number of reconstruction steps used for new users. Without reconstruction, validation RMSE was much worse. The first few reconstruction steps produced most of the gain, and the curve saturated around 5–10 steps.

Representative final validation RMSE values:

| Evaluation reconstruction steps | Final val RMSE |
|---:|---:|
| 0 | 1.0571 |
| 1 | 1.0318 |
| 2 | 1.0158 |
| 5 | 0.9963 |
| 10 | 0.9921 |
| 20 | 0.9928 |

### 3. Client update steps and learning rate

Increasing the number of local update steps for global parameters improved quality for a fixed number of communication rounds. The cost is additional client-side computation, while the transmitted message size per round remains unchanged.

### 4. Quantization trade-off

Quantization was tested for global deltas. fp16, int8 and int4 produced almost the same validation RMSE as full precision in this simulation. int4 reduced transmitted data from about 367 MB to about 46 MB in the 30-round quantization sweep. Sign compression reduced communication more aggressively but degraded RMSE.

### 5. Client heterogeneity

Users were grouped by the number of ratings. Reconstruction helped all groups, but the effect differed. Users with fewer ratings saturated quickly, while users with many ratings benefited from additional reconstruction steps but remained a harder group.

### 6. Client dropout

The dropout stress test simulated clients that were selected but did not return updates. Quality remained stable when enough clients effectively participated in each round. Degradation became more visible when the number of successful client updates was too small.

### 7. Paper-like sanity check

A separate no-bias configuration used embedding dimension 50, minibatch size 5 and 100 clients per round. This experiment was not a full reproduction of the original FedRecon paper, but it verified that the implementation remains stable in a closer setting and that increasing reconstruction/update steps improves quality under safe learning rates.

## Main conclusions

- Local reconstruction is the key mechanism for adapting to new users.
- A small reconstruction budget is usually enough; returns diminish after the first several steps.
- More local client update steps can improve quality but increase client computation.
- Moderate update quantization, especially int4, is promising for reducing communication in this simulation.
- The current experiments are limited by the simplified MovieLens setup and an idealized communication model.

## Future work

- Adaptive reconstruction budgets based on the number of local ratings.
- More realistic communication and client availability models.
- Differential privacy.
- Larger recommendation datasets.
- Additional server optimizers and personalization baselines.
