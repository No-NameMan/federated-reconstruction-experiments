# FedRecon privacy checks

These experiments examine what a passive server can reconstruct from one uncompressed client update. The client reconstructs a profile on support data, then holds it fixed during SGD on query ratings. The decoder receives the initial global model, the final delta and public optimizer settings. Ground truth is used for scoring.

The current result is a reproducible sufficient condition for recovering query item IDs and rating differences in this model. A synthetic example with hidden minibatch order gives identical updates for different ratings. The MovieLens diagnostic uses 30 selected training clients and one original checkpoint. Its outcomes describe those updates. This is an application and verification of mechanisms with known predecessors. [FedRecon](https://arxiv.org/abs/2102.03448v6) · [Federated matrix factorization](https://arxiv.org/abs/1906.05108v1) · [Multi-step weight/bias ratio, §5.3](https://arxiv.org/abs/2112.02918v2)

## Run the independent verification

Run from the repository root. The reference environment uses Python 3.13, PyTorch 2.12.0 CPU and the pinned packages in [requirements.txt](requirements.txt). With a virtual environment activated:

~~~sh
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/privacy/requirements.txt
python research/privacy/run_synthetic.py --output runs/privacy/synthetic
~~~

Expect `PASS`, 480/480 exact relative-rating results, 60 additional minibatch profile checks and an exact collision between two hidden-order messages. The 480 settings reuse 60 synthetic fixtures across step counts and numeric precision. Add `--controls` and use a fresh output path to run 320 additional checks of protocol assumptions. Outputs in `runs/` include JSON, CSV and logs. These checks use synthetic data and CPU only.

## Run with MovieLens

Extract the [MovieLens 1M release](https://grouplens.org/datasets/movielens/1m/) so `data/ml-1m/ratings.dat` is available, or use another directory containing `ml-1m/ratings.dat` via `--data-dir`. The script checks the input hash. The first command verifies that the original simulator trains and saves a checkpoint. The second repeats the reference quality comparison for one seed.

~~~sh
python research/privacy/run_movielens.py smoke --data-dir data --output runs/privacy/smoke
python research/privacy/run_movielens.py reference --data-dir data --output runs/privacy/reference
~~~

These training and replay modes require the simulator and debug config from [coursework revision 02293ae](https://github.com/No-NameMan/federated-reconstruction-experiments/tree/02293ae3dd917ea0f45ceafc3f30facc4a7af982). The default is the main checkout while those files still match. If the simulator has changed, use `git worktree add --detach .local/coursework-baseline 02293ae`, then pass `--repo .local/coursework-baseline`. The output records the checkout and baseline revisions. Existing output directories are rejected.

The optional `pilot` mode also requires the original May `final_model.pt` checkpoint, available separately from the author. Its archive member and SHA-256 are recorded in [reference/observed.json](reference/observed.json).

~~~sh
python research/privacy/run_movielens.py pilot --data-dir data --checkpoint .local/final_model.pt --output runs/privacy/pilot
~~~

Expected pilot outcomes are 2,153 query pairs, 30/30 exact relative rating vectors and 18/30 unique correct absolute rating vectors within the query-stage model. The reference checkpoint and dataset are kept outside Git. Training a new checkpoint serves as a separate quality comparison.

## Inspect the evidence

- [Independent forward model and decoder](independent_math/verify_math.py) and [test grid](independent_math/config.json)
- [Hidden-order example](independent_math/batch_observability.py) and [recorded messages](reference/batch_observability.json)
- [Saved synthetic numerical results](reference/synthetic_summary.json)
- [MovieLens reference, ablation and pilot summaries](reference/observed.json) and [smoke-run log by round](reference/training_smoke_rounds.csv)

The theorem and the diagnostics concern an individual message under the declared protocol. Population exposure, secure aggregation and constraints imposed by support reconstruction require separate analysis.
