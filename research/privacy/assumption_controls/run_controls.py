"""Independent synthetic controls for frozen-local factor update leakage.

No imports from the historical FedRecon client or attack implementation.
Training uses PyTorch autograd/optimizers; estimators use only public model and
the released message. Private truth enters only training and metric functions.
Run: <project python> run_controls.py [--output PATH]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


@dataclass
class PublicModel:
    Q: np.ndarray
    b: np.ndarray
    mu: float


@dataclass
class PrivateClient:
    p: np.ndarray
    user_bias: float
    ids: np.ndarray
    ratings: np.ndarray


@dataclass
class Message:
    dQ: np.ndarray
    db: np.ndarray | None
    dmu: float | None


def fixture(seed: int, n_items: int = 18, dim: int = 4):
    rng = np.random.default_rng(seed)
    public = PublicModel(rng.normal(0, .2, (n_items, dim)),
                         rng.normal(0, .15, n_items), 2.8)
    private = PrivateClient(rng.normal(0, .65, dim), .25,
                            np.array([1, 3, 4, 8, 11, 13, 16]),
                            rng.integers(1, 6, 7).astype(float))
    return public, private


def train(public: PublicModel, private: PrivateClient, *, steps=9,
          lr_q=.08, lr_b=.08, lr_mu=.08, item_bias=True, global_bias=True,
          minibatch=False, momentum=0., decay=0., optimizer="sgd",
          eps=1e-8, joint=False, grad_clip=None):
    """Client simulation. Returns message, private final p, and public diagnostics."""
    t = lambda a, grad=True: torch.tensor(a, dtype=torch.float64,
                                          requires_grad=grad)
    q, b, mu = t(public.Q), t(public.b, item_bias), t(public.mu, global_bias)
    p = t(private.p, joint)
    groups = [{"params": [q], "lr": lr_q}]
    if item_bias:
        groups.append({"params": [b], "lr": lr_b})
    if global_bias:
        groups.append({"params": [mu], "lr": lr_mu})
    if joint:
        groups.append({"params": [p], "lr": lr_q})
    if optimizer == "sgd":
        opt = torch.optim.SGD(groups, momentum=momentum, weight_decay=decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(groups, eps=eps, weight_decay=decay)
    else:
        raise ValueError(optimizer)
    clip_factors = []
    for step in range(steps):
        if minibatch:
            # Deterministic overlapping schedule, query exhausted over 7 steps.
            positions = np.array([(2 * step + j) % len(private.ids) for j in range(3)])
        else:
            positions = np.arange(len(private.ids))
        ids = torch.tensor(private.ids[positions], dtype=torch.int64)
        target = t(private.ratings[positions], False)
        prediction = q[ids] @ p + mu + private.user_bias
        if item_bias:
            prediction = prediction + b[ids]
        opt.zero_grad()
        ((prediction - target) ** 2).mean().backward()
        if grad_clip is not None:
            params = [v for group in groups for v in group["params"]]
            total_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            clip_factors.append(min(1., float(grad_clip / (total_norm + 1e-6))))
        opt.step()
    message = Message(q.detach().numpy() - public.Q,
                      b.detach().numpy() - public.b if item_bias else None,
                      float(mu.detach()) - public.mu if global_bias else None)
    return message, p.detach().numpy(), {"min_clip_factor": min(clip_factors, default=1.)}


def ratio(message: Message, lr_ratio: float = 1.):
    """Only observable deltas and, if provided, known learning-rate ratio."""
    if message.db is None or np.dot(message.db, message.db) == 0:
        raise ValueError("nonzero item bias message required")
    return message.dQ.T @ message.db / np.dot(message.db, message.db) / lr_ratio


def global_bias_ratio(message: Message, lr_ratio: float = 1.):
    if message.dmu is None or abs(message.dmu) < 1e-12:
        raise ValueError("nonzero global bias message required")
    return message.dQ.sum(axis=0) / message.dmu / lr_ratio


def remove_decay(public: PublicModel, message: Message, rho: float):
    """Known initial global state and deterministic dense decay, not private p."""
    return Message(message.dQ - (rho - 1) * public.Q,
                   message.db - (rho - 1) * public.b,
                   message.dmu)


def relative_error(estimated, truth):
    return float(np.linalg.norm(estimated - truth) / np.linalg.norm(truth))


def invariant_residual(message, true_p):
    return float(np.linalg.norm(message.dQ - np.outer(message.db, true_p))
                 / max(np.linalg.norm(message.dQ), 1e-30))


def support(message, threshold=1e-12):
    return np.flatnonzero(np.linalg.norm(message.dQ, axis=1) > threshold)


def aggregate(messages, weights):
    return Message(sum(w * m.dQ for w, m in zip(weights, messages)),
                   sum(w * m.db for w, m in zip(weights, messages)),
                   sum(w * m.dmu for w, m in zip(weights, messages)))


def anonymous_clusters(message, tolerance=1e-8):
    """Cluster observable per-row ratios; no true masks or profiles supplied."""
    clusters = []
    for i in np.flatnonzero(np.abs(message.db) > 1e-12):
        v = message.dQ[i] / message.db[i]
        for c in clusters:
            if np.linalg.norm(v - np.array(c["profile"])) < tolerance:
                c["rows"].append(int(i))
                break
        else:
            clusters.append({"profile": v.tolist(), "rows": [int(i)]})
    return clusters


def quantize(x, bits=4):
    limit = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / limit
    return np.round(x / scale).clip(-limit, limit) * scale if scale else x.copy()


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, details = [], {}

    def record(name, seed, m, p, *, estimate=None, extra=None):
        est = ratio(m) if estimate is None else estimate
        row = {"case": name, "seed": seed,
               "relative_p_error": relative_error(est, p),
               "invariant_residual": invariant_residual(m, p) if m.db is not None else None}
        row.update(extra or {})
        rows.append(row)
        return row

    for seed in range(20):
        public, private = fixture(seed)
        p = private.p
        baseline, _, _ = train(public, private)
        for name, kwargs in [
            ("sgd_full_batch", {}),
            ("sgd_minibatch", {"minibatch": True}),
            ("momentum_zero_initial_state", {"minibatch": True, "momentum": .8}),
            ("common_gradient_clip", {"minibatch": True, "grad_clip": .03}),
        ]:
            m, _, diagnostics = train(public, private, **kwargs)
            row = record(name, seed, m, p, extra=diagnostics)
            assert row["relative_p_error"] < 1e-10, row
        delta_norm = np.sqrt(np.sum(baseline.dQ**2) + np.sum(baseline.db**2) + baseline.dmu**2)
        clip_factor = min(1., .03 / delta_norm)
        clipped = Message(clip_factor * baseline.dQ, clip_factor * baseline.db, clip_factor * baseline.dmu)
        assert record("common_delta_clip", seed, clipped, p,
                      extra={"min_clip_factor": clip_factor})["relative_p_error"] < 1e-10

        m, _, _ = train(public, private, lr_q=.08, lr_b=.02)
        record("different_lr_naive", seed, m, p)
        assert record("different_lr_known_ratio", seed, m, p,
                      estimate=ratio(m, 4.))["relative_p_error"] < 1e-10

        m, _, _ = train(public, private, decay=.15)
        record("dense_weight_decay_naive", seed, m, p,
               extra={"support_fp": len(set(support(m)) - set(private.ids))})
        adjusted = remove_decay(public, m, (1 - .08 * .15) ** 9)
        row = record("dense_weight_decay_compensated", seed, adjusted, p,
                     extra={"support_fp": len(set(support(adjusted)) - set(private.ids))})
        assert row["relative_p_error"] < 1e-10 and row["support_fp"] == 0, row

        m, _, _ = train(public, private, optimizer="adam")
        record("adam_naive", seed, m, p,
               extra={"relative_sign_p_error": relative_error(ratio(m), np.sign(p))})
        m, final_p, _ = train(public, private, joint=True)
        record("joint_p_naive", seed, m, p,
               extra={"error_vs_final_p": relative_error(ratio(m), final_p)})

        m, _, _ = train(public, private, item_bias=False, minibatch=True)
        row = record("no_item_bias_global_bias_ratio", seed, m, p,
                     estimate=global_bias_ratio(m), extra={"delta_mu": m.dmu})
        assert row["relative_p_error"] < 1e-10, row

        m = Message(quantize(baseline.dQ), quantize(baseline.db), quantize(np.array(baseline.dmu)).item())
        record("symmetric_int4_per_tensor", seed, m, p)
        rng = np.random.default_rng(10000 + seed)
        for level in [.001, .01, .1]:
            q_noise = level * np.sqrt(np.mean(baseline.dQ ** 2))
            b_noise = level * np.sqrt(np.mean(baseline.db ** 2))
            m = Message(baseline.dQ + rng.normal(0, q_noise, baseline.dQ.shape),
                        baseline.db + rng.normal(0, b_noise, baseline.db.shape), baseline.dmu)
            record(f"noise_relative_rms_{level}", seed, m, p)

    # Aggregation controls share the public model. Masks are used for evaluation only.
    public, a = fixture(800)
    _, b = fixture(801)
    a = replace(a, ids=np.array([0, 1, 2, 3, 4, 5, 6]))
    cases = {"disjoint": np.array([8, 9, 10, 11, 12, 13, 14]),
             "partial_overlap": np.array([4, 5, 6, 7, 8, 9, 10]),
             "complete_overlap": a.ids.copy()}
    for name, ids in cases.items():
        second = replace(b, ids=ids)
        ma, _, _ = train(public, a)
        mb, _, _ = train(public, second)
        ag = aggregate([ma, mb], [.4, .6])
        clusters = anonymous_clusters(ag)
        singleton_rows = set(a.ids) ^ set(second.ids)
        singleton_errors = []
        for i in singleton_rows:
            true = a.p if i in set(a.ids) else second.p
            singleton_errors.append(relative_error(ag.dQ[i] / ag.db[i], true))
        # Clusters with repeated ratios are attack-derived candidate profiles.
        repeated = [c for c in clusters if len(c["rows"]) >= 2]
        details[f"aggregation_{name}"] = {
            "weights": [.4, .6], "num_clients": 2, "item_sets_for_evaluation": [a.ids.tolist(), second.ids.tolist()],
            "clusters_from_observation": clusters, "repeated_ratio_clusters": len(repeated),
            "singleton_rows": len(singleton_rows),
            "max_singleton_relative_error": max(singleton_errors, default=None),
            "pooled_ratio_error_to_user_a": relative_error(ratio(ag), a.p),
            "pooled_ratio_error_to_user_b": relative_error(ratio(ag), second.p),
            "identity_attribution": "not available from this experiment"}
        if name in ("disjoint", "partial_overlap"):
            assert len(repeated) == 2
            assert max(singleton_errors) < 1e-10

    # An interacted row can have no observable update: zero residual everywhere.
    public, private = fixture(900)
    exact_predictions = public.Q[private.ids] @ private.p + public.b[private.ids] + public.mu + private.user_bias
    zero = replace(private, ratings=exact_predictions)
    zm, _, _ = train(public, zero)
    details["zero_residual_support"] = {"interacted_rows": len(private.ids), "observed_rows": len(support(zm)),
                                          "message_norm": float(np.linalg.norm(zm.dQ))}
    assert len(support(zm)) == 0

    # Two steps at the exact oscillation boundary return Q, b, mu to initial values.
    # Balanced residual is orthogonal to 1, so mu stays fixed and eigenvalue c is enough.
    ids = np.array([0, 1, 2, 3])
    p = np.array([.5, -.25, .75, .125])
    e = np.array([.2, -.2, .1, -.1])
    osc = PrivateClient(p, .15, ids, public.Q[ids] @ p + public.b[ids] + public.mu + .15 - e)
    eta = len(ids) / (1 + np.dot(p, p))
    om, _, _ = train(public, osc, steps=2, lr_q=eta, lr_b=eta, lr_mu=eta)
    details["two_step_exact_cancellation"] = {"steps": 2, "lr": eta,
        "interacted_rows": len(ids), "observed_rows_threshold_1e_12": len(support(om)),
        "max_abs_message": max(float(np.max(np.abs(om.dQ))), float(np.max(np.abs(om.db))), abs(om.dmu))}
    assert details["two_step_exact_cancellation"]["max_abs_message"] < 1e-12

    # Without item bias, sum residual = 0 also makes delta mu zero.
    # One-step messages coincide for p' = 2p, e' = e/2 and corresponding ratings.
    no_b_a = PrivateClient(p, .0, ids, public.Q[ids] @ p + public.mu - e)
    no_b_b = PrivateClient(2*p, .0, ids, public.Q[ids] @ (2*p) + public.mu - e/2)
    na, _, _ = train(public, no_b_a, steps=1, item_bias=False)
    nb, _, _ = train(public, no_b_b, steps=1, item_bias=False)
    _, _, vh = np.linalg.svd(na.dQ, full_matrices=False)
    cosine = abs(float(vh[0] @ p / np.linalg.norm(p)))
    details["no_item_bias_zero_global_delta_ambiguity"] = {
        "p_a": p.tolist(), "p_b": (2*p).tolist(),
        "ratings_a": no_b_a.ratings.tolist(), "ratings_b": no_b_b.ratings.tolist(),
        "delta_mu_a": na.dmu, "delta_mu_b": nb.dmu,
        "max_abs_message_difference": float(np.max(np.abs(na.dQ-nb.dQ))),
        "rank_one_direction_absolute_cosine": cosine,
        "max_profile_scale_ratio": 2., "ratings_type": "continuous, not integer-constrained"}
    assert np.max(np.abs(na.dQ-nb.dQ)) < 1e-12 and abs(na.dmu-nb.dmu) < 1e-12
    assert cosine > 1-1e-12

    summary = {}
    for name in sorted({r["case"] for r in rows}):
        selected = [r for r in rows if r["case"] == name]
        errors = [r["relative_p_error"] for r in selected]
        summary[name] = {"trials": len(selected), "mean_relative_p_error": float(np.mean(errors)),
                         "max_relative_p_error": max(errors), "min_relative_p_error": min(errors)}
    config = {"seeds": list(range(20)), "dtype": "float64", "device": "cpu",
        "n_catalog_items": 18, "latent_dim": 4, "query_size": 7, "query_steps": 9,
        "default_lr": .08, "default_loss": "mean squared error", "quantization": "symmetric per tensor, [-7,7]",
        "noise": "Gaussian, independent Q/b, sigma = fraction of that tensor RMS; NOT a DP mechanism",
        "no_external_data": True, "historical_code_imported": False,
        "seeded_fixtures": "NumPy default_rng, seeds defined in run_controls.py"}
    payload = {"config": config, "summary": summary, "details": details, "rows": rows,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    run(parser.parse_args().output)
