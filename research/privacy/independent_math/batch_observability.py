"""Query-stage minibatch/duplicate-item observability; no historical imports.

Uses the independent PyTorch oracle from verify_math, but a new NumPy attack
builds the entire linear observation operator for a disclosed batch schedule.
No private ratings, bias or factor are provided to the attacker.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from verify_math import Public, oracle, recover_p, message_distance

HERE = Path(__file__).resolve().parent


def known_schedule_attack(public, message, query_count, schedule_item_ids, eta):
    p = recover_p(message)
    ids = np.flatnonzero((message.db != 0) | np.any(message.dq != 0, axis=1))
    if len(ids) != query_count:
        raise ValueError('Not every unique query item is observable')
    positions = {int(item): j for j, item in enumerate(ids)}
    n = query_count
    a = (1 + float(p @ p)) * np.eye(n) + np.ones((n, n))
    residual_operator = np.eye(n)
    bias_operator = np.zeros((n, n))
    for batch in schedule_item_ids:
        counts = np.zeros(n)
        for item in batch:
            counts[positions[int(item)]] += 1
        w = np.diag(counts / len(batch))
        bias_operator -= 2 * eta * w @ residual_operator
        residual_operator = (np.eye(n) - 2 * eta * a @ w) @ residual_operator
    rank = np.linalg.matrix_rank(bias_operator)
    if rank != n:
        raise ValueError('Singular batch observation operator')
    e0 = np.linalg.solve(bias_operator, message.db[ids])
    return {'ids': ids, 'p': p, 'relative_ratings': public.q[ids] @ p + public.b[ids] + public.mu - e0,
            'rank': int(rank), 'condition': float(np.linalg.cond(bias_operator))}


def message_json(message):
    return {'dq': message.dq.tolist(), 'db': message.db.tolist(), 'dmu': message.dmu}


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    trials = []
    # Every item is visited at least once; later batches sample with replacement.
    # The schedule is ADDITIONAL attacker knowledge, not sent by the old client.
    for seed in range(10):
        rng = np.random.default_rng(91000 + seed)
        n, d, steps, eta = 7, 4, 11, .04
        public = Public(rng.normal(0, .2, (12, d)), rng.normal(0, .1, 12), 2.7)
        ids = np.sort(rng.choice(12, n, replace=False))
        p, bu = rng.normal(0, .3, d), float(rng.normal(0, .2))
        ratings = rng.integers(1, 6, n).astype(float)
        schedule_positions = [[int(j)] for j in rng.permutation(n)]
        schedule_positions += [rng.integers(0, n, 3).tolist() for _ in range(steps - n)]
        message, _, initial = oracle(public, p, bu, ids, ratings, eta, steps,
                                     'float64', schedule_positions)
        schedule_ids = [ids[batch].tolist() for batch in schedule_positions]
        inferred = known_schedule_attack(initial, message, n, schedule_ids, eta)
        error = float(np.max(np.abs(inferred['relative_ratings'] - (ratings - bu))))
        assert error < 1e-10
        trials.append({'seed': seed, 'n': n, 'd': d, 'steps': steps, 'eta': eta,
                       'schedule_item_ids': schedule_ids,
                       'max_relative_rating_abs_error': error,
                       'observation_rank': inferred['rank'], 'condition': inferred['condition']})

    # EXACT counterexample, binary-exact constants, distinct integer ratings.
    # Only the order of two singleton minibatches differs; both visit all items.
    public = Public(np.zeros((2, 1)), np.zeros(2), 4.)
    p, bu, ids, eta = np.array([.5]), 0., np.arange(2), .25
    r_a, r_b = np.array([2., 1.]), np.array([1., 2.])
    schedule_a, schedule_b = [[0], [1]], [[1], [0]]
    m_a, _, initial = oracle(public, p, bu, ids, r_a, eta, 2, 'float64', schedule_a)
    m_b, _, _ = oracle(public, p, bu, ids, r_b, eta, 2, 'float64', schedule_b)
    inferred_a = known_schedule_attack(initial, m_a, 2, schedule_a, eta)
    inferred_b = known_schedule_attack(initial, m_a, 2, schedule_b, eta)
    assert message_distance(m_a, m_b) == 0
    assert np.array_equal(inferred_a['relative_ratings'], r_a)
    assert np.array_equal(inferred_b['relative_ratings'], r_b)
    unknown_order = {'ratings_a': r_a.tolist(), 'ratings_b': r_b.tolist(),
                     'p_both': p.tolist(), 'user_bias_both': bu,
                     'eta': eta, 'steps': 2, 'n': 2,
                     'schedule_a': schedule_a, 'schedule_b': schedule_b,
                     'max_message_difference': message_distance(m_a, m_b),
                     'shared_message': message_json(m_a),
                     'scope': 'Query stage with hidden minibatch order; same known public initial model, n, steps, lr, even p and user bias. Integer ratings 1..5. No support reconstruction constraint.'}

    # Duplicate user-item observations leak their weighted mean, not each label.
    duplicate_ids = np.array([0, 0, 1, 1])
    r_a, r_b = np.array([1., 5., 2., 4.]), np.array([2., 4., 3., 3.])
    m_a, _, _ = oracle(public, p, bu, duplicate_ids, r_a, .125, 5, 'float64')
    m_b, _, _ = oracle(public, p, bu, duplicate_ids, r_b, .125, 5, 'float64')
    assert message_distance(m_a, m_b) < 1e-13
    duplicate = {'query_item_ids': duplicate_ids.tolist(), 'ratings_a': r_a.tolist(),
                 'ratings_b': r_b.tolist(), 'per_item_means_both': [3., 3.],
                 'eta': .125, 'steps': 5, 'max_message_difference': message_distance(m_a, m_b),
                 'shared_message': message_json(m_a),
                 'scope': 'Full-batch MSE with duplicates; number and mean of ratings per item fixed. Individual ratings differ.'}

    payload = {'status': 'PASS', 'device': 'cpu', 'dtype': 'float64',
               'runtime_seconds': time.perf_counter() - start,
               'known_schedule_trials': trials,
               'known_schedule_max_error': max(r['max_relative_rating_abs_error'] for r in trials),
               'hidden_order_counterexample': unknown_order, 'duplicate_items_counterexample': duplicate,
               'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
               'code_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                               for name in ['batch_observability.py', 'verify_math.py']}}
    (HERE / 'batch_observability.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
