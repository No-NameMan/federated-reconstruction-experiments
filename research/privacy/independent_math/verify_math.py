"""Independent NumPy attack / PyTorch autograd oracle, CPU only.

No imports from the FedRecon repository or from its earlier leakage probe.
All attacker routines take only initial public parameters, transmitted deltas,
and explicitly declared public metadata. Truth is used only by verification.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Public:
    q: np.ndarray
    b: np.ndarray
    mu: float


@dataclass(frozen=True)
class Message:
    dq: np.ndarray
    db: np.ndarray
    dmu: float


@dataclass(frozen=True)
class Metadata:
    n: int
    steps: int
    eta: float
    global_bias_trainable: bool = True


def oracle(public, p, bu, ids, ratings, eta, steps, dtype, batches=None,
           global_bias_trainable=True):
    """Victim computation. Only Message is given to the attacks."""
    tdtype = getattr(torch, dtype)
    q = torch.tensor(public.q, dtype=tdtype, requires_grad=True)
    b = torch.tensor(public.b, dtype=tdtype, requires_grad=True)
    mu = torch.tensor(public.mu, dtype=tdtype,
                      requires_grad=global_bias_trainable)
    frozen_p = torch.tensor(p, dtype=tdtype)
    frozen_bu = torch.tensor(bu, dtype=tdtype)
    indices = torch.tensor(ids, dtype=torch.long)
    r = torch.tensor(ratings, dtype=tdtype)
    q_start, b_start, mu_start = q.detach().clone(), b.detach().clone(), mu.detach().clone()
    params = [q, b] + ([mu] if global_bias_trainable else [])
    residuals = []
    for step in range(steps):
        residuals.append((q[indices] @ frozen_p + b[indices] + mu + frozen_bu - r).detach().double().numpy())
        chosen = torch.arange(len(ids)) if batches is None else torch.tensor(batches[step])
        chosen_ids = indices[chosen]
        predicted = q[chosen_ids] @ frozen_p + b[chosen_ids] + mu + frozen_bu
        loss = ((predicted - r[chosen]) ** 2).mean()
        gradients = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for parameter, gradient in zip(params, gradients):
                parameter.add_(gradient, alpha=-eta)
    residuals.append((q[indices] @ frozen_p + b[indices] + mu + frozen_bu - r).detach().double().numpy())
    message = Message((q.detach() - q_start).double().numpy(),
                      (b.detach() - b_start).double().numpy(),
                      float((mu.detach() - mu_start).item()))
    actual_public = Public(q_start.double().numpy(), b_start.double().numpy(), float(mu_start.item()))
    return message, np.array(residuals), actual_public


def recover_p(message):
    denominator = float(message.db @ message.db)
    if denominator == 0:
        raise ValueError("No nonzero item-bias signal: p is not identified by this ratio")
    return message.dq.T @ message.db / denominator


def spectral_response(lam, a, steps):
    """Positive h(lambda), where db=-H e0; signed outside stable region."""
    x = a * lam
    base = 1.0 - x
    if base > 0:
        numerator = -np.expm1(steps * np.log1p(-x))
    else:
        numerator = 1.0 - base ** steps
    return float(numerator / lam)


def recover_relative(public, message, metadata):
    """Public attack; never receives p, bu, ratings or true query IDs."""
    p = recover_p(message)
    observed_ids = np.flatnonzero(np.any(message.dq != 0.0, axis=1) | (message.db != 0.0))
    if len(observed_ids) != metadata.n:
        raise ValueError("Observed support does not reveal all n query items")
    db = message.db[observed_ids]
    c = 1.0 + float(p @ p)
    a = 2.0 * metadata.eta / metadata.n
    lam_mean = c + metadata.n if metadata.global_bias_trainable else c
    h_mean = spectral_response(lam_mean, a, metadata.steps)
    h_perp = spectral_response(c, a, metadata.steps)
    if abs(h_mean) < 1e-14 or (metadata.n > 1 and abs(h_perp) < 1e-14):
        raise ValueError("Singular residual-to-delta map")
    mean = db.mean()
    initial_residual = -(db - mean) / h_perp - mean / h_mean
    z = public.q[observed_ids] @ p + public.b[observed_ids] + public.mu
    relative = z - initial_residual
    normalized_integers = np.rint(relative - relative.min()).astype(int)
    integer_grid_residual = max_abs(relative - relative.min() - normalized_integers)
    candidate_ratings = [normalized_integers + shift
                         for shift in range(1, 6 - int(normalized_integers.max()))] if integer_grid_residual < 1e-3 else []
    hs = [abs(h_mean)] + ([abs(h_perp)] if metadata.n > 1 else [])
    return dict(p=p, ids=observed_ids, continuous=relative,
                normalized=normalized_integers, candidates=candidate_ratings,
                integer_grid_residual=integer_grid_residual,
                condition=max(hs)/min(hs), inverse_gain=1/min(hs),
                h_mean=h_mean, h_perp=h_perp)


def recover_rating_differences(public, message, metadata):
    """Centered ratings need only the centered response, not the mean response.

    This distinction matters at a singular mean mode: r-bu is not fully
    identified there, while pairwise rating differences can remain identified.
    """
    p = recover_p(message)
    ids = np.flatnonzero(np.any(message.dq != 0.0, axis=1) | (message.db != 0.0))
    if len(ids) != metadata.n:
        raise ValueError('Incomplete item support')
    db = message.db[ids]
    h = spectral_response(1 + float(p @ p), 2 * metadata.eta / metadata.n, metadata.steps)
    if abs(h) < 1e-14:
        raise ValueError('Singular centered response')
    z = public.q[ids] @ p + public.b[ids] + public.mu
    return z - z.mean() + (db - db.mean()) / h


def make_case(seed, n, d, dtype):
    rng = np.random.default_rng(seed * 10000 + n * 100 + d)
    m = n + 5
    ids = np.sort(rng.choice(m, n, replace=False))
    p = rng.normal(0, 0.3, d)
    public = Public(rng.normal(0, 0.25, (m, d)), rng.normal(0, 0.15, m), 2.375)
    ratings = rng.integers(1, 6, n).astype(float)
    bu = float(rng.normal(0, 0.25))
    # Data actually consumed by victim, including dtype conversion.
    ndtype = getattr(np, dtype)
    p = p.astype(ndtype).astype(float)
    bu = float(ndtype(bu))
    return public, p, bu, ids, ratings


def max_abs(x):
    return float(np.max(np.abs(x)))


def comparison_row(public, message, p, bu, ids, ratings, meta, residuals, dtype):
    attacked = recover_relative(public, message, meta)
    assert np.array_equal(attacked['ids'], ids)
    c = 1 + float(p @ p)
    a = 2 * meta.eta / meta.n
    amat = c * np.eye(meta.n) + np.ones((meta.n, meta.n))
    transition = np.eye(meta.n) - a * amat
    recurrence_error = max_abs(residuals[1:] - (transition @ residuals[:-1].T).T)
    # A separately evaluated matrix geometric sum, not attack's eigenspace expression.
    hmat = np.zeros((meta.n, meta.n))
    power = np.eye(meta.n)
    for _ in range(meta.steps):
        hmat += a * power
        power = transition @ power
    matrix_e0 = np.linalg.solve(hmat, -message.db[ids])
    truth_normalized = (ratings - ratings.min()).astype(int)
    error_continuous = max_abs(attacked['continuous'] - (ratings - bu))
    identity_error = max_abs(message.dq - message.db[:, None] * p)
    factor_error = float(np.linalg.norm(attacked['p'] - p) / np.linalg.norm(p))
    relative_exact = np.array_equal(attacked['normalized'], truth_normalized)
    true_candidate_present = any(np.array_equal(r, ratings.astype(int)) for r in attacked['candidates'])
    expected_candidates = 5 - int(ratings.max() - ratings.min())
    assert relative_exact and true_candidate_present
    assert len(attacked['candidates']) == expected_candidates
    assert max_abs(message.db[np.setdiff1d(np.arange(len(public.b)), ids)]) == 0
    assert factor_error < (2e-10 if dtype == 'float64' else 2e-4)
    assert error_continuous < (2e-9 if dtype == 'float64' else 2e-3)
    assert recurrence_error < (2e-12 if dtype == 'float64' else 2e-5)
    assert max_abs(matrix_e0 - residuals[0]) < (2e-10 if dtype == 'float64' else 2e-3)
    assert abs(message.dmu - float(message.db.sum())) < (2e-12 if dtype == 'float64' else 2e-5)
    return dict(p_relative_l2=factor_error,
                identity_max_abs=identity_error,
                continuous_max_abs=error_continuous,
                residual_recurrence_max_abs=recurrence_error,
                independent_matrix_e0_max_abs=max_abs(matrix_e0 - residuals[0]),
                relative_ratings_exact=bool(relative_exact),
                absolute_candidates=len(attacked['candidates']),
                true_candidate_present=bool(true_candidate_present),
                global_bias_sum_error=abs(message.dmu - float(message.db.sum())),
                inverse_condition=attacked['condition'],
                inverse_gain=attacked['inverse_gain'])


def full_batch_grid(config):
    rows = []
    for seed, n, d, steps, dtype in itertools.product(config['seeds'], config['n_values'],
            config['d_values'], config['step_values'], config['dtypes']):
        public, p, bu, ids, ratings = make_case(seed, n, d, dtype)
        eta = config['learning_rate']
        message, residuals, initial = oracle(public, p, bu, ids, ratings, eta, steps, dtype)
        row = comparison_row(initial, message, p, bu, ids, ratings,
                             Metadata(n, steps, eta), residuals, dtype)
        rows.append(dict(seed=seed, n=n, d=d, steps=steps, dtype=dtype, eta=eta, **row))
    return rows


def minibatch_grid(config):
    rows = []
    for seed, n, steps, dtype in itertools.product(config['seeds'], config['minibatch_n'],
            config['minibatch_steps'], config['dtypes']):
        public, p, bu, ids, ratings = make_case(seed, n, 4, dtype)
        rng = np.random.default_rng(7000 + seed + n + steps)
        batches = [rng.choice(n, int(rng.integers(1, n+1)), replace=False).tolist() for _ in range(steps)]
        message, _, _ = oracle(public, p, bu, ids, ratings, config['learning_rate'], steps, dtype, batches)
        phat = recover_p(message)
        error = float(np.linalg.norm(phat-p)/np.linalg.norm(p))
        assert error < (1e-10 if dtype == 'float64' else 2e-4)
        rows.append(dict(seed=seed, n=n, d=4, steps=steps, dtype=dtype,
                         p_relative_l2=error,
                         identity_max_abs=max_abs(message.dq-message.db[:,None]*p),
                         batches=json.dumps(batches)))
    return rows


def message_distance(first, second):
    return max(max_abs(first.dq-second.dq), max_abs(first.db-second.db), abs(first.dmu-second.dmu))


def counterexamples(config):
    rows = []
    public = Public(np.array([[0.125, 0.25], [-0.125, 0.25], [0.25, -0.125]]), np.zeros(3), 0.25)
    p, bu, ids = np.array([0.5, -0.25]), 0.375, np.arange(3)
    ratings = np.array([1., 2., 3.])
    first, _, initial = oracle(public,p,bu,ids,ratings,0.125,7,'float64')
    second, _, _ = oracle(public,p,bu+1,ids,ratings+1,0.125,7,'float64')
    estimate = recover_relative(initial,first,Metadata(3,7,0.125))
    distance = message_distance(first,second)
    # The algebraic symmetry is exact; different evaluation orders of shifted
    # ratings and biases need not be bitwise equal after floating-point updates.
    assert distance < 1e-13
    assert len(estimate['candidates']) == 3
    rows.append(dict(name='common_shift_nonidentifiability', max_message_difference=distance,
                     ratings_first=ratings.tolist(), ratings_second=(ratings+1).tolist(),
                     bu_first=bu, bu_second=bu+1, absolute_candidates=[r.tolist() for r in estimate['candidates']],
                     scope='Query stage only; no further constraint from support/reconstruction'))

    # Exact singular mean mode: n=2, ||p||^2=1, c=2, a=.5, k=2.
    # Initial residual means disappear from the final parameter delta.
    public = Public(np.zeros((2,2)),np.zeros(2),0.)
    p, ids = np.array([1.,0.]), np.arange(2)
    first, _, initial = oracle(public,p,0.,ids,np.array([1.,2.]),.5,2,'float64')
    second, _, _ = oracle(public,p,0.,ids,np.array([2.,3.]),.5,2,'float64')
    error_type = None
    try:
        recover_relative(initial,first,Metadata(2,2,.5))
    except ValueError as error:
        error_type = str(error)
    assert message_distance(first,second) == 0 and error_type == 'Singular residual-to-delta map'
    centered = recover_rating_differences(initial, first, Metadata(2, 2, .5))
    assert np.array_equal(centered, np.array([-.5, .5]))
    rows.append(dict(name='singular_mean_mode', n=2, d=2, eta=.5, steps=2,
                     p_recovered=recover_p(first).tolist(), db=first.db.tolist(), dmu=first.dmu,
                     first_ratings=[1,2], second_ratings=[2,3], bu_both=0,
                     max_message_difference=message_distance(first,second), attack_rejection=error_type,
                     centered_ratings_still_recovered=centered.tolist(),
                     eigenvalue_mean=4, mean_response=spectral_response(4,.5,2)))

    # Exact singular centered mode; identical mean and different rating contrasts.
    # c=2, a=1, k=2, eta=n/2=1 gives perpendicular multiplier -1.
    first, _, initial = oracle(public,p,0.,ids,np.array([1.,3.]),1.,2,'float64')
    second, _, _ = oracle(public,p,0.,ids,np.array([2.,2.]),1.,2,'float64')
    error_type = None
    try:
        recover_relative(initial,first,Metadata(2,2,1.))
    except ValueError as error:
        error_type = str(error)
    assert message_distance(first,second) == 0 and error_type == 'Singular residual-to-delta map'
    rows.append(dict(name='singular_centered_mode', n=2,d=2,eta=1.,steps=2,
                     first_ratings=[1,3], second_ratings=[2,2],bu_both=0,
                     max_message_difference=message_distance(first,second),
                     db=first.db.tolist(), attack_rejection=error_type,
                     eigenvalue_centered=2, centered_response=spectral_response(2,1.,2)))

    public = Public(np.array([[1.,0.],[2.,0.]]), np.zeros(2),0.)
    first,_,_=oracle(public,np.array([1.,0.]),0.,np.arange(2),np.array([1.,2.]),.1,3,'float64')
    try:
        recover_p(first)
        raise AssertionError('Zero-signal case must reject factor reconstruction')
    except ValueError as error:
        rows.append(dict(name='zero_residual', all_delta_zero=bool(max_abs(first.dq)==0 and max_abs(first.db)==0 and first.dmu==0),
                         attack_rejection=str(error)))

    # Cancellation hides all query rows, even with nonzero starting residual.
    public = Public(np.zeros((2,2)),np.zeros(2),0.)
    first,residuals,_=oracle(public,np.array([1.,0.]),0.,np.arange(2),np.array([2.,2.]),.5,2,'float64')
    assert max_abs(first.dq)==0 and max_abs(first.db)==0 and first.dmu==0
    rows.append(dict(name='nonzero_residual_full_cancellation', starting_residual=residuals[0].tolist(),
                     all_delta_zero=True, query_items_hidden=2))
    return rows


def conditioning_grid(config):
    rows=[]
    p=np.array([1.,0.])
    n=2; c=2.
    public=Public(np.zeros((2,2)),np.zeros(2),0.)
    ratings=np.array([1.,3.])
    for eps, dtype in itertools.product(config['singular_epsilon_values'], config['dtypes']):
        eta=.5*(1-eps)
        message,residuals,initial=oracle(public,p,0.,np.arange(n),ratings,eta,2,dtype)
        attacked=recover_relative(initial,message,Metadata(n,2,eta))
        hmean=spectral_response(c+n,2*eta/n,2)
        hperp=spectral_response(c,2*eta/n,2)
        perturb=np.ones(n)*(config['message_perturbation_l2']/np.sqrt(n))
        # Perturb both channels consistently; p stays exact. This isolates only
        # the residual inversion sensitivity, not additional error in p.
        perturbed=Message(message.dq+perturb[:,None]*p, message.db+perturb,
                          message.dmu+float(perturb.sum()))
        noisy=recover_relative(initial,perturbed,Metadata(n,2,eta))
        observed_gain=float(np.linalg.norm(noisy['continuous']-attacked['continuous'])/np.linalg.norm(perturb))
        rows.append(dict(epsilon=eps,dtype=dtype,eta=eta,n=n,steps=2,
                         h_mean=hmean,h_perp=hperp,condition=max(abs(hmean),abs(hperp))/min(abs(hmean),abs(hperp)),
                         expected_mean_inverse_gain=1/abs(hmean),observed_mean_inverse_gain=observed_gain,
                         error_without_perturbation=max_abs(attacked['continuous']-ratings),
                         error_with_perturbation=max_abs(noisy['continuous']-ratings),
                         perturbation_l2=config['message_perturbation_l2']))
    return rows


def write_csv(name, rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows):
    fields=['p_relative_l2','identity_max_abs','continuous_max_abs',
            'residual_recurrence_max_abs','independent_matrix_e0_max_abs',
            'global_bias_sum_error','inverse_condition','inverse_gain']
    result={}
    for dtype in ['float64','float32']:
        subset=[r for r in rows if r['dtype']==dtype]
        result[dtype]=dict(cases=len(subset),
                          relative_ratings_exact=sum(r['relative_ratings_exact'] for r in subset),
                          single_absolute_candidate=sum(r['absolute_candidates']==1 for r in subset),
                          **{f'max_{key}':max(r[key] for r in subset) for key in fields})
    return result


def main():
    start=time.perf_counter()
    config=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    full=full_batch_grid(config)
    mini=minibatch_grid(config)
    negative=counterexamples(config)
    conditioning=conditioning_grid(config)
    write_csv('full_batch_cases.csv',full)
    write_csv('minibatch_cases.csv',mini)
    write_csv('conditioning_cases.csv',conditioning)
    (HERE/'counterexamples.json').write_text(json.dumps(negative,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=dict(status='PASS',runtime_seconds=time.perf_counter()-start,
                 full_batch=summarize(full),
                 minibatch={dtype:dict(cases=sum(r['dtype']==dtype for r in mini),
                     max_p_relative_l2=max(r['p_relative_l2'] for r in mini if r['dtype']==dtype))
                     for dtype in ['float64','float32']},
                 counterexamples=len(negative), conditioning_cases=len(conditioning),
                 implementation_independence='Independent PyTorch autograd simulation and NumPy spectral and matrix-sum checks. The verification imports no simulator or attack code from the coursework project.',
                 python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
                 device='cpu',
                 sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                     for name in ['verify_math.py','config.json','full_batch_cases.csv','minibatch_cases.csv','conditioning_cases.csv','counterexamples.json']})
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
