"""Paired bootstrap over queries for the comparisons the report makes (CPU, a few seconds).

Resamples whole queries with replacement (2,000 resamples, seed 0), so pairs of the same query stay together.
For each resample it recomputes micro-F1 and macro-F1 of both models on the same resampled pairs and records
the difference. Reports the observed difference and the 2.5/97.5 percentiles. Writes results/significance.json.
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
D, R = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'results')
B, SEED = 2000, 0

def per_query_confusion(qid, y, pred):
    """(n_queries, 16) counts of (true, pred) cells per query."""
    q = pd.factorize(qid)[0]
    out = np.zeros((q.max() + 1, 16))
    np.add.at(out, (q, y * 4 + pred), 1)
    return out

def scores(cm16):
    """cm16: (..., 16) confusion counts -> micro-F1, macro-F1 arrays."""
    cm = cm16.reshape(*cm16.shape[:-1], 4, 4)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return tp.sum(-1) / cm.sum((-2, -1)), f1.mean(-1)

def load(split):
    return pd.read_parquet(f'{D}/{split}.parquet', columns=['query_id', 'y'])

def pred(name):
    return np.load(f'{R}/{name}.npy').argmax(1)

def compare(split, a, b, weights):
    df = load(split)
    ca = per_query_confusion(df.query_id.values, df.y.values, pred(a))
    cb = per_query_confusion(df.query_id.values, df.y.values, pred(b))
    obs_a, obs_b = scores(ca.sum(0)), scores(cb.sum(0))
    ra, rb = scores(weights[split] @ ca), scores(weights[split] @ cb)
    out = {'split': split, 'a': a, 'b': b}
    for i, m in enumerate(('micro_f1', 'macro_f1')):
        d = ra[i] - rb[i]
        out[m] = {'a': float(obs_a[i]), 'b': float(obs_b[i]), 'diff': float(obs_a[i] - obs_b[i]),
                  'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out

def single(split, a, weights):
    df = load(split)
    c = per_query_confusion(df.query_id.values, df.y.values, pred(a))
    obs, r = scores(c.sum(0)), scores(weights[split] @ c)
    return {'split': split, 'model': a, **{m: {'value': float(obs[i]), 'ci95': [float(np.percentile(r[i], 2.5)), float(np.percentile(r[i], 97.5))]}
                                         for i, m in enumerate(('micro_f1', 'macro_f1'))}}

if __name__ == '__main__':
    rng = np.random.RandomState(SEED)
    weights = {}
    for split in ('val', 'test'):
        nq = load(split).query_id.nunique()
        weights[split] = rng.multinomial(nq, np.full(nq, 1 / nq), size=B).astype(float)
    res = {'resamples': B, 'seed': SEED, 'unit': 'query', 'comparisons': {
        'test_deberta_base_vs_deberta_small': compare('test', 'deberta_base_test_proba', 'deberta_small_test_proba', weights),
        'test_deberta_small_vs_minilm': compare('test', 'deberta_small_test_proba', 'minilm_test_proba', weights),
        'val_deberta_base_vs_hybrid_lex+ce_all': compare('val', 'deberta_base_val_proba', 'hybrid_lex+ce_all_val_oof', weights),
        'val_hybrid_lex+ce_all_vs_ce_all_only': compare('val', 'hybrid_lex+ce_all_val_oof', 'hybrid_ce_all_only_val_oof', weights),
        'val_minilm_vs_minilm_f_title': compare('val', 'minilm_val_proba', 'minilm_f_title_val_proba', weights),
        'val_minilm_seed1_vs_minilm': compare('val', 'minilm_seed1_val_proba', 'minilm_val_proba', weights),
    }, 'single': {
        'test_deberta_base': single('test', 'deberta_base_test_proba', weights),
        'val_deberta_base': single('val', 'deberta_base_val_proba', weights),
    }}
    json.dump(res, open(f'{R}/significance.json', 'w'), indent=1)
    for k, v in res['comparisons'].items():
        print(k, {m: (round(v[m]['diff'], 4), [round(x, 4) for x in v[m]['ci95']]) for m in ('micro_f1', 'macro_f1')})
    for k, v in res['single'].items():
        print(k, {m: (round(v[m]['value'], 4), [round(x, 4) for x in v[m]['ci95']]) for m in ('micro_f1', 'macro_f1')})
