"""Classical baselines, tuned on the query-grouped validation split."""
import time, itertools, sys, os
import numpy as np, pandas as pd, scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from common import *

tr, va = pd.read_parquet(f'{D}/train.parquet'), pd.read_parquet(f'{D}/val.parquet')
ytr, yva = tr.y.values, va.y.values
Ftr, Fva = pd.read_parquet(f'{D}/feat_train.parquet'), pd.read_parquet(f'{D}/feat_val.parquet')
out = {'majority': metrics(yva, np.zeros_like(yva))}
print('majority', out['majority']['micro_f1'], out['majority']['macro_f1'], flush=True)

# --- LightGBM on hand-crafted lexical features
lgb_grid = []
for nl, mcs, cw in itertools.product([31, 127], [50, 200], [None, 'balanced']):
    t = time.time()
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=nl, min_child_samples=mcs,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight=cw,
                           n_jobs=-1, verbose=-1)
    m.fit(Ftr, ytr, eval_set=[(Fva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    r = metrics(yva, m.predict(Fva))
    r.update(params={'num_leaves': nl, 'min_child_samples': mcs, 'class_weight': cw}, best_iter=int(m.best_iteration_),
             secs=round(time.time() - t))
    lgb_grid.append(r); print('lgb', r['params'], r['best_iter'], r['micro_f1'], r['macro_f1'], flush=True)
    if r is max(lgb_grid, key=lambda r: r['micro_f1']):
        np.save(f'{R}/lgb_val_proba.npy', m.predict_proba(Fva))
        gain = m.booster_.feature_importance(importance_type='gain')
        out['best_lgb_importance_gain'] = sorted(zip(Ftr.columns.tolist(), [float(g) for g in gain]), key=lambda kv: -kv[1])
    save('baseline_partial_full', {**out, 'lgb_grid': lgb_grid})
best_lgb = max(lgb_grid, key=lambda r: r['micro_f1'])
out['lgb_grid'] = lgb_grid

# --- Logistic regression on TF-IDF (query, title, query*title word & char) + scaled lexical features
Xtr, Xva = sp.load_npz(f'{D}/tfidf_train.npz'), sp.load_npz(f'{D}/tfidf_val.npz')
sc = StandardScaler().fit(Ftr)
Xtr = sp.hstack([Xtr, sp.csr_matrix(sc.transform(Ftr).astype(np.float32))]).tocsr()
Xva = sp.hstack([Xva, sp.csr_matrix(sc.transform(Fva).astype(np.float32))]).tocsr()
def fit_lr(Xtr, ytr, Xva, yva, C, cw, max_iter=300):
    t = time.time()
    m = LogisticRegression(C=C, class_weight=cw, max_iter=max_iter, solver='lbfgs')
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)
    r = metrics(yva, p.argmax(1))
    n_iter = int(np.max(m.n_iter_))
    r.update(params={'C': C, 'class_weight': cw}, n_iter=n_iter, converged=n_iter < max_iter, secs=round(time.time() - t))
    print('lr', r['params'], r['n_iter'], r['micro_f1'], r['macro_f1'], r['secs'], flush=True)
    return r, p

# lbfgs is effectively single-threaded here, so the 8 settings run as separate processes.
res = Parallel(n_jobs=int(os.environ.get('LR_JOBS', 8)), backend='loky')(
    delayed(fit_lr)(Xtr, ytr, Xva, yva, C, cw) for C, cw in itertools.product([0.1, 0.3, 1.0, 3.0], [None, 'balanced']))
lr_grid = [r for r, _ in res]
np.save(f'{R}/lr_val_proba.npy', max(res, key=lambda rp: rp[0]['micro_f1'])[1])
out['lr_grid'] = lr_grid
out['best_lgb'] = best_lgb
out['best_lr'] = max(lr_grid, key=lambda r: r['micro_f1'])
save('baseline', out)
