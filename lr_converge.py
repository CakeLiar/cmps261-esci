"""Rerun the logistic-regression grid from baseline.py with a higher iteration limit, so the fits converge.
Same features and grid as baseline.py; only max_iter changes (300 -> MAX_ITER, default 3000)."""
import time, itertools, os
import numpy as np, pandas as pd, scipy.sparse as sp, joblib
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *

MAX_ITER = int(os.environ.get('MAX_ITER', 3000))
tr, va = pd.read_parquet(f'{D}/train.parquet'), pd.read_parquet(f'{D}/val.parquet')
ytr, yva = tr.y.values, va.y.values
Ftr, Fva = pd.read_parquet(f'{D}/feat_train.parquet'), pd.read_parquet(f'{D}/feat_val.parquet')
Xtr, Xva = sp.load_npz(f'{D}/tfidf_train.npz'), sp.load_npz(f'{D}/tfidf_val.npz')
sc = StandardScaler().fit(Ftr)
Xtr = sp.hstack([Xtr, sp.csr_matrix(sc.transform(Ftr).astype(np.float32))]).tocsr()
Xva = sp.hstack([Xva, sp.csr_matrix(sc.transform(Fva).astype(np.float32))]).tocsr()

def fit_lr(C, cw):
    t = time.time()
    m = LogisticRegression(C=C, class_weight=cw, max_iter=MAX_ITER, solver='lbfgs')
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)
    r = metrics(yva, p.argmax(1))
    n_iter = int(np.max(m.n_iter_))
    r.update(params={'C': C, 'class_weight': cw}, n_iter=n_iter, converged=n_iter < MAX_ITER, secs=round(time.time() - t))
    print('lr', r['params'], r['n_iter'], r['micro_f1'], r['macro_f1'], r['secs'], flush=True)
    return r, p, m

res = Parallel(n_jobs=int(os.environ.get('LR_JOBS', 8)), backend='loky')(
    delayed(fit_lr)(C, cw) for C, cw in itertools.product([0.1, 0.3, 1.0, 3.0], [None, 'balanced']))
best_r, best_p, best_m = max(res, key=lambda x: x[0]['micro_f1'])
np.save(f'{R}/lr_conv_val_proba.npy', best_p)
os.makedirs(f'{ROOT}/models', exist_ok=True)
joblib.dump({'scaler': sc, 'model': best_m}, f'{ROOT}/models/lr_best.joblib')
save('lr_converged', {'max_iter': MAX_ITER, 'lr_grid': [r for r, _, _ in res], 'best_lr': best_r})
