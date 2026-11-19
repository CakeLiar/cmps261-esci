"""Final, one-time evaluation on the held-out test split.
Models were selected on validation before this script was run. Writes results/test.json."""
import os, re, json, time
import numpy as np, pandas as pd, scipy.sparse as sp, joblib, lightgbm as lgb
from sklearn.metrics import accuracy_score
from common import *
import hybrid as H   # re-runs the validation-only hybrid CV and gives the selected variant and its design

tr, va, te = (pd.read_parquet(f'{D}/{p}.parquet') for p in ('train', 'val', 'test'))
Ftr, Fva, Fte = (pd.read_parquet(f'{D}/feat_{p}.parquet') for p in ('train', 'val', 'test'))
ytr, yva, yte = tr.y.values, va.y.values, te.y.values
out = {'test_pairs': len(te), 'test_queries': int(te.query_id.nunique()), 'models': {}, 'val': {}}
pred = {}

# Majority class
pred['majority'] = np.zeros_like(yte)

# LightGBM: refit the selected baseline setting exactly as in baseline.py, then predict test once.
base = json.load(open(f'{R}/baseline.json'))['best_lgb']['params']
t = time.time()
m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=base['num_leaves'], min_child_samples=base['min_child_samples'],
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight=base['class_weight'], n_jobs=-1, verbose=-1)
m.fit(Ftr, ytr, eval_set=[(Fva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
out['val']['lightgbm_refit'] = dict(metrics(yva, m.predict(Fva)), best_iter=int(m.best_iteration_), fit_secs=round(time.time() - t))
m.booster_.save_model(f'{ROOT}/models/lgb_best.txt')
t = time.time(); pred['lightgbm'] = m.predict(Fte); out['val']['lightgbm_refit']['test_infer_secs'] = round(time.time() - t, 1)

# Logistic regression (converged rerun)
if os.path.exists(f'{ROOT}/models/lr_best.joblib'):
    lr = joblib.load(f'{ROOT}/models/lr_best.joblib')
    Xte = sp.hstack([sp.load_npz(f'{D}/tfidf_test.npz'), sp.csr_matrix(lr['scaler'].transform(Fte).astype(np.float32))]).tocsr()
    pred['logistic_regression'] = lr['model'].predict(Xte)

# Cross-encoders
P_te = {}
for tag in H.CE:
    path = f'{R}/{tag}_test_proba.npy'
    if os.path.exists(path):
        P_te[tag] = np.load(path); pred[tag] = P_te[tag].argmax(1)

# Hybrid: fit the variant selected by validation CV on all validation pairs, predict test.
sel = H.out['selected']; v = H.VARIANTS[sel]
hm = lgb.LGBMClassifier(class_weight=v['cw'], **H.PARAMS).fit(H.design(v, Fva, H.P), yva)
pred[f'hybrid_{sel}'] = hm.predict(H.design(v, Fte, P_te)).astype(int)
hm.booster_.save_model(f'{ROOT}/models/hybrid_lgb.txt')
out['hybrid_selected'] = sel

# Final model = highest validation micro-F1 (hybrid scored out-of-fold), decided without test labels.
val_scores = {'lightgbm': json.load(open(f'{R}/baseline.json'))['best_lgb']['micro_f1'],
              f'hybrid_{sel}': H.out['variants'][sel]['micro_f1']}
if 'logistic_regression' in pred:
    val_scores['logistic_regression'] = json.load(open(f'{R}/lr_converged.json'))['best_lr']['micro_f1']
for tag in P_te:
    val_scores[tag] = json.load(open(f'{R}/{tag}.json'))['best']['micro_f1']
out['val_micro_used_for_selection'] = val_scores
out['final_model'] = max(val_scores, key=val_scores.get)
print('final model (chosen on validation):', out['final_model'], flush=True)

for k, p in pred.items():
    out['models'][k] = metrics(yte, p)
    print(k, out['models'][k]['micro_f1'], out['models'][k]['macro_f1'], out['models'][k]['per_class_f1'], flush=True)

# Failure slices on test, same definitions as error_analysis.py
TOK = re.compile(r'[a-z0-9]+(?:[.\-][a-z0-9]+)*')
vocab = set()
for col in ('product_title', 'product_bullet_point'):
    for s in tr.drop_duplicates('product_id')[col].values:
        vocab.update(TOK.findall(s))
mis = np.array([any(w.isalpha() and len(w) >= 4 and w not in vocab for w in TOK.findall(q)) for q in te.query_clean.values])
slices = {'all': np.ones_like(yte, dtype=bool),
          'high_overlap_labelled_I_or_C': (Fte.ov_title.values >= 0.75) & np.isin(yte, [2, 3]),
          'low_overlap_labelled_E': (Fte.ov_title.values <= 0.25) & (yte == 0),
          'likely_misspelled_query': mis, 'query_has_number': Fte.q_num.values > 0, 'query_has_negation': Fte.q_neg.values > 0}
out['slices'] = {name: dict(n_pairs=int(mask.sum()), share=round(float(mask.mean()), 4),
                            **{k: round(float(accuracy_score(yte[mask], p[mask])), 4) for k, p in pred.items()})
                 for name, mask in slices.items()}
save('test', out)
