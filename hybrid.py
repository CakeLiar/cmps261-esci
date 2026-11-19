"""Hybrid model: LightGBM stacked on cross-encoder probabilities plus the 33 lexical features.
The cross-encoders never saw validation labels in training, so the stacker is trained on the validation split.
We estimate its quality with 5-fold cross-validation grouped by query (out-of-fold predictions), and compare
with the cross-encoders alone on exactly the same pairs. The test split is not used here."""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from common import *

va = pd.read_parquet(f'{D}/val.parquet'); Fva = pd.read_parquet(f'{D}/feat_val.parquet'); y = va.y.values
CE = [t for t in ('minilm', 'deberta_small', 'deberta_base') if os.path.exists(f'{R}/{t}_val_proba.npy')]
P = {t: np.load(f'{R}/{t}_val_proba.npy') for t in CE}
ce_scores = {t: json.load(open(f'{R}/{t}.json'))['best']['micro_f1'] for t in CE}
best_ce = max(CE, key=ce_scores.get)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)

def design(variant, F, probs):
    cols = {'lex': F.reset_index(drop=True)}
    use = {'ce_best': [best_ce], 'ce_all': CE}[variant['ce']]
    for t in use:
        cols[t] = pd.DataFrame(probs[t], columns=[f'{t}_{l}' for l in LAB])
    parts = [cols[t] for t in use] + ([cols['lex']] if variant['lex'] else [])
    return pd.concat(parts, axis=1)

VARIANTS = {
    'ce_all_only': {'ce': 'ce_all', 'lex': False, 'cw': None},
    'lex+ce_best': {'ce': 'ce_best', 'lex': True, 'cw': None},
    'lex+ce_all': {'ce': 'ce_all', 'lex': True, 'cw': None},
    'lex+ce_all_balanced': {'ce': 'ce_all', 'lex': True, 'cw': 'balanced'},
}
folds = list(GroupKFold(n_splits=5).split(va, y, groups=va.query_id.values))
out = {'cross_encoders': CE, 'best_cross_encoder': best_ce, 'params': PARAMS, 'variants': {},
       'cross_encoder_alone': {t: metrics(y, P[t].argmax(1)) for t in CE}}
for name, v in VARIANTS.items():
    X = design(v, Fva, P); oof = np.zeros((len(y), 4))
    for tr_i, te_i in folds:
        m = lgb.LGBMClassifier(class_weight=v['cw'], **PARAMS).fit(X.iloc[tr_i], y[tr_i])
        oof[te_i] = m.predict_proba(X.iloc[te_i])
    r = metrics(y, oof.argmax(1)); r['variant'] = v
    out['variants'][name] = r
    np.save(f'{R}/hybrid_{name}_val_oof.npy', oof)
    print(name, r['micro_f1'], r['macro_f1'], r['per_class_f1'], flush=True)
out['selected'] = max(out['variants'], key=lambda k: out['variants'][k]['micro_f1'])
save('hybrid', out)
print('selected', out['selected'])
