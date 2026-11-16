"""Slice validation predictions to see where models fail. Every threshold is fixed here, before looking at results."""
import re, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from common import *

HIGH_OV, LOW_OV = 0.75, 0.25
TOK = re.compile(r'[a-z0-9]+(?:[.\-][a-z0-9]+)*')
va = pd.read_parquet(f'{D}/val.parquet'); F = pd.read_parquet(f'{D}/feat_val.parquet')
tr = pd.read_parquet(f'{D}/train.parquet', columns=['product_id', 'product_title', 'product_bullet_point'])
y = va.y.values

# A likely misspelling: an alphabetic query word of 4+ letters that appears in no training title or bullet text.
prod = tr.drop_duplicates('product_id')
vocab = set()
for col in ('product_title', 'product_bullet_point'):
    for s in prod[col].values:
        vocab.update(TOK.findall(s))
misspelled = np.array([any(w.isalpha() and len(w) >= 4 and w not in vocab for w in TOK.findall(q)) for q in va.query_clean.values])

base = load = __import__('json').load(open(f'{R}/baseline.json'))
lgb_micro, lr_micro = base['best_lgb']['micro_f1'], base['best_lr']['micro_f1']
preds = {'best_baseline': np.load(f'{R}/{"lgb" if lgb_micro >= lr_micro else "lr"}_val_proba.npy').argmax(1)}
for name in ('minilm', 'deberta_small', 'deberta_base'):
    path = f'{R}/{name}_val_proba.npy'
    if os.path.exists(path):
        preds[name] = np.load(path).argmax(1)

slices = {
    'all': np.ones_like(y, dtype=bool),
    'high_overlap_labelled_I_or_C': (F.ov_title.values >= HIGH_OV) & np.isin(y, [2, 3]),
    'low_overlap_labelled_E': (F.ov_title.values <= LOW_OV) & (y == 0),
    'likely_misspelled_query': misspelled,
    'query_has_number': F.q_num.values > 0,
    'query_has_negation': F.q_neg.values > 0,
}
out = {'thresholds': {'high_overlap_ov_title_ge': HIGH_OV, 'low_overlap_ov_title_le': LOW_OV,
                      'misspelled': 'alphabetic query token, 4+ letters, absent from train titles and bullets'},
       'best_baseline_source': 'lgb' if lgb_micro >= lr_micro else 'lr', 'slices': {}}
for name, mask in slices.items():
    out['slices'][name] = {'n_pairs': int(mask.sum()), 'share': round(float(mask.mean()), 4), 'n_queries': int(va.query_id[mask].nunique()),
                           'label_counts': dict(zip(LAB, np.bincount(y[mask], minlength=4).tolist()))}
    for m, p in preds.items():
        s = {'accuracy': round(float(accuracy_score(y[mask], p[mask])), 4),
             'pred_counts': dict(zip(LAB, np.bincount(p[mask], minlength=4).tolist()))}
        if name in ('all', 'likely_misspelled_query', 'query_has_number', 'query_has_negation'):
            k = metrics(y[mask], p[mask]); s.update(micro_f1=k['micro_f1'], macro_f1=k['macro_f1'], per_class_f1=k['per_class_f1'])
        out['slices'][name][m] = s

comp = y == 2
out['complement'] = {m: {'n': int(comp.sum()), 'pred_counts': dict(zip(LAB, np.bincount(p[comp], minlength=4).tolist())),
                         'recall': round(float((p[comp] == 2).mean()), 4)} for m, p in preds.items()}
save('error_analysis', out)
print(__import__('json').dumps(out, indent=1))
