"""Preprocessing for ESCI (US locale, reduced 'small_version' set).
Outputs train/val/test parquet files with cleaned text fields."""
import re, html, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

import os
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'raw')
OUT = os.path.join(ROOT, 'data')
import os; os.makedirs(OUT, exist_ok=True)

ex = pd.read_parquet(f'{RAW}/examples.parquet')
ex = ex[(ex.product_locale == 'us') & (ex.small_version == 1)].copy()
stats = {'pairs_raw': len(ex)}

tbl = pq.read_table(f'{RAW}/products.parquet',
                    columns=['product_id', 'product_title', 'product_description',
                             'product_bullet_point', 'product_brand', 'product_color', 'product_locale'],
                    filters=[('product_locale', '=', 'us')])
prod = tbl.to_pandas().drop(columns='product_locale')
prod = prod[prod.product_id.isin(set(ex.product_id))]

TAG = re.compile(r'<[^>]+>')
WS = re.compile(r'\s+')
def clean(s, maxlen=None):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = html.unescape(str(s))
    s = TAG.sub(' ', s)
    s = s.replace(' ', ' ')
    s = WS.sub(' ', s).strip().lower()
    if maxlen:
        s = s[:maxlen]
    return s

missing = {c: float((prod[c].fillna('').str.strip() == '').mean())
           for c in ['product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']}
for c, m in [('product_title', None), ('product_brand', None), ('product_color', None),
             ('product_bullet_point', 1000), ('product_description', 1000)]:
    prod[c] = [clean(x, m) for x in prod[c].values]

df = ex.merge(prod, on='product_id', how='left')
for c in ['product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']:
    df[c] = df[c].fillna('')
df['query_clean'] = [clean(q) for q in df['query'].values]
stats['pairs_missing_product'] = int((df.product_title == '').sum())
stats['dup_pairs'] = int(df.duplicated(['query_id', 'product_id']).sum())
df = df.drop_duplicates(['query_id', 'product_id'])

# Query-grouped validation split (10% of train queries), official test held out.
rng = np.random.RandomState(42)
tr_q = df.loc[df.split == 'train', 'query_id'].unique()
val_q = set(rng.choice(tr_q, size=int(0.1 * len(tr_q)), replace=False))
df['part'] = np.where(df.split == 'test', 'test', np.where(df.query_id.isin(val_q), 'val', 'train'))

LABELS = {'E': 0, 'S': 1, 'C': 2, 'I': 3}
df['y'] = df.esci_label.map(LABELS)
keep = ['example_id', 'query_id', 'query', 'query_clean', 'product_id', 'product_title', 'product_brand',
        'product_color', 'product_bullet_point', 'product_description', 'esci_label', 'y', 'part']
for p in ['train', 'val', 'test']:
    d = df[df.part == p][keep]
    d.to_parquet(f'{OUT}/{p}.parquet', index=False)
    stats[p] = {'pairs': len(d), 'queries': int(d.query_id.nunique()),
                'label_dist': d.esci_label.value_counts(normalize=True).round(4).to_dict()}
stats['missing_rate_products'] = {k: round(v, 4) for k, v in missing.items()}
stats['n_products'] = len(prod)
stats['query_len_words_mean'] = float(df.query_clean.str.split().str.len().mean())
stats['title_len_words_mean'] = float(df.product_title.str.split().str.len().mean())
json.dump(stats, open(f'{OUT}/prep_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
