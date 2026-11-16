"""Lexical query-product features + sparse TF-IDF matrices for the classical baseline."""
import re, math, json, pickle
from collections import Counter
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
parts = {p: pd.read_parquet(f'{D}/{p}.parquet') for p in ['train', 'val', 'test']}
TOK = re.compile(r'[a-z0-9]+(?:[.\-][a-z0-9]+)*')
NUM = re.compile(r'\d')
NEG = {'not', 'without', 'no', 'non', 'free', 'except'}
COMP_WORDS = {'for', 'compatible', 'case', 'cover', 'replacement', 'charger', 'cable', 'adapter',
              'protector', 'accessory', 'accessories', 'kit', 'refill', 'holder', 'mount', 'stand', 'strap'}

def toks(s):
    return TOK.findall(s)

# document frequency over product titles in train (for IDF-weighted overlap / BM25)
tr_titles = parts['train'].drop_duplicates('product_id').product_title.values
df_cnt = Counter()
for t in tr_titles:
    df_cnt.update(set(toks(t)))
N = len(tr_titles)
def idf(w):
    return math.log((N - df_cnt.get(w, 0) + 0.5) / (df_cnt.get(w, 0) + 0.5) + 1)
avg_tl = np.mean([len(toks(t)) for t in tr_titles])

def char_ngrams(s, n=3):
    s = f' {s} '
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def bm25(q, d, k1=1.2, b=0.75):
    tf = Counter(d); L = len(d); sc = 0.0
    for w in set(q):
        f = tf.get(w, 0)
        if f:
            sc += idf(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg_tl))
    return sc

def row(q, title, brand, color, bullet, desc):
    qt, tt, bt, dt = toks(q), toks(title), toks(bullet), toks(desc)
    qs, ts, bs, ds = set(qt), set(tt), set(bt), set(dt)
    brs, cs = set(toks(brand)), set(toks(color))
    nq = max(len(qs), 1)
    qidf = sum(idf(w) for w in qs) or 1.0
    qnum = {w for w in qs if NUM.search(w)}
    qc, tc = char_ngrams(q), char_ngrams(title)
    miss = qs - ts - bs - brs
    f = {
        'q_len': len(qt), 't_len': len(tt), 'b_len': len(bt), 'd_len': len(dt),
        'has_desc': int(bool(dt)), 'has_bullet': int(bool(bt)), 'has_brand': int(bool(brs)),
        'ov_title': len(qs & ts) / nq, 'ov_bullet': len(qs & bs) / nq, 'ov_desc': len(qs & ds) / nq,
        'ov_brand': len(qs & brs) / nq, 'ov_color': len(qs & cs) / nq,
        'ov_any': len(qs & (ts | bs | ds | brs | cs)) / nq,
        'idf_ov_title': sum(idf(w) for w in qs & ts) / qidf,
        'idf_ov_any': sum(idf(w) for w in qs & (ts | bs | ds | brs)) / qidf,
        'jacc_title': len(qs & ts) / max(len(qs | ts), 1),
        'char_jacc': len(qc & tc) / max(len(qc | tc), 1),
        'char_cov': len(qc & tc) / max(len(qc), 1),
        'bm25_title': bm25(qt, tt), 'bm25_bullet': bm25(qt, bt), 'bm25_desc': bm25(qt, dt),
        'q_num': len(qnum), 'num_match': len(qnum & (ts | bs)) / max(len(qnum), 1) if qnum else -1,
        'brand_in_q': int(bool(brs) and brs <= qs),
        'max_missing_idf': max([idf(w) for w in miss], default=0.0),
        'n_missing': len(miss),
        'q_neg': int(bool(qs & NEG)),
        't_comp_words': len(ts & COMP_WORDS), 'q_comp_words': len(qs & COMP_WORDS),
        'title_for_after_q': int(' for ' in title and any(w in title.split(' for ', 1)[1] for w in qs)),
        'first_q_in_title': int(bool(qt) and qt[0] in ts), 'last_q_in_title': int(bool(qt) and qt[-1] in ts),
        'q_mean_idf': qidf / nq,
    }
    return f

for p, d in parts.items():
    F = pd.DataFrame([row(*r) for r in zip(d.query_clean.values, d.product_title.values, d.product_brand.values,
                                          d.product_color.values, d.product_bullet_point.values,
                                          d.product_description.values)])
    F.to_parquet(f'{D}/feat_{p}.parquet', index=False)
    print(p, F.shape)

# Sparse text representation: TF-IDF of query, of title, and their element-wise product (shared terms)
vec_w = TfidfVectorizer(token_pattern=r'[a-z0-9]+', ngram_range=(1, 2), min_df=3, max_features=300000,
                        sublinear_tf=True, dtype=np.float32)
vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=5, max_features=200000,
                        sublinear_tf=True, dtype=np.float32)
tr = parts['train']
vec_w.fit(pd.concat([tr.query_clean, tr.drop_duplicates('product_id').product_title]))
vec_c.fit(pd.concat([tr.query_clean, tr.drop_duplicates('product_id').product_title]))
for p, d in parts.items():
    Qw, Tw = vec_w.transform(d.query_clean), vec_w.transform(d.product_title)
    Qc, Tc = vec_c.transform(d.query_clean), vec_c.transform(d.product_title)
    X = sp.hstack([Qw, Tw, Qw.multiply(Tw), Qc.multiply(Tc)]).tocsr().astype(np.float32)
    sp.save_npz(f'{D}/tfidf_{p}.npz', X)
    print(p, X.shape, X.nnz)
