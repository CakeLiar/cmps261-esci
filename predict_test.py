"""Predict the held-out test split with saved cross-encoder checkpoints (run once, for the final models only).
Also re-predicts validation as a check that the loaded checkpoint matches the saved validation predictions.
Usage: TAGS=minilm,deberta_small,deberta_base python predict_test.py"""
import os, json, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import *

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
va, te = pd.read_parquet(f'{D}/val.parquet'), pd.read_parquet(f'{D}/test.parquet')
out = json.load(open(f'{R}/test_predict_log.json')) if os.path.exists(f'{R}/test_predict_log.json') else {}

for tag in os.environ['TAGS'].split(','):
    cfg = json.load(open(f'{R}/{tag}.json'))['config']
    fields = cfg.get('FIELDS', ['title', 'brandcolor', 'bullets'])
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    model = AutoModelForSequenceClassification.from_pretrained(f'{ROOT}/models/{tag}', dtype=torch.float32).to(dev).eval()

    def second(d):
        s = f' {tok.sep_token} '
        parts = {'title': d.product_title, 'brandcolor': d.product_brand + ' ' + d.product_color,
                 'bullets': d.product_bullet_point.str.slice(0, 300), 'desc': d.product_description.str.slice(0, 300)}
        o = parts[fields[0]]
        for f in fields[1:]:
            o = o + s + parts[f]
        return o.tolist()

    @torch.no_grad()
    def predict(d, bs=512):
        E = tok(d.query_clean.tolist(), second(d), truncation='only_second', max_length=cfg['MAXLEN'],
                padding='max_length', return_tensors='pt')
        ps = []
        for i in range(0, len(d), bs):
            b = {k: v[i:i + bs].to(dev) for k, v in E.items()}
            with torch.amp.autocast('cuda', enabled=dev == 'cuda'):
                ps.append(F.softmax(model(**b).logits.float(), -1).cpu())
        return torch.cat(ps).numpy()

    pv = predict(va)
    saved = np.load(f'{R}/{tag}_val_proba.npy')
    agree = float((pv.argmax(1) == saved.argmax(1)).mean())
    t = time.time(); pt = predict(te); secs = time.time() - t
    np.save(f'{R}/{tag}_test_proba.npy', pt)
    out[tag] = {'val_argmax_agreement_with_saved': round(agree, 5), 'test_pairs': len(te),
                'test_infer_secs': round(secs, 1), 'test_pairs_per_sec': round(len(te) / secs)}
    print(tag, out[tag], flush=True)
    del model; torch.cuda.empty_cache()
save('test_predict_log', out)
