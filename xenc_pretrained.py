"""Fine-tune a pretrained cross-encoder for 4-way ESCI classification. Configured through env vars.

Defaults reproduce the runs in the progress report. Extra options (all optional):
  FIELDS  comma list of product fields for text B, from title,brandcolor,bullets,desc
          (default title,brandcolor,bullets)
  FRAC    fraction of training queries to keep, grouped by query (default 1.0 = all)
  CW      'balanced' to weight the loss by inverse class frequency (default: no weights)
  SAVE    1 to save the best checkpoint to models/{TAG} (default 1)
"""
import os, time, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
from common import *

MODEL = os.environ.get('MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
TAG = os.environ.get('TAG', 'minilm')
MAXLEN = int(os.environ.get('MAXLEN', 128)); EPOCHS = int(os.environ.get('EPOCHS', 2))
BS = int(os.environ.get('BS', 64)); LR = float(os.environ.get('LR', 3e-5))
EVAL_STEPS = int(os.environ.get('EVAL_STEPS', 2000)); SEED = int(os.environ.get('SEED', 0))
EVAL_BS = int(os.environ.get('EVAL_BS', 256))
FIELDS = os.environ.get('FIELDS', 'title,brandcolor,bullets').split(',')
FRAC = float(os.environ.get('FRAC', 1.0)); CW = os.environ.get('CW', '')
SAVE = os.environ.get('SAVE', '1') == '1'

set_seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
amp = dev == 'cuda'
print('device', dev, 'model', MODEL, 'fields', FIELDS, 'frac', FRAC, 'cw', CW or 'none', flush=True)

tr, va = pd.read_parquet(f'{D}/train.parquet'), pd.read_parquet(f'{D}/val.parquet')
if FRAC < 1:
    # Learning curve: keep a random subset of training queries (fixed seed 0, independent of SEED).
    q = tr.query_id.unique()
    keep = set(np.random.RandomState(0).choice(q, size=int(FRAC * len(q)), replace=False))
    tr = tr[tr.query_id.isin(keep)].reset_index(drop=True)
print('train pairs', len(tr), flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)

def second(d):
    """Text B: the chosen product fields joined by the separator token."""
    s = f' {tok.sep_token} '
    parts = {'title': d.product_title, 'brandcolor': d.product_brand + ' ' + d.product_color,
             'bullets': d.product_bullet_point.str.slice(0, 300), 'desc': d.product_description.str.slice(0, 300)}
    out = parts[FIELDS[0]]
    for f in FIELDS[1:]:
        out = out + s + parts[f]
    return out.tolist()

def encode(d):
    return tok(d.query_clean.tolist(), second(d), truncation='only_second', max_length=MAXLEN, padding='max_length', return_tensors='pt')

t0 = time.time()
Etr, Eva = encode(tr), encode(va)
ytr, yva = torch.tensor(tr.y.values), va.y.values
print('encoded', Etr['input_ids'].shape, round(time.time() - t0), 's', flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4, ignore_mismatched_sizes=True, dtype=torch.float32).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print('params', n_params, flush=True)

weight = None
if CW == 'balanced':
    cnt = np.bincount(tr.y.values, minlength=4)
    weight = torch.tensor(len(tr) / (4 * cnt), dtype=torch.float32, device=dev)
    print('class weights', weight.tolist(), flush=True)

steps = EPOCHS * math.ceil(len(ytr) / BS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler('cuda', enabled=amp)

@torch.no_grad()
def predict():
    model.eval(); out = []
    for i in range(0, len(yva), EVAL_BS):
        b = {k: v[i:i + EVAL_BS].to(dev) for k, v in Eva.items()}
        with torch.amp.autocast('cuda', enabled=amp):
            out.append(F.softmax(model(**b).logits.float(), -1).cpu())
    model.train(); return torch.cat(out).numpy()

config = dict(model=MODEL, MAXLEN=MAXLEN, EPOCHS=EPOCHS, BS=BS, LR=LR, SEED=SEED, warmup=0.05, weight_decay=0.01,
              FIELDS=FIELDS, FRAC=FRAC, CW=CW or None, train_pairs=len(tr),
              params=n_params, device=dev, transformers=transformers.__version__, torch=torch.__version__,
              gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None)
log = []; step = 0; best = -1; t_train = time.time()

def evaluate(ep):
    global best
    t = time.time(); p = predict(); infer = time.time() - t
    r = metrics(yva, p.argmax(1))
    r.update(step=step, epoch=ep, secs=round(time.time() - t_train), val_infer_secs=round(infer, 1))
    if dev == 'cuda':
        r['peak_gpu_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    log.append(r); print('VAL', step, r['micro_f1'], r['macro_f1'], r['per_class_f1'], flush=True)
    if r['micro_f1'] > best:
        best = r['micro_f1']; np.save(f'{R}/{TAG}_val_proba.npy', p)
        if SAVE:
            model.save_pretrained(f'{ROOT}/models/{TAG}')
    save(TAG, {'config': config, 'log': log, 'best': max(log, key=lambda r: r['micro_f1'])})

model.train()
for ep in range(EPOCHS):
    perm = torch.randperm(len(ytr)); run = 0; t = time.time()
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        b = {k: v[idx].to(dev) for k, v in Etr.items()}
        with torch.amp.autocast('cuda', enabled=amp):
            loss = F.cross_entropy(model(**b).logits, ytr[idx].to(dev), weight=weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; run += loss.item()
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{steps} loss {run / 200:.4f} {time.time() - t:.0f}s', flush=True); run = 0
        if step % EVAL_STEPS == 0 or step == steps:
            evaluate(ep)
print('done', round(time.time() - t_train), 's', flush=True)
