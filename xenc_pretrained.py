"""Fine-tune a pretrained cross-encoder for 4-way ESCI classification. Configured through env vars."""
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

set_seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
amp = dev == 'cuda'
print('device', dev, 'model', MODEL, flush=True)

tr, va = pd.read_parquet(f'{D}/train.parquet'), pd.read_parquet(f'{D}/val.parquet')
tok = AutoTokenizer.from_pretrained(MODEL)

def second(d):
    s = f' {tok.sep_token} '
    return (d.product_title + s + d.product_brand + ' ' + d.product_color + s + d.product_bullet_point.str.slice(0, 300)).tolist()

def encode(d):
    return tok(d.query_clean.tolist(), second(d), truncation='only_second', max_length=MAXLEN, padding='max_length', return_tensors='pt')

t0 = time.time()
Etr, Eva = encode(tr), encode(va)
ytr, yva = torch.tensor(tr.y.values), va.y.values
print('encoded', Etr['input_ids'].shape, round(time.time() - t0), 's', flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=4, ignore_mismatched_sizes=True, dtype=torch.float32).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print('params', n_params, flush=True)

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
              params=n_params, device=dev, transformers=transformers.__version__, torch=torch.__version__)
log = []; step = 0; best = -1; t_train = time.time()

def evaluate(ep):
    global best
    p = predict(); r = metrics(yva, p.argmax(1)); r.update(step=step, epoch=ep, secs=round(time.time() - t_train))
    log.append(r); print('VAL', step, r['micro_f1'], r['macro_f1'], r['per_class_f1'], flush=True)
    if r['micro_f1'] > best:
        best = r['micro_f1']; np.save(f'{R}/{TAG}_val_proba.npy', p); model.save_pretrained(f'{ROOT}/models/{TAG}')
    save(TAG, {'config': config, 'log': log, 'best': max(log, key=lambda r: r['micro_f1'])})

model.train()
for ep in range(EPOCHS):
    perm = torch.randperm(len(ytr)); run = 0; t = time.time()
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        b = {k: v[idx].to(dev) for k, v in Etr.items()}
        with torch.amp.autocast('cuda', enabled=amp):
            loss = F.cross_entropy(model(**b).logits, ytr[idx].to(dev))
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
