"""Cross-encoder transformer trained from scratch (no pretrained weights reachable in this environment).
Input: [CLS] query [SEP] title [SEP] brand | color [SEP] bullet points (truncated)."""
import time, math, sys, os, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
from common import *

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count())
MAXLEN = int(os.environ.get('MAXLEN', 96)); EPOCHS = int(os.environ.get('EPOCHS', 2))
BS = 128; LR = 5e-4; VOCAB = 16000; HEADS = 4
D_MODEL = int(os.environ.get('D_MODEL', 128)); LAYERS = int(os.environ.get('LAYERS', 2))

tr, va = pd.read_parquet(f'{D}/train.parquet'), pd.read_parquet(f'{D}/val.parquet')

def second(d):
    return (d.product_title + ' [SEP] ' + d.product_brand + ' ' + d.product_color + ' [SEP] ' +
            d.product_bullet_point.str.slice(0, 300)).tolist()

tok_path = f'{R}/wordpiece.json'
if not os.path.exists(tok_path):
    tok = Tokenizer(models.WordPiece(unk_token='[UNK]'))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trn = trainers.WordPieceTrainer(vocab_size=VOCAB, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
    corpus = pd.concat([tr.query_clean, tr.drop_duplicates('product_id').product_title,
                        tr.drop_duplicates('product_id').product_bullet_point.str.slice(0, 300)]).tolist()
    tok.train_from_iterator(corpus, trn)
    tok.save(tok_path)
tok = Tokenizer.from_file(tok_path)
tok.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                   special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
tok.enable_truncation(MAXLEN, strategy='only_second'); tok.enable_padding(length=MAXLEN, pad_id=0)

def encode(d):
    enc = tok.encode_batch(list(zip(d.query_clean.tolist(), second(d))))
    ids = np.array([e.ids for e in enc], dtype=np.int32)
    seg = np.array([e.type_ids for e in enc], dtype=np.int8)
    return torch.from_numpy(ids).long(), torch.from_numpy(seg).long()

t0 = time.time()
Itr, Str = encode(tr); Iva, Sva = encode(va)
ytr, yva = torch.tensor(tr.y.values), va.y.values
print('encoded', Itr.shape, round(time.time() - t0), 's', flush=True)

class XEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN, D_MODEL); self.seg = nn.Embedding(2, D_MODEL)
        self.ln = nn.LayerNorm(D_MODEL); self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, 4 * D_MODEL, 0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(D_MODEL, 4))
    def forward(self, ids, seg):
        pos = torch.arange(ids.size(1)).unsqueeze(0)
        h = self.drop(self.ln(self.tok(ids) + self.pos(pos) + self.seg(seg)))
        h = self.enc(h, src_key_padding_mask=ids.eq(0))
        return self.head(h[:, 0])

model = XEnc()
print('params', sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(ytr) / BS); warm = int(0.05 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s / steps, 1)))))

@torch.no_grad()
def predict(I, S):
    model.eval(); out = []
    for i in range(0, len(I), 512):
        out.append(F.softmax(model(I[i:i + 512], S[i:i + 512]), -1))
    model.train(); return torch.cat(out).numpy()

log = []; step = 0; best = -1
for ep in range(EPOCHS):
    perm = torch.randperm(len(ytr)); t = time.time(); run = 0
    for bi, i in enumerate(range(0, len(perm), BS)):
        b = perm[i:i + BS]
        loss = F.cross_entropy(model(Itr[b], Str[b]), ytr[b])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; run += loss.item()
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{steps} loss {run / 200:.4f} {time.time() - t:.0f}s', flush=True); run = 0
        if step % 1000 == 0 or step == steps:
            p = predict(Iva, Sva); r = metrics(yva, p.argmax(1)); r.update(step=step, epoch=ep)
            log.append(r); print('VAL', step, r['micro_f1'], r['macro_f1'], r['per_class_f1'], flush=True)
            if r['micro_f1'] > best:
                best = r['micro_f1']; np.save(f'{R}/xenc_val_proba.npy', p); torch.save(model.state_dict(), f'{R}/xenc.pt')
            save('xenc', {'config': dict(MAXLEN=MAXLEN, EPOCHS=EPOCHS, BS=BS, LR=LR, D_MODEL=D_MODEL, LAYERS=LAYERS,
                                         HEADS=HEADS, VOCAB=VOCAB, params=sum(p.numel() for p in model.parameters())),
                          'log': log, 'best': max(log, key=lambda r: r['micro_f1'])})
