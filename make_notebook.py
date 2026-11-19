"""Build CMPS261_ESCI.ipynb: every script of the project in order, with explanations, plus analysis cells
that load the saved results. Heavy steps (preprocessing, features, training) run only when the matching
flag is set to True; with the defaults the notebook loads the saved outputs of those runs instead."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

def script(name, text, flag, cmd=None):
    md(text)
    src = open(name).read()
    code(f'%%writefile {name}\n{src}')
    code(f'if {flag}:\n    !{cmd or "python " + name}')

md("""# CMPS261 Project: Amazon ESCI query–product relevance
**Authors:** Allie Saoud, Zulfiquar Fneiche · Project option 4.2

Task: for a shopping query and a product, predict Exact (E), Substitute (S), Complement (C) or Irrelevant (I).
This notebook holds all the code: data cleaning and processing, features, baselines, cross-encoder training,
hyper-parameter tuning, model selection, diagnosis and the final test evaluation.

Heavy steps were run on GCP VMs with one NVIDIA L4 GPU each (see `infra/`). Set the flags below to `True` to rerun them;
with the default `False` the notebook loads the saved outputs in `results/`.""")
code("""RUN_PREP = False      # download-free: needs raw/examples.parquet and raw/products.parquet
RUN_FEATS = False
RUN_BASELINES = False # LightGBM and logistic regression grids (CPU, about 1.5 h)
RUN_TRAIN = False     # cross-encoder fine-tuning (GPU)
RUN_TEST = False      # final one-time test evaluation""")

script('common.py', '## 0. Shared helpers\nPaths and the metric function used everywhere (micro-F1, macro-F1, per-class F1, confusion matrix).', 'False')
script('prep.py', """## 1. Data cleaning and preprocessing
US locale, reduced set (`small_version == 1`). HTML unescape, tag removal, whitespace normalisation, lowercasing.
Validation = 10% of training queries (seed 42), grouped by query. The official test split is held out.""", 'RUN_PREP')
script('feats.py', """## 2. Feature engineering
33 lexical query–product features and sparse TF-IDF matrices (fit on training data only).""", 'RUN_FEATS')
script('baseline.py', """## 3. Classical baselines and hyper-parameter tuning
LightGBM (8 settings) and logistic regression on TF-IDF (8 settings); selection by validation micro-F1.""", 'RUN_BASELINES')
script('lr_converge.py', """### 3b. Logistic regression until convergence
The first run stopped at 300 iterations. This reruns the same grid with a higher limit.""", 'RUN_BASELINES')
script('xenc_pretrained.py', """## 4. Cross-encoder fine-tuning
Pretrained cross-encoders with a new 4-class head. The env vars select the model, fields, max length, seed,
training fraction and class weights. `run_experiments.sh` lists every configuration we ran.""", 'RUN_TRAIN',
       'bash run_experiments.sh')
script('error_analysis.py', '## 5. Failure analysis on validation\nAccuracy on slices fixed before looking at results.', 'RUN_TRAIN')
script('hybrid.py', """## 6. Hybrid model
LightGBM stacked on cross-encoder probabilities and lexical features, trained on validation and scored with
5-fold cross-validation grouped by query.""", 'RUN_TRAIN')
script('predict_test.py', '## 7. Test predictions for the final models (run once)', 'RUN_TEST',
       'TAGS=minilm,deberta_small,deberta_base python predict_test.py')
script('final_test.py', '## 8. Final test evaluation (run once)', 'RUN_TEST')

md('## 9. Results\nEverything below is loaded from the saved run outputs in `results/`.')
code("""import json, glob, os
import pandas as pd
from IPython.display import display, Image
J = lambda n: json.load(open(f'results/{n}.json'))
def row(name, r):
    return dict(model=name, micro=r['micro_f1'], macro=r['macro_f1'], **{k: v for k, v in r['per_class_f1'].items()})""")
md('### 9.1 Baseline tuning grids (validation)')
code("""b = J('baseline'); lc = J('lr_converged')
display(pd.DataFrame([dict(**r['params'], trees=r['best_iter'], micro=r['micro_f1'], macro=r['macro_f1']) for r in b['lgb_grid']]))
display(pd.DataFrame([dict(**r['params'], iters=r['n_iter'], converged=r['converged'], micro=r['micro_f1'], macro=r['macro_f1']) for r in lc['lr_grid']]))""")
md('### 9.2 Every cross-encoder run (validation, best checkpoint)')
code("""runs = []
for p in sorted(glob.glob('results/*.json')):
    j = json.load(open(p))
    if isinstance(j, dict) and 'config' in j and 'best' in j and 'model' in j['config']:
        c, r = j['config'], j['best']
        runs.append(dict(tag=os.path.basename(p)[:-5], model=c['model'].split('/')[-1], fields=','.join(c.get('FIELDS', ['title','brandcolor','bullets'])),
                         maxlen=c['MAXLEN'], frac=c.get('FRAC', 1.0), cw=c.get('CW'), seed=c['SEED'], step=r['step'],
                         micro=r['micro_f1'], macro=r['macro_f1'], C=r['per_class_f1']['C'], train_min=round(j['log'][-1]['secs'] / 60, 1),
                         peak_gpu_gb=j['log'][-1].get('peak_gpu_mem_gb')))
pd.DataFrame(runs).sort_values('micro', ascending=False)""")
md('### 9.3 Hybrid (5-fold out-of-fold on validation)')
code("""h = J('hybrid')
display(pd.DataFrame([row(k, v) for k, v in h['variants'].items()] + [row(k + ' alone', v) for k, v in h['cross_encoder_alone'].items()]))
print('selected:', h['selected'])""")
md('### 9.4 Failure slices (validation)')
code("""ea = J('error_analysis')
pd.DataFrame({k: {m: v[m]['accuracy'] for m in v if isinstance(v.get(m), dict) and 'accuracy' in v[m]} | {'pairs': v['n_pairs']} for k, v in ea['slices'].items()}).T""")
md('### 9.5 Test results (held-out split, evaluated once)')
code("""t = J('test')
display(pd.DataFrame([row(k, v) for k, v in t['models'].items()]).sort_values('micro', ascending=False))
display(pd.DataFrame(t['slices']).T)""")
md('### 9.6 Figures')
code("""for f in sorted(glob.glob('figures/*.png')):
    print(f); display(Image(f, width=650))""")
nb['cells'] = C
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'CMPS261_ESCI.ipynb')
print('wrote CMPS261_ESCI.ipynb with', len(C), 'cells')
