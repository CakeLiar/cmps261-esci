"""Write report/final_report.tex from the saved run outputs in results/ (every number in the report is read
from those files), then compile it with pdflatex if it is installed."""
import json, os, glob, subprocess, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, 'results'); OUT = os.path.join(ROOT, 'report'); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(f'{R}/{n}.json'))
has = lambda n: os.path.exists(f'{R}/{n}.json')
REPO = 'https://github.com/CakeLiar/cmps261-esci'
LAB = ['E', 'S', 'C', 'I']

ps, base, test, hyb, ea = J('prep_stats'), J('baseline'), J('test'), J('hybrid'), J('error_analysis')
lrc = J('lr_converged') if has('lr_converged') else None
tpl = J('test_predict_log') if has('test_predict_log') else {}
runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f'{R}/*.json')}
runs = {k: v for k, v in runs.items() if isinstance(v, dict) and 'config' in v and 'best' in v and 'model' in v['config']}
sig = J('significance') if has('significance') else None

def exact_from_confusion(o):
    """The JSON metrics are rounded to 4 decimals. Wherever a confusion matrix is stored, recompute micro-F1,
    macro-F1 and per-class F1 from it so the report rounds only once."""
    if isinstance(o, dict):
        cm = o.get('confusion')
        if isinstance(cm, list) and len(cm) == 4 and all(isinstance(r, list) and len(r) == 4 for r in cm):
            f = []
            for k in range(4):
                d = sum(cm[k]) + sum(cm[i][k] for i in range(4))
                f.append(2 * cm[k][k] / d if d else 0.0)
            o['micro_f1'] = sum(cm[k][k] for k in range(4)) / sum(map(sum, cm))
            o['macro_f1'] = sum(f) / 4
            o['per_class_f1'] = dict(zip(LAB, f))
        for v in o.values():
            exact_from_confusion(v)
    elif isinstance(o, list):
        for v in o:
            exact_from_confusion(v)
for _o in (base, test, hyb, lrc, runs):
    exact_from_confusion(_o)

f3 = lambda x: f'{x:.3f}'
sg = lambda x, d=3: f'${x:+.{d}f}$'
ci = lambda c, d=3: f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'.replace('-', '$-$')
pct = lambda x: f'{100 * x:.1f}\\%'
num = lambda x: f'{x:,}'.replace(',', '{,}')
PRETTY = {'minilm': 'MiniLM-L6', 'deberta_small': 'DeBERTa-v3-small', 'deberta_base': 'DeBERTa-v3-base',
          'lightgbm': 'LightGBM (lexical features)', 'logistic_regression': 'Logistic regression (TF-IDF)',
          'majority': 'Majority class (always E)'}
HNAME = {'ce_all_only': 'cross-encoder probabilities only', 'lex+ce_best': 'lexical + best cross-encoder',
         'lex+ce_all': 'lexical + all cross-encoders', 'lex+ce_all_balanced': 'lexical + all cross-encoders, balanced'}
def pretty(k):
    if k.startswith('hybrid_'):
        return f'Hybrid stacker ({HNAME.get(k[7:], k[7:])})'
    return PRETTY.get(k, k)

def last(tag): return runs[tag]['log'][-1]
def train_min(tag): return last(tag)['secs'] / 60
def row(name, r, bold=False):
    cells = [name, f3(r['micro_f1']), f3(r['macro_f1'])] + [f3(r['per_class_f1'][l]) for l in LAB]
    if bold:
        cells = [f'\\textbf{{{c}}}' for c in cells]
    return ' & '.join(cells) + r' \\'

# ---------- derived numbers ----------
ce_tags = [t for t in ('minilm', 'deberta_small', 'deberta_base') if t in runs]
best_ce = hyb['best_cross_encoder']
final = test['final_model']
lgb_v = base['best_lgb']
lr_v = lrc['best_lr'] if lrc else base['best_lr']
tm = test['models']
n_lgb_trees = test['val']['lightgbm_refit']['best_iter']
gain_final_vs_lgb = tm[final]['micro_f1'] - tm['lightgbm']['micro_f1']
hsel = hyb['selected']; hv = hyb['variants'][hsel]
ce_alone_best = hyb['cross_encoder_alone'][best_ce]

def seed_pair(a, b):
    if a in runs and b in runs:
        return last(a)['micro_f1'], last(b)['micro_f1'], last(a)['macro_f1'], last(b)['macro_f1']
    return None
seeds = {'MiniLM-L6': seed_pair('minilm', 'minilm_seed1')}
def at_step(tag, step):
    return next((r for r in runs[tag]['log'] if r['step'] == step), None) if tag in runs else None
ds1 = runs.get('deberta_small_seed1')
ds1_last = ds1['log'][-1]['step'] if ds1 else None
ds_total = runs['deberta_small']['log'][-1]['step']

# ---------- tables ----------
def t_data():
    rows = []
    for s in ('train', 'val', 'test'):
        d = ps[s]; rows.append(f"{ {'train': 'Train', 'val': 'Validation', 'test': 'Test'}[s]} & {num(d['queries'])} & {num(d['pairs'])} & "
                               + ' & '.join(pct(d['label_dist'][l]) for l in LAB) + r' \\')
    return '\n'.join(rows)

def t_val_main():
    rows = [row('Majority class (always E)', base['majority']),
            row('Logistic regression (TF-IDF + features)', lr_v),
            row('LightGBM (33 lexical features)', lgb_v)]
    for t in ce_tags:
        rows.append(row(PRETTY[t], runs[t]['best']))
    rows.append(row(f'Hybrid stacker, {HNAME[hsel]} (5-fold OOF)', hv, bold=final == f'hybrid_{hsel}'))
    return '\n'.join(rows)

def t_ablation():
    spec = [('minilm', 'Reference: title, brand+colour, bullets; length 128'),
            ('minilm_f_title', 'Title only'),
            ('minilm_f_title_bullets', 'Title + bullets'),
            ('minilm_f_all_desc', 'Reference fields + description'),
            ('minilm_len64', 'Max length 64'),
            ('minilm_len256', 'Max length 256'),
            ('minilm_balanced', 'Class-weighted loss (balanced)'),
            ('minilm_seed1', 'Seed 1 instead of 0')]
    rows = []
    for tag, name in spec:
        if tag in runs:
            r = last(tag)
            rows.append(f"{name} & {f3(r['micro_f1'])} & {f3(r['macro_f1'])} & {f3(r['per_class_f1']['C'])} & {f3(r['per_class_f1']['I'])} & {train_min(tag):.1f}" + r' \\')
    return '\n'.join(rows)

def t_hybrid():
    rows = []
    for t in hyb['cross_encoders']:
        rows.append(row(f'{PRETTY[t]} alone', hyb['cross_encoder_alone'][t]))
    for k, v in hyb['variants'].items():
        rows.append(row(f'Stacker: {HNAME[k]}', v, bold=k == hsel))
    return '\n'.join(rows)

def t_test():
    order = sorted(tm, key=lambda k: -tm[k]['micro_f1'])
    return '\n'.join(row(pretty(k) + (' $\\star$' if k == final else ''), tm[k], bold=k == final) for k in order)

def t_resources():
    rows = []
    for t in ce_tags + [x for x in ('minilm_len64', 'minilm_len256') if x in runs]:
        c, l = runs[t]['config'], last(t)
        name = PRETTY.get(t, t.replace('minilm_len', 'MiniLM-L6, length '))
        tp = tpl.get(t, {}).get('test_pairs_per_sec')
        test_thr = num(tp) if tp else '--'
        vsrc, mark = t, ''
        if not l.get('val_infer_secs') and t == 'minilm' and 'minilm_seed1' in runs:
            vsrc, mark = 'minilm_seed1', '$^\\dagger$'   # same model and length, timed in the second round
        vs = last(vsrc).get('val_infer_secs')
        val_thr = f"{num(round(ps['val']['pairs'] / vs))}{mark}" if vs else '--'
        mem = f"{l['peak_gpu_mem_gb']:.1f}" if l.get('peak_gpu_mem_gb') else '--'
        rows.append(f"{name} & {c['params'] / 1e6:.0f}M & {c['MAXLEN']} & {c['BS']} & {train_min(t):.0f} & {mem} & {val_thr} & {test_thr}" + r' \\')
    return '\n'.join(rows)

def t_curve():
    rows = []
    for frac, tag in ((0.1, 'minilm_frac10'), (0.25, 'minilm_frac25'), (0.5, 'minilm_frac50'), (1.0, 'minilm')):
        if tag in runs:
            l = last(tag); n = runs[tag]['config'].get('train_pairs') or ps['train']['pairs']
            rows.append(f"{int(frac * 100)}\\% & {num(n)} & {f3(l['micro_f1'])} & {f3(l['macro_f1'])} & {f3(l['per_class_f1']['C'])}" + r' \\')
    return '\n'.join(rows)

SL = {'all': 'All pairs', 'high_overlap_labelled_I_or_C': 'High title overlap, label I or C',
      'low_overlap_labelled_E': 'Low title overlap, label E', 'likely_misspelled_query': 'Likely misspelled query',
      'query_has_number': 'Query has a number', 'query_has_negation': 'Query has a negation'}
def t_slices():
    cols = ['lightgbm', best_ce, final] if final not in ('lightgbm', best_ce) else ['lightgbm', best_ce]
    head = ' & '.join(['Slice', 'Pairs'] + [pretty(c).split(' (')[0] for c in cols])
    rows = []
    for k, name in SL.items():
        s = test['slices'][k]
        rows.append(f"{name} & {num(s['n_pairs'])} & " + ' & '.join(f3(s[c]) for c in cols) + r' \\')
    return head, len(cols), '\n'.join(rows)

def complement_text():
    c = ea['complement']
    parts = []
    for m in ('best_baseline', 'minilm', 'deberta_small', 'deberta_base'):
        if m in c:
            nm = 'LightGBM' if m == 'best_baseline' else PRETTY[m]
            parts.append(f"{nm} {pct(c[m]['pred_counts']['C'] / c[m]['n'])}")
    return ', '.join(parts)

conf = tm[final]['confusion']
err_total = sum(conf[i][j] for i in range(4) for j in range(4) if i != j)
es_share = (conf[0][1] + conf[1][0]) / err_total
c_row = conf[2]; c_total = sum(c_row)
s_row = conf[1]; s_total = sum(s_row)
slh = ea['slices']['high_overlap_labelled_I_or_C']
sh, ncols, srows = t_slices()
db = 'deberta_base' in runs
lr_conv_note = (f"With the limit raised to {num(lrc['max_iter'])} iterations, "
                f"{sum(r['converged'] for r in lrc['lr_grid'])} of {len(lrc['lr_grid'])} settings converged; the best (C={lr_v['params']['C']}, "
                f"class weight {lr_v['params']['class_weight'] or 'none'}) reached validation micro-F1 {f3(lr_v['micro_f1'])}, against {f3(base['best_lr']['micro_f1'])} at 300 iterations.") if lrc else ''

seed_lines = []
for name, s in seeds.items():
    if s:
        seed_lines.append(f"{name}: micro-F1 {f3(s[0])} (seed 0) and {f3(s[1])} (seed 1), a difference of {abs(s[0] - s[1]):.3f}; macro-F1 {f3(s[2])} and {f3(s[3])}")
seed_text = '; '.join(seed_lines) + '.' if seed_lines else ''
if ds1:
    a, b = at_step('deberta_small', ds1_last), ds1['log'][-1]
    seed_text += (f" For DeBERTa-v3-small we only have part of the seed-1 run: the log stops at step 2{{,}}600 of {num(ds_total)} with no error, even though the status file says it exited normally."
                  f" From the step times it was sharing the GPU with the other jobs in that queue most of the time, and we deleted the VM before we noticed, so we don't know what happened to the full run."
                  f" At the last validation it finished (step {num(ds1_last)}), the two seeds give micro-F1 {f3(a['micro_f1'])} (seed 0) and {f3(b['micro_f1'])} (seed 1),"
                  f" macro-F1 {f3(a['macro_f1'])} and {f3(b['macro_f1'])}.")
seed_gap = abs(seeds['MiniLM-L6'][0] - seeds['MiniLM-L6'][1]) if seeds['MiniLM-L6'] else None

agree = ', '.join(f"{PRETTY[t]} {v['val_argmax_agreement_with_saved']:.4f}" for t, v in tpl.items() if t in PRETTY)

def cmpd(k, m='micro_f1'):
    return sig['comparisons'][k][m] if sig else None
def sig_sentence():
    if not sig:
        return ''
    t1, t2 = cmpd('test_deberta_base_vs_deberta_small'), cmpd('test_deberta_small_vs_minilm')
    t1m = cmpd('test_deberta_base_vs_deberta_small', 'macro_f1')
    hb, sd = cmpd('val_deberta_base_vs_hybrid_lex+ce_all'), cmpd('val_minilm_seed1_vs_minilm')
    tb = sig['single']['test_deberta_base']
    return (r"\textbf{Uncertainty.} To check which differences are bigger than just noise, we did a paired bootstrap over queries "
            f"({num(sig['resamples'])} resamples of whole queries, seed {sig['seed']}; \\texttt{{significance.py}}) and give the 95\\% interval of the difference. "
            f"On test, DeBERTa-v3-base is better than DeBERTa-v3-small by {sg(t1['diff'])} micro-F1 {ci(t1['ci95'])} but not clearly on macro-F1 ({sg(t1m['diff'])}, {ci(t1m['ci95'])}), "
            f"and DeBERTa-v3-small is better than MiniLM-L6 by {sg(t2['diff'])} {ci(t2['ci95'])}. "
            f"On validation, DeBERTa-v3-base minus the stacker is {sg(hb['diff'])} {ci(hb['ci95'])}, and MiniLM-L6 seed 1 minus seed 0 (best checkpoints) is {sg(sd['diff'])} {ci(sd['ci95'])}; both of these could be zero. "
            f"The final model's test micro-F1 interval is [{f3(tb['micro_f1']['ci95'][0])}, {f3(tb['micro_f1']['ci95'][1])}], so on test a difference of a few thousandths is about the smallest thing we can tell apart.")

capped = [g for g in base['lgb_grid'] if g['best_iter'] >= 2000]
lgb_cap_note = (f"{len(capped)} of the {len(base['lgb_grid'])} settings, all with balanced class weights, hit the limit of 2,000 trees before early stopping kicked in; they are all much worse than the unweighted ones anyway (best {f3(max(g['micro_f1'] for g in capped))})." if capped else '')
ckpt_note = ', '.join(f"{PRETTY[t]} at step {num(runs[t]['best']['step'])} of {num(last(t)['step'])}" for t in ce_tags) + '.'
_hv = sorted(hyb['variants'].items(), key=lambda kv: -kv[1]['micro_f1'])
hyb_tie = (f"{HNAME[_hv[1][0]]} ({f3(_hv[1][1]['micro_f1'])} against {f3(_hv[0][1]['micro_f1'])}"
           + (f"; bootstrap difference {ci(sig['comparisons']['val_hybrid_lex+ce_all_vs_ce_all_only']['micro_f1']['ci95'])})" if sig and hsel == 'lex+ce_all' and _hv[1][0] == 'ce_all_only' else ')'))
_cv = [(tag, last(tag)) for tag in ('minilm_frac10', 'minilm_frac25', 'minilm_frac50', 'minilm') if tag in runs]
curve_text = (f"Going from 10\\% to 100\\% of the training queries improves micro-F1 by {sg(_cv[-1][1]['micro_f1'] - _cv[0][1]['micro_f1'])} and Complement F1 from {f3(_cv[0][1]['per_class_f1']['C'])} to {f3(_cv[-1][1]['per_class_f1']['C'])}; "
              f"even the last step (50\\% to 100\\%) still adds {sg(_cv[-1][1]['micro_f1'] - _cv[-2][1]['micro_f1'])} micro-F1, so we think more data would still help, especially for Complement which is the rarest class."
              if len(_cv) == 4 else '')

def r_(tag, key='micro_f1'): return f3(runs[tag]['best'][key]) if tag in runs else 'n/a'

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{mathptmx}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small}
\setlength{\parskip}{3pt}
\title{Classifying Query--Product Relevance on Amazon ESCI with Lexical Baselines, Cross-Encoders and a Stacked Hybrid\\[4pt]\large CMPS261 Final Project Report (Option 4.2)}
\author{Allie Saoud \and Zulfiquar Fneiche}
\date{}
\begin{document}
\maketitle

\begin{abstract}
In this project we take a shopping query and a product from the Amazon Shopping Queries dataset (ESCI) and try to predict if the product is Exact, Substitute, Complement or Irrelevant for that query.
We worked on the US reduced set (""" + num(ps['train']['pairs'] + ps['val']['pairs'] + ps['test']['pairs']) + r""" pairs) and tried a majority-class baseline, logistic regression on TF-IDF, LightGBM on 33 lexical features we made ourselves, three pretrained cross-encoders that we fine-tuned (MiniLM-L6, DeBERTa-v3-small and DeBERTa-v3-base), and a LightGBM stacker that mixes the cross-encoder probabilities with the lexical features.
We picked every model and setting on a validation split grouped by query, and we only ran on the test split one time at the end.
Our final model, """ + pretty(final) + r""", reaches test micro-F1 """ + f3(tm[final]['micro_f1']) + r""" and macro-F1 """ + f3(tm[final]['macro_f1']) + r""", against """ + f3(tm['lightgbm']['micro_f1']) + ' and ' + f3(tm['lightgbm']['macro_f1']) + r""" for LightGBM.
Complement is still the hardest class (test F1 """ + f3(tm[final]['per_class_f1']['C']) + r"""), and """ + pct(es_share) + r""" of the errors that are left are Exact and Substitute getting mixed up.
We also did ablations on the product fields, sequence length, class weights, training set size and random seed, and we measured how much compute each model needs.
\end{abstract}

\section{Introduction}
When someone searches on an online store, the store needs to know how well each product matches what they typed.
The ESCI dataset \cite{reddy2022} gives four labels for this: \emph{Exact} (the product is what the person asked for), \emph{Substitute} (it is close and could be used instead), \emph{Complement} (it is used together with what they asked for) and \emph{Irrelevant}.
What makes it hard is that the label often depends on a small detail. For example, for the query ``phone case for iPhone 12'', an iPhone 12 itself is a Complement, while a case for an iPhone 11 is a Substitute or Irrelevant.
Just counting shared words cannot tell these apart.

We wanted to answer three questions: (1) how far can a well tuned lexical model go, (2) how much do pretrained cross-encoders add, and which choices matter for them, and (3) does combining the two help.
In the progress report we had the baselines and two cross-encoders. For this final report we added a bigger cross-encoder, ablations where we change one thing at a time, the hybrid stacker, a look at compute cost, and the test evaluation.
Everything (code, notebook, logs) is on our GitHub, \url{""" + REPO + r"""}, and the trained models can be downloaded from release v1.0, \url{""" + REPO + r"""/releases/tag/v1.0}.

\section{Dataset}
We use the US part of the reduced version (\texttt{small\_version == 1}) of ESCI \cite{reddy2022} and join it with the product catalogue (""" + num(ps['n_products']) + r""" products).
Every pair had its product in the catalogue and we found no duplicate pairs.
All products have a title, but the description is missing for """ + pct(ps['missing_rate_products']['product_description']) + r""" of products, bullet points for """ + pct(ps['missing_rate_products']['product_bullet_point']) + r""", brand for """ + pct(ps['missing_rate_products']['product_brand']) + r""" and colour for """ + pct(ps['missing_rate_products']['product_color']) + r""".
Queries are short (on average """ + f"{ps['query_len_words_mean']:.1f}" + r""" words) while titles are much longer (on average """ + f"{ps['title_len_words_mean']:.1f}" + r""" words).

\textbf{Cleaning.} We unescape HTML, remove tags, fix extra whitespace and lowercase everything. Missing fields are replaced with empty strings.

\textbf{Splits.} We did not touch the official test split until the end. From the official train split we took 10\% of the \emph{queries} (not the pairs) as validation, with seed 42, so the same query never shows up in both train and validation.
Table~\ref{tab:data} has the sizes and the label shares. The data is imbalanced, Complement is less than 5\% of the pairs.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Split & Queries & Pairs & E & S & C & I \\\midrule
""" + t_data() + r"""
\bottomrule\end{tabular}
\caption{Split sizes and label distribution.}\label{tab:data}
\end{table}

\textbf{Metric.} Like KDD Cup 2022 task 2 \cite{reddy2022}, our main metric is micro-F1 (which is the same as accuracy when each pair has one label). We also report macro-F1 and F1 per class, since micro-F1 alone hides how badly a model does on Complement.
The reduced set has the harder queries of the full dataset \cite{reddy2022} and we only train on it, so our numbers can not be compared directly with the KDD Cup leaderboard, where teams trained on more data and were scored on a hidden test set.

\section{Models}
\textbf{Majority class.} Always predicts Exact.

\textbf{Lexical features + LightGBM.} For each pair we compute 33 features (\texttt{feats.py}): token overlap between the query and the title, bullets, description, brand and colour (plain and IDF-weighted, fitted on training titles); Jaccard and character 3-gram overlap with the title; BM25-style scores \cite{robertson2009} against title, bullets and description; query, title, bullet and description lengths and whether each field is present; the count of numbers in the query and the share of them found in the title or bullets; whether the brand appears in the query; the number and highest IDF of query words found in no product field; whether the query has a negation; counts of complement cue words (such as ``case'', ``charger'', ``for'') in query and title; whether a query word appears after ``for'' in the title; whether the first and last query words are in the title; and the mean IDF of the query.
We train LightGBM \cite{ke2017} with learning rate 0.05 and early stopping on validation (50 rounds), and we tune \texttt{num\_leaves} $\in\{31,127\}$, \texttt{min\_child\_samples} $\in\{50,200\}$ and class weights (none or balanced), so 8 settings in total.
The best one used """ + str(lgb_v['params']['num_leaves']) + r""" leaves, minimum """ + str(lgb_v['params']['min_child_samples']) + r""" samples per leaf, no class weights, and stopped at """ + str(lgb_v['best_iter']) + r""" trees. """ + lgb_cap_note + r"""

\textbf{Logistic regression.} We use multinomial logistic regression \cite{pedregosa2011} on word and character TF-IDF of the query and the title (""" + num(1045120) + r""" sparse columns, fit on training text only) plus the standardised lexical features, with $C\in\{0.1,0.3,1,3\}$ and optional balanced class weights.
In the progress report these fits stopped at 300 lbfgs iterations before converging, so we ran them again. """ + lr_conv_note + r"""

\textbf{Cross-encoders.} A cross-encoder takes the query and the product text together as one input, so the model can look at both at the same time before deciding \cite{nogueira2019}.
We fine-tune three pretrained models with a new 4-class head and cross-entropy loss: MiniLM-L6 \cite{wang2020}, which was already trained as a re-ranker on MS MARCO \cite{bajaj2016}, and DeBERTa-v3-small and DeBERTa-v3-base \cite{he2023}.
The first input is the cleaned query. The second is the title, brand and colour, and the first 300 characters of the bullet points, joined with the separator token and cut at 128 tokens.
We train for 2 epochs with AdamW \cite{loshchilov2019} (weight decay 0.01, 5\% linear warm-up, then linear decay), learning rate $3\times10^{-5}$ and batch 64 for MiniLM and $2\times10^{-5}$ and batch 48 for DeBERTa, with mixed precision \cite{wolf2020}.
Gradients are clipped at norm 1.0. Every 2,000 steps we evaluate on validation and keep the checkpoint with the best micro-F1. The kept checkpoints are: """ + ckpt_note + r"""

\textbf{Hybrid stacker.} The cross-encoders read the text but do not directly see our hand made signals (for example if a number in the query also appears in the title), and LightGBM can not read the text at all. So we tried stacking them \cite{wolpert1992}: a second LightGBM gets the class probabilities of the cross-encoders plus the 33 lexical features.
The cross-encoders were trained on the train split, so their probabilities on train pairs are too confident to train on. Instead we train the stacker on the \emph{validation} split and score it with 5-fold cross-validation grouped by query, which means each validation prediction comes from a model that never saw that query.
We tried four different inputs (Table~\ref{tab:hybrid}) with the same fixed settings (300 trees, learning rate 0.05, 31 leaves, at least 100 samples per leaf) and picked by out-of-fold micro-F1.

\textbf{Model selection.} Our final model is simply the one with the highest validation micro-F1 (out-of-fold for the stacker). For the test run the stacker is refit on all the validation pairs.

\section{Results}
We trained everything on Google Cloud VMs with one NVIDIA L4 GPU (24\,GB) each; for the second round we used \texttt{g2-standard-32} (32 vCPUs, \texttt{infra/config.sh}).
All times below are wall-clock, so they also include CPU work like tokenization.

\subsection{Validation results and model selection}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_val_main() + r"""
\bottomrule\end{tabular}
\caption{Validation results (""" + num(ps['val']['pairs']) + r""" pairs). Cross-encoders: best checkpoint. Stacker: 5-fold out-of-fold predictions.}\label{tab:val}
\end{table}

Table~\ref{tab:val} is the main comparison. LightGBM is better than the majority baseline mostly on Substitute and Irrelevant, but it almost never predicts Complement (F1 """ + f3(lgb_v['per_class_f1']['C']) + r""").
All the cross-encoders beat it, and the biggest gains are on Complement and Irrelevant. The bigger the encoder the better it does: """ + ', '.join(f"{PRETTY[t]} {r_(t)}" for t in ce_tags) + r""" micro-F1.

\subsection{Hybrid stacker}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Input & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_hybrid() + r"""
\bottomrule\end{tabular}
\caption{Hybrid stacker variants, validation, 5-fold out-of-fold grouped by query. ``Alone'' rows are the cross-encoders' own argmax on the same pairs.}\label{tab:hybrid}
\end{table}

The best input by out-of-fold micro-F1 is """ + HNAME[hsel] + r""" (Table~\ref{tab:hybrid}), but it is basically a tie with """ + hyb_tie + r""". Compared to the best single cross-encoder, """ + PRETTY[best_ce] + r""", on the same pairs, it changes micro-F1 by """ + sg(hv['micro_f1'] - ce_alone_best['micro_f1']) + r""" and macro-F1 by """ + sg(hv['macro_f1'] - ce_alone_best['macro_f1']) + r""".
""" + (r"""So stacking did not help us. One reason is that the stacker only trains on validation pairs (around four fifths of """ + num(ps['val']['pairs']) + r""" per fold), and the lexical features don't add much that the cross-encoders did not already learn. Adding the lexical features to only the best cross-encoder (""" + f3(hyb['variants']['lex+ce_best']['micro_f1']) + r""") is also worse than that cross-encoder by itself.""" if hv['micro_f1'] <= ce_alone_best['micro_f1'] else '') + r"""
Using balanced class weights in the stacker drops micro-F1 to """ + f3(hyb['variants']['lex+ce_all_balanced']['micro_f1']) + r""" and leave Complement F1 unchanged (""" + f3(hyb['variants']['lex+ce_all_balanced']['per_class_f1']['C']) + r""" against """ + f3(hyb['variants']['lex+ce_all']['per_class_f1']['C']) + r""").

\subsection{Ablations (MiniLM-L6, validation)}
To see which choices actually matter, we changed one setting at a time on MiniLM-L6 since it is the cheapest cross-encoder (Table~\ref{tab:abl}).

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}\toprule
Setting & micro-F1 & macro-F1 & F1 C & F1 I & Train min \\\midrule
""" + t_ablation() + r"""
\bottomrule\end{tabular}
\caption{Ablations on MiniLM-L6 (one change at a time, validation). All rows use the last checkpoint (end of epoch 2) so that we compare the settings and not which checkpoint happened to be the best. The reference run is from the first round on a different L4 VM that was also running the LightGBM grid, so its training time can't be compared directly.}\label{tab:abl}
\end{table}

Class weighting makes micro-F1 a lot worse. From the field and length changes, only using the title alone (no bullets, brand or colour) clearly hurts (""" + sg(-cmpd('val_minilm_vs_minilm_f_title')['diff']) + r""" micro-F1 at the best checkpoint, bootstrap interval """ + ci([-x for x in cmpd('val_minilm_vs_minilm_f_title')['ci95']][::-1]) + r"""); the other changes are only a few thousandths away from the reference. The description row is not a very fair test, because we add the description at the end and at 128 tokens it usually gets cut off.

\textbf{Seed variance.} """ + seed_text + r""" So if two settings differ by less than the MiniLM-L6 seed gap in micro-F1, we don't count it as a real difference.

""" + sig_sentence() + r"""

\subsection{Complexity and resource analysis}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrrr}\toprule
Model & Params & Max len & Batch & Train min & Peak GPU GB & Val pairs/s & Test pairs/s \\\midrule
""" + t_resources() + r"""
\bottomrule\end{tabular}
\caption{Cost of the cross-encoders on one L4 GPU. Train minutes are wall-clock for 2 epochs, including the validation passes; the first-round runs (MiniLM-L6, DeBERTa-v3-small) were sharing the VM's CPUs with other jobs. \emph{Val pairs/s}: forward passes on the GPU over the already tokenized validation split, batch 256, measured the same way for every row ($^\dagger$ timed on the same model's seed-1 run; we did not log it for the first-round DeBERTa-v3-small). \emph{Test pairs/s}: the test prediction, batch 512, which also includes tokenizing on the CPU, so it is lower and should not be compared with the validation column. We did not log peak memory in the first round (--).}\label{tab:res}
\end{table}

Table~\ref{tab:res} shows the cost. A cross-encoder gets more expensive with both model size and sequence length, while the classical models are way cheaper: refitting LightGBM took """ + str(test['val']['lightgbm_refit']['fit_secs']) + r""" seconds on CPU (""" + str(n_lgb_trees) + r""" trees) and predicted the """ + num(test['test_pairs']) + r""" test pairs in """ + str(test['val']['lightgbm_refit']['test_infer_secs']) + r""" seconds, and the stacker is only one small LightGBM on top of the cross-encoder outputs.
On the other hand, each logistic regression fit on """ + num(1045120) + r""" sparse columns took """ + (f"{min(r['secs'] for r in lrc['lr_grid']) / 60:.0f}--{max(r['secs'] for r in lrc['lr_grid']) / 60:.0f}" if lrc else '') + r""" minutes per setting (all 8 ran in parallel on one 32-vCPU VM while DeBERTa-v3-base was training on its GPU), which is about as long as fine-tuning MiniLM-L6, and the scores are much lower.

\begin{figure}[H]\centering
\includegraphics[width=0.62\linewidth]{../figures/final_learning_curve.png}
\caption{Learning curve for MiniLM-L6: validation F1 at the last checkpoint against the number of training pairs (training queries subsampled, seed 0).}\label{fig:curve}
\end{figure}

\begin{table}[H]\centering\small
\begin{tabular}{rrrrr}\toprule
Share of train queries & Pairs & micro-F1 & macro-F1 & F1 C \\\midrule
""" + t_curve() + r"""
\bottomrule\end{tabular}
\caption{Learning curve data (MiniLM-L6, last checkpoint, validation).}\label{tab:curve}
\end{table}

Figure~\ref{fig:curve} and Table~\ref{tab:curve} show what happens when we use less training data. """ + curve_text + r"""

\subsection{Test results}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_test() + r"""
\bottomrule\end{tabular}
\caption{Held-out test split (""" + num(test['test_pairs']) + r""" pairs, """ + num(test['test_queries']) + r""" queries), evaluated once. $\star$ marks the model chosen on validation.}\label{tab:test}
\end{table}

Table~\ref{tab:test} has the test results. The model we picked on validation, """ + pretty(final) + r""", scores """ + f3(tm[final]['micro_f1']) + r""" micro-F1, """ + sg(gain_final_vs_lgb) + r""" over LightGBM.
Its validation micro-F1 was """ + f3(runs[final]['best']['micro_f1'] if final in runs else test['val_micro_used_for_selection'][final]) + r""", so test and validation are very close and the validation split was a good guide.
The hybrid stacker, which we did not pick, gets """ + f3(tm[f'hybrid_{hsel}']['micro_f1']) + r""" on test; the gap to our model is smaller than the seed variation we measured, so we don't see it as a real difference.
Figure~\ref{fig:pc} compares F1 per class on test: the cross-encoders improve the most on Complement and Irrelevant, which are the classes where word overlap fools LightGBM.
""" + (f"To make sure the saved checkpoints are the ones we evaluated, we reloaded each model and predicted validation again; the share of pairs matching the saved predictions was: {agree}." if agree else '') + r"""

\begin{figure}[H]\centering
\includegraphics[width=0.85\linewidth]{../figures/final_test_per_class_f1.png}
\caption{Per-class test F1 of LightGBM, the best single cross-encoder and the hybrid stacker.}\label{fig:pc}
\end{figure}

\subsection{Failure analysis}
\begin{figure}[H]\centering
\includegraphics[width=0.48\linewidth]{../figures/final_test_confusion.png}
\caption{Test confusion matrix of the final model (row-normalised shares and counts).}\label{fig:conf}
\end{figure}

\textbf{Confusions.} In the final model's test confusion matrix (Figure~\ref{fig:conf}), """ + pct(s_row[0] / s_total) + r""" of Substitute pairs are predicted Exact, and only """ + pct(c_row[2] / c_total) + r""" of Complement pairs are predicted Complement (""" + pct(c_row[0] / c_total) + r""" are predicted Exact).
Exact vs Substitute usually comes down to one detail in the query, like a size, a model number or a brand.

\textbf{Complement recall.} On validation the Complement recall is: """ + complement_text() + r""".
Complement products can share a lot of words with the query (like a charger whose title has the name of the phone in the query), and this pushes models that rely on overlap towards Exact. The high-overlap slice below shows this.

\textbf{Slices.} We chose five slices before looking at any results (Table~\ref{tab:slices}, which also has all pairs for reference): pairs with high title overlap (at least 75\% of query tokens in the title) labelled Irrelevant or Complement; pairs with low overlap (at most 25\%) labelled Exact; queries with a likely misspelling (a 4+ letter word never seen in training titles or bullets); and queries with a number or a negation.
On validation (the table is test), LightGBM only got """ + pct(slh['best_baseline']['accuracy']) + r""" of the high-overlap I/C pairs right since it depends on overlap, compared to """ + ', '.join(f"{pct(slh[m]['accuracy'])} for {PRETTY[m]}" for m in ('minilm', 'deberta_small', 'deberta_base') if m in slh) + r""".

\begin{table}[H]\centering\small
\begin{tabular}{lr""" + 'r' * ncols + r"""}\toprule
""" + sh + r""" \\\midrule
""" + srows + r"""
\bottomrule\end{tabular}
\caption{Test accuracy on the failure slices.}\label{tab:slices}
\end{table}

\section{Conclusion}
Fine-tuned cross-encoders are clearly better than lexical models on ESCI, mostly because they get Complement and Irrelevant pairs right where word overlap gets them wrong, and bigger encoders do even better but cost more.
""" + (r"""Stacking the cross-encoder outputs with the lexical features did not beat the best single cross-encoder on validation, so our final model is """ + pretty(final) + r""" (test micro-F1 """ + f3(tm[final]['micro_f1']) + r""").""" if not final.startswith('hybrid') else r"""Stacking the cross-encoder outputs with lexical features gave a further gain at almost no extra cost, and this stacker was our final model.""") + r"""
From the ablations, class weights in the loss hurt, using more fields than just the title helps a little, and a lot of differences of a few thousandths are just seed noise.
The main problem that is still open is Complement and the line between Exact and Substitute. If we had more time we would train on the full dataset instead of the reduced one, try bigger encoders or longer training, give the description more room with a longer input, and add query-level features like the other products that came up for the same query.

\begin{thebibliography}{99}\small
\bibitem{reddy2022} C.~K.~Reddy et~al. Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search. arXiv:2206.06588, 2022.
\bibitem{ke2017} G.~Ke, Q.~Meng, T.~Finley, T.~Wang, W.~Chen, W.~Ma, Q.~Ye, T.-Y.~Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.
\bibitem{he2023} P.~He, J.~Gao, W.~Chen. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR, 2023.
\bibitem{wang2020} W.~Wang, F.~Wei, L.~Dong, H.~Bao, N.~Yang, M.~Zhou. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS, 2020.
\bibitem{nogueira2019} R.~Nogueira, K.~Cho. Passage Re-ranking with BERT. arXiv:1901.04085, 2019.
\bibitem{bajaj2016} P.~Bajaj et~al. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268, 2016.
\bibitem{robertson2009} S.~Robertson, H.~Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 2009.
\bibitem{loshchilov2019} I.~Loshchilov, F.~Hutter. Decoupled Weight Decay Regularization. ICLR, 2019.
\bibitem{pedregosa2011} F.~Pedregosa et~al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.
\bibitem{wolf2020} T.~Wolf et~al. Transformers: State-of-the-Art Natural Language Processing. EMNLP (System Demonstrations), 2020.
\bibitem{wolpert1992} D.~H.~Wolpert. Stacked Generalization. Neural Networks 5(2), 1992.
\end{thebibliography}
\end{document}
"""
open(f'{OUT}/final_report.tex', 'w').write(tex)
print('wrote report/final_report.tex')
if shutil.which('pdflatex'):
    for _ in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'final_report.tex'], cwd=OUT, capture_output=True, text=True)
    print('pdflatex exit', r.returncode)
    if r.returncode:
        print(r.stdout[-3000:])
    for ext in ('aux', 'log', 'out'):
        p = f'{OUT}/final_report.{ext}'
        if os.path.exists(p) and r.returncode == 0:
            os.remove(p)
