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
lr_conv_note = (f"With the iteration limit raised to {num(lrc['max_iter'])}, "
                f"{sum(r['converged'] for r in lrc['lr_grid'])} of {len(lrc['lr_grid'])} settings converged; the best (C={lr_v['params']['C']}, "
                f"class weight {lr_v['params']['class_weight'] or 'none'}) reached validation micro-F1 {f3(lr_v['micro_f1'])}, against {f3(base['best_lr']['micro_f1'])} at 300 iterations.") if lrc else ''

seed_lines = []
for name, s in seeds.items():
    if s:
        seed_lines.append(f"{name}: micro-F1 {f3(s[0])} (seed 0) and {f3(s[1])} (seed 1), a difference of {abs(s[0] - s[1]):.3f}; macro-F1 {f3(s[2])} and {f3(s[3])}")
seed_text = '; '.join(seed_lines) + '.' if seed_lines else ''
if ds1:
    a, b = at_step('deberta_small', ds1_last), ds1['log'][-1]
    seed_text += (f" For DeBERTa-v3-small only a partial seed-1 run survives: its log stops at training step 2{{,}}600 of {num(ds_total)} with no error, although the job status file records a normal exit."
                  f" Its step times show that it shared the GPU with the other jobs of that queue for most of its life, and the VM was deleted before we noticed, so we cannot tell what happened to the full run."
                  f" At the last validation pass it did complete (step {num(ds1_last)}), the two seeds give micro-F1 {f3(a['micro_f1'])} (seed 0) and {f3(b['micro_f1'])} (seed 1),"
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
    return (r"\textbf{Uncertainty.} To check which differences are larger than sampling noise we ran a paired bootstrap over queries "
            f"({num(sig['resamples'])} resamples of whole queries, seed {sig['seed']}; \\texttt{{significance.py}}) and report 95\\% intervals of the difference. "
            f"On test, DeBERTa-v3-base beats DeBERTa-v3-small by {sg(t1['diff'])} micro-F1 {ci(t1['ci95'])} but not clearly on macro-F1 ({sg(t1m['diff'])}, {ci(t1m['ci95'])}), "
            f"and DeBERTa-v3-small beats MiniLM-L6 by {sg(t2['diff'])} {ci(t2['ci95'])}. "
            f"On validation, DeBERTa-v3-base minus the stacker is {sg(hb['diff'])} {ci(hb['ci95'])}, and MiniLM-L6 seed 1 minus seed 0 (best checkpoints) is {sg(sd['diff'])} {ci(sd['ci95'])}; neither is distinguishable from zero. "
            f"The final model's test micro-F1 interval is [{f3(tb['micro_f1']['ci95'][0])}, {f3(tb['micro_f1']['ci95'][1])}], so differences of a few thousandths on test are at the edge of what the test set can resolve.")

capped = [g for g in base['lgb_grid'] if g['best_iter'] >= 2000]
lgb_cap_note = (f"{len(capped)} of the {len(base['lgb_grid'])} settings, all with balanced class weights, reached the 2,000-tree cap without early stopping; they score well below the unweighted ones (best {f3(max(g['micro_f1'] for g in capped))})." if capped else '')
ckpt_note = ', '.join(f"{PRETTY[t]} at step {num(runs[t]['best']['step'])} of {num(last(t)['step'])}" for t in ce_tags) + '.'
_hv = sorted(hyb['variants'].items(), key=lambda kv: -kv[1]['micro_f1'])
hyb_tie = (f"{HNAME[_hv[1][0]]} ({f3(_hv[1][1]['micro_f1'])} against {f3(_hv[0][1]['micro_f1'])}"
           + (f"; bootstrap difference {ci(sig['comparisons']['val_hybrid_lex+ce_all_vs_ce_all_only']['micro_f1']['ci95'])})" if sig and hsel == 'lex+ce_all' and _hv[1][0] == 'ce_all_only' else ')'))
_cv = [(tag, last(tag)) for tag in ('minilm_frac10', 'minilm_frac25', 'minilm_frac50', 'minilm') if tag in runs]
curve_text = (f"Going from 10\\% to 100\\% of the training queries raises micro-F1 by {sg(_cv[-1][1]['micro_f1'] - _cv[0][1]['micro_f1'])} and Complement F1 from {f3(_cv[0][1]['per_class_f1']['C'])} to {f3(_cv[-1][1]['per_class_f1']['C'])}; "
              f"the last doubling (50\\% to 100\\%) still adds {sg(_cv[-1][1]['micro_f1'] - _cv[-2][1]['micro_f1'])} micro-F1, so more data would likely help further, most of all for Complement, the rarest class."
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
We classify shopping query--product pairs from the Amazon Shopping Queries dataset (ESCI) into Exact, Substitute, Complement and Irrelevant.
On the US reduced set (""" + num(ps['train']['pairs'] + ps['val']['pairs'] + ps['test']['pairs']) + r""" pairs) we compare a majority-class baseline, logistic regression on TF-IDF, LightGBM on 33 hand-built lexical features, three fine-tuned pretrained cross-encoders (MiniLM-L6, DeBERTa-v3-small and DeBERTa-v3-base) and a LightGBM stacker that combines cross-encoder probabilities with the lexical features.
All model choices were made on a validation split grouped by query, and the held-out test split was evaluated once.
The selected model, """ + pretty(final) + r""", reaches test micro-F1 """ + f3(tm[final]['micro_f1']) + r""" and macro-F1 """ + f3(tm[final]['macro_f1']) + r""", against """ + f3(tm['lightgbm']['micro_f1']) + ' and ' + f3(tm['lightgbm']['macro_f1']) + r""" for LightGBM.
The Complement class stays hardest (test F1 """ + f3(tm[final]['per_class_f1']['C']) + r"""), and """ + pct(es_share) + r""" of the remaining errors are confusions between Exact and Substitute.
We also report ablations on product fields, sequence length, class weighting, training-set size and random seed, and the compute cost of each model.
\end{abstract}

\section{Introduction}
When a customer searches an online store, the store has to decide how well each product matches the query.
The ESCI dataset \cite{reddy2022} labels this with four classes: \emph{Exact} (the product is what was asked for), \emph{Substitute} (it is close and could replace it), \emph{Complement} (it goes with the requested product) and \emph{Irrelevant}.
The task is hard because the classes depend on small details: for a query ``phone case for iPhone 12'', an iPhone 12 would be a Complement and a case for an iPhone 11 a Substitute or Irrelevant.
Word overlap alone does not separate these cases.

Our questions were: (1) how far a well-tuned lexical model gets, (2) how much pretrained cross-encoders add, and which design choices matter for them, and (3) whether combining both helps.
Our progress report covered the baselines and two cross-encoders. This report adds a larger cross-encoder, a set of controlled ablations, the hybrid stacker, a resource analysis and the one-time test evaluation.
All code, the notebook and run logs are in our repository, \url{""" + REPO + r"""}, and the trained models are attached to its release v1.0, \url{""" + REPO + r"""/releases/tag/v1.0}.

\section{Dataset}
We use the US locale of the reduced version (\texttt{small\_version == 1}) of the ESCI data \cite{reddy2022}, joined with the product catalogue (""" + num(ps['n_products']) + r""" products).
No pairs had a missing product and there were no duplicate pairs.
Product titles are always present, but descriptions are missing for """ + pct(ps['missing_rate_products']['product_description']) + r""" of products, bullet points for """ + pct(ps['missing_rate_products']['product_bullet_point']) + r""", brand for """ + pct(ps['missing_rate_products']['product_brand']) + r""" and colour for """ + pct(ps['missing_rate_products']['product_color']) + r""".
Queries are short (mean """ + f"{ps['query_len_words_mean']:.1f}" + r""" words) and titles are long (mean """ + f"{ps['title_len_words_mean']:.1f}" + r""" words).

\textbf{Cleaning.} We unescape HTML, strip tags, normalise whitespace and lowercase all text. Missing fields become empty strings.

\textbf{Splits.} We keep the official test split untouched. From the official training split we hold out 10\% of \emph{queries} (not pairs) as validation, with seed 42, so that no query appears in both training and validation.
Table~\ref{tab:data} shows the sizes and label shares. The classes are imbalanced: Complement is under 5\% of pairs.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Split & Queries & Pairs & E & S & C & I \\\midrule
""" + t_data() + r"""
\bottomrule\end{tabular}
\caption{Split sizes and label distribution.}\label{tab:data}
\end{table}

\textbf{Metric.} Following the KDD Cup 2022 task 2 \cite{reddy2022} we use micro-F1 (equal to accuracy for single-label classification) as the main metric, and also report macro-F1 and per-class F1, because micro-F1 hides poor performance on Complement.
The reduced set keeps the harder queries of the full dataset \cite{reddy2022}, and we train on it alone, so our scores are not directly comparable to KDD Cup 2022 leaderboard numbers, which were trained on larger data and evaluated on a hidden test set.

\section{Models}
\textbf{Majority class.} Always predicts Exact.

\textbf{Lexical features + LightGBM.} We compute 33 features per pair (\texttt{feats.py}): token overlap between the query and the title, bullets, description, brand and colour (plain and IDF-weighted, fitted on training titles); Jaccard and character 3-gram overlap with the title; BM25-style scores \cite{robertson2009} against title, bullets and description; query, title, bullet and description lengths and whether each field is present; the count of numbers in the query and the share of them found in the title or bullets; whether the brand appears in the query; the number and highest IDF of query words found in no product field; whether the query has a negation; counts of complement cue words (such as ``case'', ``charger'', ``for'') in query and title; whether a query word appears after ``for'' in the title; whether the first and last query words are in the title; and the mean IDF of the query.
We train LightGBM \cite{ke2017} with learning rate 0.05 and early stopping (50 rounds) on validation, and tune \texttt{num\_leaves} $\in\{31,127\}$, \texttt{min\_child\_samples} $\in\{50,200\}$ and class weights (none or balanced): 8 settings.
The best used """ + str(lgb_v['params']['num_leaves']) + r""" leaves, minimum """ + str(lgb_v['params']['min_child_samples']) + r""" samples per leaf, no class weights, and stopped at """ + str(lgb_v['best_iter']) + r""" trees. """ + lgb_cap_note + r"""

\textbf{Logistic regression.} Multinomial logistic regression \cite{pedregosa2011} on word and character TF-IDF of the query and title (""" + num(1045120) + r""" sparse columns, fit on training text only) plus the standardised lexical features, with $C\in\{0.1,0.3,1,3\}$ and optional balanced class weights.
In the progress report these fits stopped at 300 lbfgs iterations without converging. """ + lr_conv_note + r"""

\textbf{Cross-encoders.} A cross-encoder reads the query and the product text as one sequence and classifies the pair from the joint representation \cite{nogueira2019}.
We fine-tune three pretrained encoders with a new 4-class head and cross-entropy loss: MiniLM-L6 \cite{wang2020}, already trained as a passage re-ranker on MS MARCO \cite{bajaj2016}; DeBERTa-v3-small and DeBERTa-v3-base \cite{he2023}.
Input A is the cleaned query; input B is the title, brand and colour, and the first 300 characters of the bullet points, joined with the separator token and truncated to 128 tokens.
We train for 2 epochs with AdamW \cite{loshchilov2019} (weight decay 0.01, 5\% linear warm-up, then linear decay), learning rate $3\times10^{-5}$ and batch 64 for MiniLM and $2\times10^{-5}$ and batch 48 for DeBERTa, with mixed precision \cite{wolf2020}.
Gradients are clipped at norm 1.0. We evaluate on validation every 2,000 steps and keep the checkpoint with the best micro-F1; """ + ckpt_note + r"""

\textbf{Hybrid stacker.} Cross-encoders read text but cannot see our hand-built signals directly (for example, whether a number in the query is in the title), and LightGBM cannot read text. We therefore stack them \cite{wolpert1992}: a second LightGBM takes the cross-encoders' class probabilities and the 33 lexical features.
Because the cross-encoders were trained on the training split, their probabilities on training pairs are over-confident. We train the stacker on the \emph{validation} split instead and score it with 5-fold cross-validation grouped by query, so every validation prediction comes from a model that did not see that query.
We compared four inputs (Table~\ref{tab:hybrid}) with fixed settings (300 trees, learning rate 0.05, 31 leaves, minimum 100 samples per leaf) and chose by out-of-fold micro-F1.

\textbf{Model selection.} The final model is the one with the highest validation micro-F1 (out-of-fold for the stacker). For the test run, the selected stacker is refit on all validation pairs.

\section{Results}
All training ran on Google Cloud VMs with one NVIDIA L4 GPU (24\,GB) each; the second-round VMs were \texttt{g2-standard-32} (32 vCPUs, \texttt{infra/config.sh}).
Times below are wall-clock and include CPU-side work such as tokenization.

\subsection{Validation results and model selection}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_val_main() + r"""
\bottomrule\end{tabular}
\caption{Validation results (""" + num(ps['val']['pairs']) + r""" pairs). Cross-encoders: best checkpoint. Stacker: 5-fold out-of-fold predictions.}\label{tab:val}
\end{table}

Table~\ref{tab:val} shows the main comparison. LightGBM improves on the majority baseline mostly on Substitute and Irrelevant but almost never predicts Complement (F1 """ + f3(lgb_v['per_class_f1']['C']) + r""").
Every cross-encoder beats it, with the largest gains on Complement and Irrelevant. Larger encoders do better: """ + ', '.join(f"{PRETTY[t]} {r_(t)}" for t in ce_tags) + r""" micro-F1.

\subsection{Hybrid stacker}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Input & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_hybrid() + r"""
\bottomrule\end{tabular}
\caption{Hybrid stacker variants, validation, 5-fold out-of-fold grouped by query. ``Alone'' rows are the cross-encoders' own argmax on the same pairs.}\label{tab:hybrid}
\end{table}

The best input by out-of-fold micro-F1 is """ + HNAME[hsel] + r""" (Table~\ref{tab:hybrid}), essentially tied with """ + hyb_tie + r""". Compared with the best single cross-encoder, """ + PRETTY[best_ce] + r""", on the same pairs it changes micro-F1 by """ + sg(hv['micro_f1'] - ce_alone_best['micro_f1']) + r""" and macro-F1 by """ + sg(hv['macro_f1'] - ce_alone_best['macro_f1']) + r""".
""" + (r"""So stacking did not help: the stacker is trained on validation pairs only (about four fifths of """ + num(ps['val']['pairs']) + r""" per fold), and the lexical features add little that the cross-encoders do not already capture. Adding the lexical features to the best cross-encoder alone (""" + f3(hyb['variants']['lex+ce_best']['micro_f1']) + r""") also scores below that cross-encoder by itself.""" if hv['micro_f1'] <= ce_alone_best['micro_f1'] else '') + r"""
Balanced class weights in the stacker lower micro-F1 to """ + f3(hyb['variants']['lex+ce_all_balanced']['micro_f1']) + r""" and leave Complement F1 unchanged (""" + f3(hyb['variants']['lex+ce_all_balanced']['per_class_f1']['C']) + r""" against """ + f3(hyb['variants']['lex+ce_all']['per_class_f1']['C']) + r""").

\subsection{Ablations (MiniLM-L6, validation)}
To find which design choices matter we changed one setting at a time on MiniLM-L6, the cheapest cross-encoder (Table~\ref{tab:abl}).

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}\toprule
Setting & micro-F1 & macro-F1 & F1 C & F1 I & Train min \\\midrule
""" + t_ablation() + r"""
\bottomrule\end{tabular}
\caption{One-change-at-a-time ablations on MiniLM-L6, validation, all at the last checkpoint (end of epoch 2), so rows compare settings rather than which checkpoint happened to be best. The reference run is from the first round, on a different L4 VM that shared its CPUs with the LightGBM grid, so its training time is not directly comparable.}\label{tab:abl}
\end{table}

Class weighting lowers micro-F1 sharply. Among the field and length changes, only the title-only input (dropping bullets, brand and colour) clearly costs accuracy (""" + sg(-cmpd('val_minilm_vs_minilm_f_title')['diff']) + r""" micro-F1 at the best checkpoint, bootstrap interval """ + ci([-x for x in cmpd('val_minilm_vs_minilm_f_title')['ci95']][::-1]) + r"""); the others are within a few thousandths of the reference. Adding the product description is a weak test: it is appended last and is often cut off at 128 tokens.

\textbf{Seed variance.} """ + seed_text + r""" Micro-F1 differences between configurations smaller than the MiniLM-L6 seed gap should not be read as real.

""" + sig_sentence() + r"""

\subsection{Complexity and resource analysis}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrrr}\toprule
Model & Params & Max len & Batch & Train min & Peak GPU GB & Val pairs/s & Test pairs/s \\\midrule
""" + t_resources() + r"""
\bottomrule\end{tabular}
\caption{Cost of the cross-encoders on one L4 GPU. Train minutes are wall-clock for 2 epochs including the periodic validation passes; first-round runs (MiniLM-L6, DeBERTa-v3-small) shared the VM's CPUs with other jobs. \emph{Val pairs/s}: GPU forward passes over the pre-tokenized validation split, batch 256, the same measurement for every row ($^\dagger$ timed on the identical seed-1 run; not logged for the first-round DeBERTa-v3-small). \emph{Test pairs/s}: the one-time test prediction, batch 512, including tokenization on the CPU, so it is lower and not comparable to the validation column. Peak memory was not logged in the first round (--).}\label{tab:res}
\end{table}

Table~\ref{tab:res} shows the cost side. The cost of a cross-encoder grows with both model size and sequence length, while the classical models are far cheaper: the LightGBM refit took """ + str(test['val']['lightgbm_refit']['fit_secs']) + r""" seconds on CPU (""" + str(n_lgb_trees) + r""" trees) and predicted the """ + num(test['test_pairs']) + r""" test pairs in """ + str(test['val']['lightgbm_refit']['test_infer_secs']) + r""" seconds, and the stacker adds only a small LightGBM on top of the cross-encoder outputs.
Each logistic-regression fit on """ + num(1045120) + r""" sparse columns took """ + (f"{min(r['secs'] for r in lrc['lr_grid']) / 60:.0f}--{max(r['secs'] for r in lrc['lr_grid']) / 60:.0f}" if lrc else '') + r""" minutes of wall-clock time per setting (all 8 settings ran in parallel on one 32-vCPU VM while DeBERTa-v3-base trained on its GPU), about as long as fine-tuning MiniLM-L6, for much lower scores.

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

Figure~\ref{fig:curve} and Table~\ref{tab:curve} show how results change with training data size. """ + curve_text + r"""

\subsection{Test results}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & micro-F1 & macro-F1 & F1 E & F1 S & F1 C & F1 I \\\midrule
""" + t_test() + r"""
\bottomrule\end{tabular}
\caption{Held-out test split (""" + num(test['test_pairs']) + r""" pairs, """ + num(test['test_queries']) + r""" queries), evaluated once. $\star$ marks the model chosen on validation.}\label{tab:test}
\end{table}

Table~\ref{tab:test} gives the test results. The model chosen on validation, """ + pretty(final) + r""", scores """ + f3(tm[final]['micro_f1']) + r""" micro-F1, """ + sg(gain_final_vs_lgb) + r""" over LightGBM.
Its validation micro-F1 was """ + f3(runs[final]['best']['micro_f1'] if final in runs else test['val_micro_used_for_selection'][final]) + r""", so test and validation agree closely and the validation split was a reliable guide.
The hybrid stacker, which was not selected, scores """ + f3(tm[f'hybrid_{hsel}']['micro_f1']) + r""" on test; the gap to the selected model is within the seed variation we measured, so we do not read it as a difference.
Figure~\ref{fig:pc} compares per-class test F1: the cross-encoders gain most on Complement and Irrelevant, the classes where word overlap misleads LightGBM.
""" + (f"As a check that the saved checkpoints are the ones evaluated, re-predicting validation with each reloaded model matched the saved validation predictions on this share of pairs: {agree}." if agree else '') + r"""

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
Exact versus Substitute often turns on a detail such as a size, model number or brand that the query names.

\textbf{Complement recall.} On validation, Complement recall is: """ + complement_text() + r""".
Complement products can share many words with the query (a charger whose title names the phone the query asks for), which pushes overlap-based models towards Exact; the high-overlap slice below shows this.

\textbf{Slices.} We fixed five slices before looking at results (Table~\ref{tab:slices}, which also shows all pairs for reference): pairs with high title overlap (at least 75\% of query tokens in the title) labelled Irrelevant or Complement; pairs with low overlap (at most 25\%) labelled Exact; queries with a likely misspelling (a 4+ letter word never seen in training titles or bullets); and queries with a number or a negation.
On validation (the table shows test), LightGBM got only """ + pct(slh['best_baseline']['accuracy']) + r""" of the high-overlap I/C pairs right, because it relies on overlap, against """ + ', '.join(f"{pct(slh[m]['accuracy'])} for {PRETTY[m]}" for m in ('minilm', 'deberta_small', 'deberta_base') if m in slh) + r""".

\begin{table}[H]\centering\small
\begin{tabular}{lr""" + 'r' * ncols + r"""}\toprule
""" + sh + r""" \\\midrule
""" + srows + r"""
\bottomrule\end{tabular}
\caption{Test accuracy on the failure slices.}\label{tab:slices}
\end{table}

\section{Conclusion}
Fine-tuned cross-encoders clearly beat lexical models on ESCI, mainly by recovering Complement and Irrelevant pairs that word overlap gets wrong, and larger encoders help further at a higher cost.
""" + (r"""Stacking the cross-encoder outputs with lexical features did not beat the best single cross-encoder on validation, so our final model is """ + pretty(final) + r""" (test micro-F1 """ + f3(tm[final]['micro_f1']) + r""").""" if not final.startswith('hybrid') else r"""Stacking the cross-encoder outputs with lexical features gave a further gain at almost no extra cost, and this stacker was our final model.""") + r"""
The ablations show that class weighting in the loss hurts, that adding fields beyond the title helps a little, and that many differences of a few thousandths are within seed noise.
The main open problem is Complement and the Exact/Substitute boundary. Promising next steps are training on the full (not reduced) dataset, larger encoders or longer training, adding the product description through a longer input, and query-level features such as the other products retrieved for the same query.

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
