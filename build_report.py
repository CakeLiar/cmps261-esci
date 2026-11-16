"""Write report.html from results/*.json, so every result number in the report traces to a saved run."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, 'results')
L = lambda n: json.load(open(f'{R}/{n}.json'))
prep, base, mini, deb, ea = L('prep_stats'), L('baseline'), L('minilm'), L('deberta_small'), L('error_analysis')
lgb, lr, m, d, maj = base['best_lgb'], base['best_lr'], mini['best'], deb['best'], base['majority']
lgb_bal = max((r for r in base['lgb_grid'] if r['params']['class_weight']), key=lambda r: r['macro_f1'])
lr_bal = max((r for r in base['lr_grid'] if r['params']['class_weight']), key=lambda r: r['macro_f1'])
assert not any(r['converged'] for r in base['lr_grid'])
S = ea['slices']; C = ea['complement']

f3 = lambda x: f'{x:.3f}'
pct = lambda x: f'{100 * x:.1f}%'
n = lambda x: f'{x:,}'
pts = lambda a, b: f'{100 * (a - b):.1f}'

def row(name, r, bold=False):
    c = r['per_class_f1']
    cells = [name, f3(r['micro_f1']), f3(r['macro_f1']), f3(c['E']), f3(c['S']), f3(c['C']), f3(c['I'])]
    tag = 'b' if bold else 'span'
    return '<tr>' + ''.join(f'<td><{tag}>{v}</{tag}></td>' for v in cells) + '</tr>'

def maj_row():
    c = maj['per_class_f1']
    return '<tr>' + ''.join(f'<td>{v}</td>' for v in ['Majority (always E)', f3(maj['micro_f1']), f3(maj['macro_f1']),
                                                       f3(c['E']), f3(c['S']), f3(c['C']), f3(c['I'])]) + '</tr>'

def slice_row(label, key):
    s = S[key]
    return (f"<tr><td>{label}</td><td>{n(s['n_pairs'])}</td><td>{pct(s['share'])}</td><td>{f3(s['best_baseline']['accuracy'])}</td>"
            f"<td>{f3(s['minilm']['accuracy'])}</td><td>{f3(s['deberta_small']['accuracy'])}</td></tr>")

tr, va, te = prep['train'], prep['val'], prep['test']
miss = prep['missing_rate_products']
ld = tr['label_dist']
gain = dict(base['best_lgb_importance_gain'])
top = [k for k, _ in base['best_lgb_importance_gain'][:5]]
NAMES = {'char_cov': 'character-trigram query coverage', 'ov_any': 'any-field word overlap', 'q_mean_idf': 'mean query IDF',
         'max_missing_idf': 'highest IDF among missing query words', 't_comp_words': 'accessory words in the title'}
top_txt = ', '.join(NAMES[k] for k in top)
dc, lc = d['confusion'], lgb['confusion']
hi = S['high_overlap_labelled_I_or_C']; lo = S['low_overlap_labelled_E']

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CMPS261 Progress Report</title><style>
@page {{ size: letter; margin: 0.7in 0.8in; }}
body {{ font: 10.2pt/1.19 "Times New Roman", Times, serif; color: #000; margin: 0; }}
h1 {{ font-size: 15pt; font-weight: bold; text-align: center; margin: 0 0 3pt; }}
.authors {{ text-align: center; font-size: 11pt; margin: 0 0 1pt; }}
.course {{ text-align: center; margin: 0 0 7pt; }}
h2 {{ font-size: 11pt; font-weight: bold; margin: 6pt 0 2pt; }}
p {{ margin: 0 0 3.5pt; text-align: justify; }}
ul {{ margin: 1pt 0 3.5pt 16pt; padding: 0; }} li {{ margin-bottom: 1pt; text-align: justify; }}
code {{ font: 9.3pt "Courier New", Courier, monospace; }}
table {{ border-collapse: collapse; margin: 3pt auto 4pt; font-size: 9.2pt; }}
th, td {{ border: 0.5pt solid #000; padding: 0.8pt 5pt; text-align: center; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ font-weight: bold; }}
.figs {{ break-before: page; }} .figs img {{ display: block; margin: 0 auto; }}
.cap {{ text-align: center; margin: 3pt 0 14pt; }}
</style></head><body>
<h1>CMPS261 Project Progress Report: Amazon ESCI Query&#8211;Product Relevance</h1>
<p class="authors">Allie Saoud and Zulfiquar Fneiche</p>
<p class="course">CMPS261, Fall 2026&#8211;27, Project Option 4.2</p>

<h2>1. Problem Statement</h2>
<p>We chose option 4.2 from the course project list and did not change the problem. Given a shopping query and a product, the goal is to predict one of four labels: Exact, Substitute, Complement or Irrelevant (ESCI). We use the US locale of the reduced dataset (<code>small_version = 1</code>), which is harder and more balanced than the full set. All results in this report are on our validation split. We have not used the test split yet.</p>

<h2>2. Data Preprocessing</h2>
<p>Our subset has {n(prep['pairs_raw'])} query&#8211;product pairs over {n(prep['n_products'])} products. There are no duplicate pairs and every pair has its product. The official test split ({n(te['pairs'])} pairs, {n(te['queries'])} queries) is held out. For validation we took 10% of the training queries (seed 42) and grouped by query so that the same query never appears in two splits. This leaves {n(tr['pairs'])} training pairs from {n(tr['queries'])} queries and {n(va['pairs'])} validation pairs from {n(va['queries'])} queries. The training labels are E {pct(ld['E'])}, S {pct(ld['S'])}, I {pct(ld['I'])} and C {pct(ld['C'])}.</p>
<p>For cleaning, we unescape HTML, remove tags, normalize whitespace and lowercase the text. Bullets and descriptions are cut to 1,000 characters. Many product fields are missing (description {pct(miss['product_description'])}, color {pct(miss['product_color'])}, bullets {pct(miss['product_bullet_point'])}, brand {pct(miss['product_brand'])}, title 0%), so we leave missing text empty and use field presence as a feature. Queries have {prep['query_len_words_mean']:.1f} words on average and titles have {prep['title_len_words_mean']:.1f}.</p>
<p>For the classical models we built 33 lexical features: field overlap (plain and IDF-weighted), word and character-trigram similarity, BM25 for each field, number matching, brand, negation and accessory cues, lengths and field presence. The linear model also gets 1,045,120 sparse TF-IDF columns. IDF values and vectorizers are fit on the training data only.</p>

<h2>3. Evaluation Metrics</h2>
<p>Our main metric is <b>micro-F1</b>, which is the classification metric reported with the ESCI dataset and is the same as accuracy when each pair has one label. Published Task 2 scores use the larger dataset, so our numbers cannot be compared to them directly. We also report <b>macro-F1</b> and <b>per-class F1</b> because Complement is only {pct(ld['C'])} of the pairs, and micro-F1 alone would not show a model that almost never predicts it.</p>

<h2>4. Classical Baseline</h2>
<p>We tuned LightGBM on the 33 features with 8 settings (leaves in {{31, 127}}, min child samples in {{50, 200}}, class weights none or balanced, learning rate 0.05, early stopping at 50 rounds). We also tuned logistic regression on TF-IDF plus the scaled features with 8 settings (C in {{0.1, 0.3, 1, 3}}, class weights none or balanced). The best baseline by validation micro-F1 is LightGBM with {lgb['params']['num_leaves']} leaves, min child samples {lgb['params']['min_child_samples']}, no class weights and {lgb['best_iter']} trees. Its most important features by gain are {top_txt}.</p>
<p>Class weighting lowers micro-F1 but raises macro-F1: the best balanced LightGBM gets {f3(lgb_bal['micro_f1'])} micro and {f3(lgb_bal['macro_f1'])} macro. One problem is that all 8 logistic regression fits reached the 300-iteration limit without converging, so its scores should be seen as a lower bound. Early stopping also used the validation split.</p>
<table><tr><th>Validation F1</th><th>Micro</th><th>Macro</th><th>E</th><th>S</th><th>C</th><th>I</th></tr>
{maj_row()}
{row(f"LR, best micro (C={lr['params']['C']})", lr)}
{row("LightGBM (selected baseline)", lgb, True)}
{row(f"MiniLM-L6 cross-encoder (step {m['step']})", m)}
{row(f"DeBERTa-v3-small cross-encoder (step {d['step']})", d, True)}
</table>

<h2>5. Sophisticated Model</h2>
<p>For the sophisticated model we fine-tune pretrained cross-encoders with a new 4-class head. Text A is the query and text B is <i>title [SEP] brand color [SEP] first 300 characters of bullets</i>, with a max length of 128. We train for 2 epochs using cross-entropy, AdamW and 5% warmup, and we evaluate on validation every 2,000 steps. We expected this to beat the baseline because a cross-encoder reads the query and the product together, so it can pick up synonyms, paraphrases and compatibility phrases like &#8220;case for iPhone 14&#8221;, which overlap counts cannot do.</p>
<p>So far both models beat the baseline. DeBERTa-v3-small ({deb['config']['params'] / 1e6:.0f}M parameters, learning rate {f"{deb['config']['LR']:g}".replace('e-0', 'e-')}) gets micro-F1 {f3(d['micro_f1'])} and macro-F1 {f3(d['macro_f1'])}, which is {pts(d['micro_f1'], lgb['micro_f1'])} and {pts(d['macro_f1'], lgb['macro_f1'])} points above LightGBM. The biggest gain is on Complements, where F1 goes from {f3(lgb['per_class_f1']['C'])} to {f3(d['per_class_f1']['C'])}. MiniLM-L6 ({mini['config']['params'] / 1e6:.0f}M parameters) gets {f3(m['micro_f1'])} micro and {f3(m['macro_f1'])} macro. Training took {mini['log'][-1]['secs'] / 60:.0f} minutes for MiniLM and {deb['log'][-1]['secs'] / 60:.0f} minutes for DeBERTa on one NVIDIA L4 GPU, with the CPU baseline jobs running on the same machine. These are single-seed runs and the checkpoints were picked on validation, so the scores are slightly optimistic. The test split will give the final estimate.</p>

<h2>6. Remaining Plan and Risks</h2>
<p><b>Early failure observations.</b> The table shows accuracy on specific groups of validation pairs. We fixed the thresholds before looking at the results.</p>
<table><tr><th>Slice</th><th>Pairs</th><th>Share</th><th>LightGBM</th><th>MiniLM</th><th>DeBERTa</th></tr>
{slice_row('All validation pairs', 'all')}
{slice_row('High title overlap (&#8805;0.75), labelled I or C', 'high_overlap_labelled_I_or_C')}
{slice_row('Low title overlap (&#8804;0.25), labelled E', 'low_overlap_labelled_E')}
{slice_row('Query with a likely misspelling', 'likely_misspelled_query')}
{slice_row('Query contains a number', 'query_has_number')}
{slice_row('Query contains a negation word', 'query_has_negation')}
</table>
<ul>
<li><b>Shared words but wrong product:</b> on high-overlap I/C pairs LightGBM is correct only {pct(hi['best_baseline']['accuracy'])} of the time and predicts E for {n(hi['best_baseline']['pred_counts']['E'])} of {n(hi['n_pairs'])}. DeBERTa gets {pct(hi['deberta_small']['accuracy'])}. This is its largest gain of any slice, but it is still weak here.</li>
<li><b>Different wording:</b> on low-overlap Exact pairs MiniLM ({pct(lo['minilm']['accuracy'])}) does better than DeBERTa ({pct(lo['deberta_small']['accuracy'])}), so the best overall model does not win every slice.</li>
<li><b>Exact vs Substitute</b> is still the main confusion. DeBERTa labels {n(dc[0][1])} Exact pairs as Substitute and {n(dc[1][0])} Substitute pairs as Exact. These labels often differ only by one product attribute.</li>
<li><b>Complements:</b> recall goes from {pct(C['best_baseline']['recall'])} (LightGBM) to {pct(C['deberta_small']['recall'])} (DeBERTa), but DeBERTa still sends {n(C['deberta_small']['pred_counts']['E'])} of {n(C['deberta_small']['n'])} to Exact.</li>
<li><b>Misspellings</b> score below average for all models. Our detector is a heuristic (a query word that is not in the training titles or bullets), so it also catches rare names.</li>
</ul>
<p><b>Plan until 20 Nov.</b></p>
<ul>
<li><b>9&#8211;11 Nov (feature analysis):</b> rerun logistic regression until it converges, run field ablations (title only, + bullets, + description, + brand/color), and compute LightGBM feature-group importance.</li>
<li><b>11&#8211;15 Nov (complexity analysis):</b> compare model sizes (MiniLM-L6, DeBERTa-small, DeBERTa-base), max lengths 64/128/256, and a learning curve at 10/25/50/100% of the training data, and add a second seed for the best models.</li>
<li><b>15&#8211;18 Nov (improvements):</b> try a hybrid that gives cross-encoder probabilities and lexical features to LightGBM, try class weighting for Complements, and do a deeper error analysis.</li>
<li><b>19&#8211;20 Nov:</b> freeze the model, run the test split once and write the final report.</li>
</ul>
<p><b>Risks.</b> The main risks are GPU time for the larger models and ablations, the rare Complement class, noise in the Exact/Substitute labels (which may limit the F1 we can reach), reusing the validation set during tuning, and the schedule, since this report is already late.</p>

<div class="figs">
<img src="figures/per_class_f1.png" style="width:6.4in">
<p class="cap"><b>Figure 1:</b> Per-class validation F1 for the selected baseline and both cross-encoders.</p>
<img src="figures/confusion.png" style="width:3.9in">
<p class="cap"><b>Figure 2:</b> Validation confusion matrix of the best model (DeBERTa-v3-small). Each cell shows the pair count and the share of its true class.</p>
</div>
</body></html>
"""
open(f'{ROOT}/report.html', 'w').write(html)
print('wrote report.html')
