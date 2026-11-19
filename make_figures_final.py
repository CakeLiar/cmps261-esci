"""Figures for the final report, built from results/*.json (test.json, hybrid.json and the run logs)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
LAB = ['E', 'S', 'C', 'I']; NAMES = ['Exact', 'Substitute', 'Complement', 'Irrelevant']
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#898781', '#ffffff'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
J = lambda n: json.load(open(f'{R}/{n}.json'))
test, hyb = J('test'), J('hybrid')
PRETTY = {'lightgbm': 'LightGBM', 'minilm': 'MiniLM-L6', 'deberta_small': 'DeBERTa-v3-small', 'deberta_base': 'DeBERTa-v3-base'}
hyb_key = f"hybrid_{test['hybrid_selected']}"
ce = [k for k in ('minilm', 'deberta_small', 'deberta_base') if k in test['models']]
best_ce = max(ce, key=lambda k: J(k)['best']['micro_f1'])   # chosen on validation
models = [('lightgbm', 'LightGBM baseline'), (best_ce, PRETTY[best_ce]), (hyb_key, 'Hybrid (LightGBM stacker)')]

def style(ax):
    ax.yaxis.grid(True, color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(MUTED); ax.tick_params(length=0)

# Figure 1: per-class test F1
fig, ax = plt.subplots(figsize=(6.4, 2.7)); w = 0.8 / len(models); x = np.arange(4)
for i, (k, name) in enumerate(models):
    r = test['models'][k]; v = [r['per_class_f1'][l] for l in LAB]
    bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, v, w - 0.02, color=SERIES[i], label=f"{name}, micro-F1 {r['micro_f1']:.3f}")
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f'{val:.2f}', ha='center', va='bottom', fontsize=7, color=INK2)
ax.set_xticks(x, NAMES); ax.set_ylabel('Test F1'); ax.set_ylim(0, 1); style(ax)
ax.legend(frameon=False, loc='upper right', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/final_test_per_class_f1.png', dpi=220); plt.close(fig)

# Figure 2: test confusion matrix of the final model (selected on validation)
final = test.get('final_model', hyb_key)
C = np.array(test['models'][final]['confusion']); N = C / C.sum(1, keepdims=True)
cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fe', '#9ec5f4', '#3987e5', '#184f95'])
fig, ax = plt.subplots(figsize=(3.9, 3.2)); ax.imshow(N, cmap=cmap, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{C[i, j]:,}\n{N[i, j]:.0%}', ha='center', va='center', fontsize=7.5, color='#ffffff' if N[i, j] > 0.55 else INK)
ax.set_xticks(range(4), NAMES, fontsize=7.5); ax.set_yticks(range(4), NAMES, fontsize=7.5)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 4), minor=True)
ax.grid(which='minor', color=SURF, lw=2); ax.tick_params(which='both', length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f'{FIG}/final_test_confusion.png', dpi=220); plt.close(fig)

# Figure 3: learning curve (MiniLM, validation, last checkpoint so every point is chosen the same way)
pts = []
for frac, tag in ((0.1, 'minilm_frac10'), (0.25, 'minilm_frac25'), (0.5, 'minilm_frac50'), (1.0, 'minilm')):
    j = J(tag); last = j['log'][-1]
    pts.append((j['config'].get('train_pairs') or J('prep_stats')['train']['pairs'], last['micro_f1'], last['macro_f1']))
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for i, (idx, name) in enumerate(((1, 'micro-F1'), (2, 'macro-F1'))):
    xs = [p[0] for p in pts]; ys = [p[idx] for p in pts]
    ax.plot(xs, ys, color=SERIES[i], lw=2, marker='o', ms=5, label=name)
    for xv, yv in zip(xs, ys):
        ax.text(xv, yv + 0.008, f'{yv:.3f}', ha='center', va='bottom', fontsize=7, color=INK2)
ax.set_xscale('log'); ax.set_xticks([p[0] for p in pts], [f'{p[0] // 1000}k' for p in pts])
ax.set_xlabel('Training pairs (log scale)'); ax.set_ylabel('Validation F1'); style(ax)
ax.legend(frameon=False, loc='lower right', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/final_learning_curve.png', dpi=220); plt.close(fig)
print('figures written; final model', final)
