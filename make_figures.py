"""Build the report's figures directly from the JSON files in results/."""
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
plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

def load(n):
    p = f'{R}/{n}.json'
    return json.load(open(p)) if os.path.exists(p) else None

base = load('baseline')
best_base = max([('LightGBM', base['best_lgb']), ('TF-IDF + LR', base['best_lr'])], key=lambda kv: kv[1]['micro_f1'])
models = [(f'Baseline ({best_base[0]})', best_base[1])]
for tag, label in (('minilm', 'MiniLM-L6 cross-encoder'), ('deberta_small', 'DeBERTa-v3-small cross-encoder')):
    r = load(tag)
    if r: models.append((label, r['best']))

# Figure 1: per-class F1 for each model
fig, ax = plt.subplots(figsize=(6.4, 2.7))
w = 0.8 / len(models); x = np.arange(4)
for i, (name, r) in enumerate(models):
    v = [r['per_class_f1'][l] for l in LAB]
    bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, v, w - 0.02, color=SERIES[i], label=f"{name}, micro-F1 {r['micro_f1']:.3f}")
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.012, f'{val:.2f}', ha='center', va='bottom', fontsize=7, color=INK2)
ax.set_xticks(x, NAMES); ax.set_ylabel('Validation F1'); ax.set_ylim(0, 1)
ax.yaxis.grid(True, color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED); ax.tick_params(length=0)
ax.legend(frameon=False, loc='upper right', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/per_class_f1.png', dpi=220); plt.close(fig)

# Figure 2: confusion matrix of the best model by validation micro-F1
name, r = max(models, key=lambda kv: kv[1]['micro_f1'])
C = np.array(r['confusion']); N = C / C.sum(1, keepdims=True)
cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fe', '#9ec5f4', '#3987e5', '#184f95'])
fig, ax = plt.subplots(figsize=(3.9, 3.2))
im = ax.imshow(N, cmap=cmap, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{C[i, j]:,}\n{N[i, j]:.0%}', ha='center', va='center', fontsize=7.5,
                color='#ffffff' if N[i, j] > 0.55 else INK)
ax.set_xticks(range(4), NAMES, fontsize=7.5); ax.set_yticks(range(4), NAMES, fontsize=7.5)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 4), minor=True)
ax.grid(which='minor', color=SURF, lw=2); ax.tick_params(which='both', length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f'{FIG}/confusion.png', dpi=220); plt.close(fig)
print('best model:', name, r['micro_f1'])
