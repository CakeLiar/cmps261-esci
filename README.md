# CMPS261 project: Amazon ESCI query–product relevance

Authors: Allie Saoud, Zulfiquar Fneiche. Course option 4.2.

Task: given a shopping query and a product, predict Exact, Substitute, Complement or Irrelevant (ESCI).
Data: US locale, reduced set (`small_version == 1`). Metric: micro-F1 (main), macro-F1 and per-class F1.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Preprocess, split | `prep.py` | `data/{train,val,test}.parquet`, `data/prep_stats.json` |
| 2. Lexical features, TF-IDF | `feats.py` | `data/feat_*.parquet`, `data/tfidf_*.npz` |
| 3. Classical baselines (LightGBM, logistic regression) | `baseline.py` | `results/baseline.json`, `results/{lgb,lr}_val_proba.npy` |
| 4. Pretrained cross-encoders | `xenc_pretrained.py` | `results/{minilm,deberta_small}.json`, `*_val_proba.npy`, `models/` |
| 5. Error slices | `error_analysis.py` | `results/error_analysis.json` |
| 6. Figures | `make_figures.py` | `figures/*.png` |
| 7. Report | `build_report.py` | `report.html` (printed to PDF with Chrome) |

`xenc.py` is an earlier cross-encoder trained from scratch. It is not used in the report.

## Reproduce

1. Download the two dataset files from [amazon-science/esci-data](https://github.com/amazon-science/esci-data) into `raw/`:

   ```bash
   mkdir -p raw
   curl -L -o raw/examples.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
   curl -L -o raw/products.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet
   ```

2. Set up the environment (Python venv in `.venv`; LightGBM on Linux also needs `libgomp1`):

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Preprocess, then copy the stats file where the report reads it:

   ```bash
   python prep.py && mkdir -p results && cp data/prep_stats.json results/
   ```

4. Run features, baselines, both cross-encoders and the error analysis. This needs a CUDA GPU:

   ```bash
   bash run_vm.sh
   ```

   `run_rest.sh` was only used to rerun DeBERTa after a mixed-precision fix. That fix is already in `xenc_pretrained.py`, so `run_vm.sh` alone runs everything.

5. Build figures and the report:

   ```bash
   python make_figures.py && python build_report.py
   ```

## Settings and seeds

- Validation split: 10% of training queries, grouped by query, `numpy.random.RandomState(42)` in `prep.py`. The official test split is held out.
- LightGBM: grid of `num_leaves` {31, 127} × `min_child_samples` {50, 200} × `class_weight` {None, balanced}; learning rate 0.05, early stopping 50 rounds. No `random_state` is passed, so LightGBM's built-in default seeds apply.
- Logistic regression: `C` {0.1, 0.3, 1, 3} × `class_weight` {None, balanced}, lbfgs, `max_iter=300` (no fit converged).
- Cross-encoders (`xenc_pretrained.py`, seed 0 via `transformers.set_seed`), max length 128, 2 epochs, AdamW, 5% warmup, validation every 2,000 steps:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`: learning rate 3e-5, batch 64.
  - `microsoft/deberta-v3-small`: learning rate 2e-5, batch 48.
- Hardware used: GCP g2-standard-32 with one NVIDIA L4, torch 2.14.1+cu130, transformers 5.20.2. Other package versions were not recorded.
- GPU training with mixed precision is not bitwise deterministic, so reruns may differ slightly in the last digits.
