# CMPS261 project: Amazon ESCI query–product relevance

Authors: Allie Saoud, Zulfiquar Fneiche. Course option 4.2.

Task: given a shopping query and a product, predict Exact, Substitute, Complement or Irrelevant (ESCI).
Data: US locale, reduced set (`small_version == 1`). Metric: micro-F1 (main), macro-F1 and per-class F1.

| What | Where |
|---|---|
| Final report (PDF and LaTeX source) | `report/final_report.pdf`, `report/final_report.tex` |
| Progress report | `report/CMPS261_progress_report.pdf` |
| Notebook with all code and results | `CMPS261_ESCI.ipynb` |
| Trained models | GitHub Release `v1.0` of this repository (see *Models* below) |
| All run outputs (metrics, logs, predictions) | `results/` |
| Cloud infrastructure scripts | `infra/` |

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Preprocess, split | `prep.py` | `data/{train,val,test}.parquet`, `data/prep_stats.json` |
| 2. Lexical features, TF-IDF | `feats.py` | `data/feat_*.parquet`, `data/tfidf_*.npz` |
| 3. Classical baselines (LightGBM, logistic regression) | `baseline.py` | `results/baseline.json` |
| 3b. Logistic regression until convergence | `lr_converge.py` | `results/lr_converged.json`, `models/lr_best.joblib` |
| 4. Pretrained cross-encoders (every configuration) | `xenc_pretrained.py`, driven by `run_vm.sh` and `run_experiments.sh` | `results/<tag>.json`, `results/<tag>_val_proba.npy`, `models/<tag>/` |
| 5. Failure slices on validation | `error_analysis.py` | `results/error_analysis.json` |
| 6. Hybrid stacker (validation, 5-fold by query) | `hybrid.py` | `results/hybrid.json` |
| 7. Test predictions, final models only | `predict_test.py` | `results/<tag>_test_proba.npy` |
| 8. One-time test evaluation | `final_test.py` | `results/test.json`, `models/lgb_best.txt`, `models/hybrid_lgb.txt` |
| 9. Figures, report, notebook | `make_figures.py`, `make_figures_final.py`, `build_report.py`, `build_final_report.py`, `make_notebook.py` | `figures/`, `report/`, `CMPS261_ESCI.ipynb` |

`xenc.py` is an earlier cross-encoder trained from scratch. It is not used in the reports.

## Infrastructure (GCP)

All training ran on Google Cloud Compute Engine VMs with NVIDIA L4 GPUs. The scripts in `infra/` create, set up and remove them:

```bash
export PROJECT=<your-gcp-project>
bash infra/create.sh          # staging bucket + VM (g2-standard-32, 1 x L4, Ubuntu 22.04 + NVIDIA 580 driver)
bash infra/push_code.sh       # copy this repository to the VM
# on the VM, in ~/esci:
bash infra/setup_vm.sh        # apt packages, venv, pinned torch/transformers, raw data download, prep.py, feats.py
QUEUES="a b" LR=1 FINAL=1 bash run_experiments.sh   # all second-round experiments and the test evaluation
# back on your machine:
bash infra/fetch.sh           # results/, models/, report/ through the bucket
bash infra/teardown.sh        # delete the VM and the bucket
```

For the report we ran queue `a` and logistic regression on VM `esci-cmps261` and queue `b` on VM `esci-cmps261-b` (one L4 each, because 2- and 4-GPU L4 machines were out of stock), then the final step on `esci-cmps261`.
The first-round runs (progress report) used `run_vm.sh` and `run_rest.sh` on one L4 VM.

## Reproduce on any machine

1. Download the data into `raw/`:

   ```bash
   mkdir -p raw
   curl -L -o raw/examples.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
   curl -L -o raw/products.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet
   ```

   `results/raw_sha256.txt` has the checksums of the files we used.

2. Environment (Python 3.10; LightGBM on Linux needs `libgomp1`):

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   ```

   `results/pip_freeze.txt` lists every package version on the training VM.

3. Run the pipeline in the order of the table above. `run_vm.sh` (first round) and `run_experiments.sh` (second round) show the exact commands and settings of every run.

## Settings and seeds

- Validation split: 10% of training queries, grouped by query, `numpy.random.RandomState(42)` in `prep.py`. The official test split was used once, at the end.
- LightGBM grid: `num_leaves` {31, 127} × `min_child_samples` {50, 200} × `class_weight` {None, balanced}; learning rate 0.05, early stopping 50 rounds; LightGBM's built-in default seeds.
- Logistic regression grid: `C` {0.1, 0.3, 1, 3} × `class_weight` {None, balanced}, lbfgs; `max_iter=300` in the first round, 3,000 in `lr_converge.py`.
- Cross-encoders: seed 0 via `transformers.set_seed` (seed 1 for the variance runs), max length 128, 2 epochs, AdamW, 5% warmup, validation every 2,000 steps, best checkpoint by validation micro-F1. Learning rate 3e-5 / batch 64 for MiniLM, 2e-5 / batch 48 for DeBERTa.
- Learning-curve subsets: training queries sampled with `RandomState(0)`.
- Hybrid: LightGBM with `random_state=0`, 5-fold `GroupKFold` by query on validation.
- GPU training with mixed precision is not bitwise deterministic, so reruns can differ slightly in the last digits.
- The `deberta_small_seed1` run ended early, after its validation pass at step 2,000 of 15,730, without an error message (`results/logs/deberta_small_seed1.log`). The report compares seeds at that shared checkpoint.
- Results: final model DeBERTa-v3-base (chosen on validation), test micro-F1 0.608, macro-F1 0.555 (`results/test.json`).

## Models

The GitHub Release `v1.0` holds the trained models:

- `deberta_base.tar.gz`, `deberta_small.tar.gz`, `minilm.tar.gz`: fine-tuned cross-encoders (Hugging Face format; load the tokenizer from the base model named in `results/<tag>.json`).
- `lgb_best.txt`: LightGBM baseline. `hybrid_lgb.txt`: hybrid stacker. `lr_best.joblib`: logistic regression with its feature scaler.
