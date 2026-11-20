# CMPS261 project: Amazon ESCI query–product relevance

Authors: Allie Saoud, Zulfiquar Fneiche. Course option 4.2.

Task: given a shopping query and a product, predict Exact, Substitute, Complement or Irrelevant (ESCI).
Data: US locale, reduced set (`small_version == 1`). Metric: micro-F1 (main), macro-F1 and per-class F1.

| What | Where |
|---|---|
| Final report | `report/final_report.pdf` |
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
| 8b. Bootstrap intervals (CPU, seconds) | `significance.py` | `results/significance.json` |
| 9. Figures, notebook | `make_figures.py`, `make_figures_final.py`, `make_notebook.py` | `figures/`, `CMPS261_ESCI.ipynb` |

`xenc.py` is an earlier cross-encoder trained from scratch. It is not used in the reports.

## Infrastructure (GCP)

All training ran on Google Cloud Compute Engine VMs with NVIDIA L4 GPUs. The scripts in `infra/` create, set up and remove them:

```bash
export PROJECT=<your-gcp-project>
bash infra/create.sh          # staging bucket + VM (g2-standard-32, 1 x L4, Ubuntu 22.04 + NVIDIA 580 driver)
bash infra/push_code.sh       # copy this repository to the VM
# on the VM, in ~/esci:
bash infra/setup_vm.sh        # apt packages, venv, pinned torch/transformers, raw data download, prep.py, feats.py
bash run_vm.sh                # first round: LightGBM/LR grid, MiniLM-L6 and DeBERTa-v3-small reference runs
QUEUES=a LR=1 bash run_experiments.sh   # second round, queue a (DeBERTa-v3-base, length 256) + LR rerun
# back on your machine:
bash infra/fetch.sh           # results/, models/, report/ through the bucket
bash infra/teardown.sh        # delete the VM and the bucket
```

`run_experiments.sh` runs one queue per GPU and stops if the machine has fewer GPUs than queues.
On 1-GPU VMs, run queue `b` (seeds, ablations, learning curve) on a second VM the same way (`VM=esci-cmps261-b bash infra/create.sh`, then `QUEUES=b bash run_experiments.sh` there).
Copy its `results/` into the first VM's `results/` (all file names differ between the queues), then run the final step there:

```bash
QUEUES="" FINAL=1 bash run_experiments.sh   # test predictions, failure slices, hybrid CV and the one-time test evaluation
python significance.py && python make_figures_final.py && python make_notebook.py
```

To skip the first round, download `minilm.tar.gz` and `deberta_small.tar.gz` from release `v1.0` and unpack them into `models/`; the final step needs those checkpoints and `results/baseline.json`, `minilm.json` and `deberta_small.json`, which are in this repository.

For the report we ran queue `a` and logistic regression on VM `esci-cmps261` and queue `b` on VM `esci-cmps261-b` (one L4 each, because 2- and 4-GPU L4 machines were out of stock), then the final step on `esci-cmps261`.
The first-round runs (progress report) used `run_vm.sh` on one L4 VM. Its DeBERTa-v3-small run crashed with `ValueError: Attempting to unscale FP16 gradients.` (status line in `results/run_vm.out`); after the fix in `xenc_pretrained.py` we reran it with `run_rest.sh`. The committed `run_vm.sh` works with the fixed code.
In `results/logs/queue_status.txt`, lines starting with `B:` were copied from VM `esci-cmps261-b` when we merged the two VMs' results.

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

   `requirements.txt` pins torch and transformers. For the exact versions of every other package on the training VM, install from `results/pip_freeze.txt` instead (`pip install -r results/pip_freeze.txt`).

3. Run the pipeline in the order of the table above. `run_vm.sh` (first round) and `run_experiments.sh` (second round) show the exact commands and settings of every run.

## Settings and seeds

- Validation split: 10% of training queries, grouped by query, `numpy.random.RandomState(42)` in `prep.py`. The official test split was used once, at the end.
- LightGBM grid: `num_leaves` {31, 127} × `min_child_samples` {50, 200} × `class_weight` {None, balanced}; learning rate 0.05, early stopping 50 rounds; LightGBM's built-in default seeds.
- Logistic regression grid: `C` {0.1, 0.3, 1, 3} × `class_weight` {None, balanced}, lbfgs; `max_iter=300` in the first round, 3,000 in `lr_converge.py`.
- Cross-encoders: seed 0 via `transformers.set_seed` (seed 1 for the variance runs), max length 128, 2 epochs, AdamW, 5% warmup, validation every 2,000 steps, best checkpoint by validation micro-F1. Learning rate 3e-5 / batch 64 for MiniLM, 2e-5 / batch 48 for DeBERTa.
- Learning-curve subsets: training queries sampled with `RandomState(0)`.
- Hybrid: LightGBM with `random_state=0`, 5-fold `GroupKFold` by query on validation.
- GPU training with mixed precision is not bitwise deterministic, so reruns can differ slightly in the last digits.
- Only a partial `deberta_small_seed1` run survives. Its log (`results/logs/deberta_small_seed1.log`) stops at training step 2,600 of 15,730 with no error, although `queue_status.txt` records exit 0. Its step times show it shared the GPU with other queue-b jobs, and VM `esci-cmps261-b` was deleted before we noticed, so the cause is unknown. The report compares seeds at step 2,000, the last validation pass it completed. Its training time, and those of `minilm_frac25` and `minilm_frac50`, which ran alongside it, are not clean timings.
- Uncertainty: `significance.py` runs a paired bootstrap over queries for the comparisons in the report (`results/significance.json`).
- Results: final model DeBERTa-v3-base (chosen on validation), test micro-F1 0.608, macro-F1 0.556 (`results/test.json`; rounded from the confusion matrix).

## Models

The GitHub Release `v1.0` holds the trained models:

- `deberta_base.tar.gz`, `deberta_small.tar.gz`, `minilm.tar.gz`: fine-tuned cross-encoders (Hugging Face format; load the tokenizer from the base model named in `results/<tag>.json`).
- `lgb_best.txt`: LightGBM baseline. `hybrid_lgb.txt`: hybrid stacker. `lr_best.joblib`: logistic regression with its feature scaler.
