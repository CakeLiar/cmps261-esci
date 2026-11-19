#!/usr/bin/env bash
# Runs ON the VM from the project root: system packages, Python env, raw data, preprocessing, features.
# Usage: BUCKET=gs://... bash infra/setup_vm.sh
set -euo pipefail
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv libgomp1 >/dev/null
python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -q -r requirements.txt
mkdir -p results raw models
pip freeze > results/pip_freeze.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu_info.txt
for f in examples products; do
  curl -sSL -o raw/$f.parquet \
    https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_$f.parquet
done
sha256sum raw/*.parquet > results/raw_sha256.txt
python prep.py > results/prep.log 2>&1 && cp data/prep_stats.json results/prep_stats_rerun.json
if [ "${FEATS:-1}" = 1 ]; then python feats.py > results/feats_rerun.log 2>&1; fi
# Checkpoints trained in the first round (progress report) are staged in the bucket.
if [ -n "${BUCKET:-}" ]; then gcloud storage cp -r "$BUCKET/models/*" models/ || true; fi
echo setup done
