#!/usr/bin/env bash
# Runs on the GCP VM from the project root. Each stage logs to results/.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
sha256sum data/feat_*.parquet > results/feat_sha_before.txt

python feats.py > results/feats.log 2>&1 || { echo "feats failed"; exit 1; }
sha256sum data/feat_*.parquet > results/feat_sha_after.txt

# Baseline uses the CPU while the cross-encoder uses the GPU.
python baseline.py > results/baseline_full.log 2>&1 &
BASE=$!

MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 TAG=minilm LR=3e-5 python xenc_pretrained.py > results/minilm.log 2>&1
MINI=$?
MODEL=microsoft/deberta-v3-small TAG=deberta_small LR=2e-5 BS=48 python xenc_pretrained.py > results/deberta_small.log 2>&1
DEB=$?

wait $BASE; B=$?
python error_analysis.py > results/error_analysis.log 2>&1; E=$?
echo "feats ok, baseline=$B minilm=$MINI deberta_small=$DEB error_analysis=$E" | tee results/DONE
