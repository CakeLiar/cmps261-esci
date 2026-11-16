#!/usr/bin/env bash
# Second stage on the VM: reruns DeBERTa-small after the FP16 fix, waits for the baseline, then runs the error analysis.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
rm -f results/DONE
mv results/deberta_small.log results/deberta_small_crash1.log 2>/dev/null
MODEL=microsoft/deberta-v3-small TAG=deberta_small LR=2e-5 BS=48 python xenc_pretrained.py > results/deberta_small.log 2>&1
DEB=$?
while pgrep -f "[b]aseline\.py" >/dev/null; do sleep 20; done
python error_analysis.py > results/error_analysis.log 2>&1; E=$?
echo "deberta_small=$DEB error_analysis=$E baseline_json=$([ -f results/baseline.json ] && echo yes || echo no)" | tee results/DONE
