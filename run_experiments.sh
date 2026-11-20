#!/usr/bin/env bash
# Second round of experiments (final report). Runs ON a VM from the project root.
#   QUEUES  which job queues to run on this VM; queue i uses GPU i (default: "a b" on one VM with 2 GPUs;
#           the script stops if there are fewer GPUs than queues). QUEUES="" runs only the FINAL step.
#           We ran QUEUES=a on VM esci-cmps261 and QUEUES=b on VM esci-cmps261-b (one L4 each).
#   LR=1    also run the logistic-regression rerun on this VM's CPUs.
#   FINAL=1 after the queues: test predictions, failure analysis, hybrid and the one-time test evaluation
#           (needs every queue's results/ in place first).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results/logs
QUEUES=${QUEUES:-a b}

run() {  # run <gpu> <tag> [ENV=VALUE ...]
  local gpu=$1 tag=$2; shift 2
  env CUDA_VISIBLE_DEVICES=$gpu TAG=$tag "$@" python xenc_pretrained.py > results/logs/$tag.log 2>&1
  echo "$tag exit=$?" >> results/logs/queue_status.txt
}
MINI="MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 LR=3e-5 BS=64 SAVE=0"
DSMALL="MODEL=microsoft/deberta-v3-small LR=2e-5 BS=48"

queue_a() {  # model size (DeBERTa-v3-base, saved for test) and max length 256
  run $1 deberta_base MODEL=microsoft/deberta-v3-base LR=2e-5 BS=48
  run $1 minilm_len256 $MINI MAXLEN=256
}
queue_b() {  # second seeds, class weighting, field ablations, max length 64, learning curve
  run $1 deberta_small_seed1 $DSMALL SEED=1 SAVE=0
  run $1 minilm_seed1 $MINI SEED=1
  run $1 minilm_balanced $MINI CW=balanced
  run $1 minilm_f_title $MINI FIELDS=title
  run $1 minilm_f_title_bullets $MINI FIELDS=title,bullets
  run $1 minilm_f_all_desc $MINI FIELDS=title,brandcolor,bullets,desc
  run $1 minilm_len64 $MINI MAXLEN=64
  run $1 minilm_frac10 $MINI FRAC=0.1
  run $1 minilm_frac25 $MINI FRAC=0.25
  run $1 minilm_frac50 $MINI FRAC=0.5
}

nq=$(echo $QUEUES | wc -w)
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$nq" -gt "$ngpu" ]; then
  echo "QUEUES='$QUEUES' needs $nq GPUs but this machine has $ngpu. Run one queue per 1-GPU VM (QUEUES=a, QUEUES=b)." >&2
  exit 1
fi
gpu=0
for q in $QUEUES; do ( queue_$q $gpu ) & gpu=$((gpu + 1)); done
if [ "${LR:-0}" = 1 ]; then
  ( python lr_converge.py > results/logs/lr_converge.log 2>&1; echo "lr_converge exit=$?" >> results/logs/queue_status.txt ) &
fi
wait
echo "QUEUES_DONE $QUEUES" >> results/logs/queue_status.txt

if [ "${FINAL:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES=0 TAGS=minilm,deberta_small,deberta_base python predict_test.py > results/logs/predict_test.log 2>&1
  echo "predict_test exit=$?" >> results/logs/queue_status.txt
  python error_analysis.py > results/logs/error_analysis.log 2>&1; echo "error_analysis exit=$?" >> results/logs/queue_status.txt
  python final_test.py > results/logs/final_test.log 2>&1; echo "final_test exit=$?" >> results/logs/queue_status.txt
  echo ALL_DONE >> results/logs/queue_status.txt
fi
