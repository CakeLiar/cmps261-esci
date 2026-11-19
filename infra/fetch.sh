#!/usr/bin/env bash
# Copy results, figures, report, notebook and checkpoints from the VM to this machine through the bucket.
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$(dirname "$0")/.."
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command \
  "cd $REMOTE_DIR && gcloud storage rsync -r results $BUCKET/results && gcloud storage rsync -r models $BUCKET/models && gcloud storage rsync -r report $BUCKET/report && gcloud storage rsync -r figures $BUCKET/figures && gcloud storage cp CMPS261_ESCI.ipynb $BUCKET/ || true"
gcloud storage rsync -r "$BUCKET/results" results
gcloud storage rsync -r "$BUCKET/models" models
gcloud storage rsync -r "$BUCKET/report" report || true
gcloud storage rsync -r "$BUCKET/figures" figures || true
gcloud storage cp "$BUCKET/CMPS261_ESCI.ipynb" . || true
