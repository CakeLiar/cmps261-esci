#!/usr/bin/env bash
# Copy the tracked project files to the VM.
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$(dirname "$0")/.."
tar czf /tmp/esci_code.tgz $(git ls-files) $(git ls-files --others --exclude-standard)
gcloud compute scp /tmp/esci_code.tgz "$VM":~/ --project "$PROJECT" --zone "$ZONE"
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR && tar xzf esci_code.tgz -C $REMOTE_DIR"
