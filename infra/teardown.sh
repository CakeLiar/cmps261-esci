#!/usr/bin/env bash
# Delete the VM and the staging bucket.
set -euo pipefail
source "$(dirname "$0")/config.sh"
gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet
gcloud storage rm -r "$BUCKET" --project "$PROJECT"
