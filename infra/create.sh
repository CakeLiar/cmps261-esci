#!/usr/bin/env bash
# Create the staging bucket and the GPU VM (Ubuntu 22.04 image with the NVIDIA 580 driver preinstalled).
set -euo pipefail
source "$(dirname "$0")/config.sh"
gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "${ZONE%-*}" --uniform-bucket-level-access
gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
  --image-family ubuntu-accelerator-2204-amd64-with-nvidia-580 --image-project ubuntu-os-accelerator-images \
  --boot-disk-size 300GB --boot-disk-type pd-balanced --maintenance-policy TERMINATE --provisioning-model STANDARD \
  --scopes storage-rw
# Let the VM's default service account read and write the staging bucket.
PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding "$BUCKET" --project "$PROJECT" \
  --member "serviceAccount:${PN}-compute@developer.gserviceaccount.com" --role roles/storage.objectAdmin >/dev/null
