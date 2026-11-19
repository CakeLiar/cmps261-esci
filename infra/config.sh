# Shared settings for the infra scripts. PROJECT must be set in the environment (e.g. PROJECT=my-gcp-project).
: "${PROJECT:?set PROJECT to your GCP project id}"
ZONE=${ZONE:-us-west1-a}
VM=${VM:-esci-cmps261}
MACHINE=${MACHINE:-g2-standard-32}      # 32 vCPU, 128 GB RAM, 1 x NVIDIA L4 (4-GPU and 2-GPU L4 types were out of stock)
BUCKET=${BUCKET:-gs://${PROJECT}-esci-cmps261}
REMOTE_DIR=esci
