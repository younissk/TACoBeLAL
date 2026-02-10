#!/bin/bash
# Example SLURM script for MCQ-ORDER Voxtral smoke/full runs on 1x A40.

#SBATCH --job-name=mcq-order-voxtral
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8.0-gcc-12.2.0-bplw5nu || true
  module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6 || true
fi

# Optional shared cache dirs (recommended on clusters):
# export HF_HOME=/path/to/shared/hf_cache
# export TRANSFORMERS_CACHE=/path/to/shared/hf_cache
# export WANDB_API_KEY=your_wandb_key

# Set FULL_BENCHMARK=1 to run without --limit.
FULL_BENCHMARK="${FULL_BENCHMARK:-0}"

make install-tracking
make download-dataset
make extract-audio
make build-mcq-dataset

if [ "${FULL_BENCHMARK}" = "1" ]; then
  make eval-mcq-order-voxtral-full \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-voxtral_mcq_order_full_a40}"
else
  make eval-mcq-order-voxtral-smoke \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-voxtral_mcq_order_smoke_a40}"
fi
