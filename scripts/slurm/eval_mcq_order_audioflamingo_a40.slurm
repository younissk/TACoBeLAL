#!/bin/bash
# Example SLURM script for MCQ-ORDER Audio Flamingo smoke/full runs on 1x A40.

#SBATCH --job-name=mcq-order-af3
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1          # half-node: 1x A40 GPU + associated memory/CPUs
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Ensure CUDA toolkit is available for DeepSpeed ops.
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8.0-gcc-12.2.0-bplw5nu || true
  module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6 || true
fi

# Optional cluster cache dirs (recommended on shared storage):
# export HF_HOME=/path/to/shared/hf_cache
# export TRANSFORMERS_CACHE=/path/to/shared/hf_cache
# export WANDB_API_KEY=your_wandb_key

# 1) Prepare code/data/repo once (skip if already done)
make setup-from-scratch

# Set FULL_BENCHMARK=1 to run without --limit.
FULL_BENCHMARK="${FULL_BENCHMARK:-0}"

if [ "${FULL_BENCHMARK}" = "1" ]; then
  make eval-mcq-order-audioflamingo-full \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-af3_mcq_order_full_a40}"
else
  make eval-mcq-order-audioflamingo-smoke \
    AF_SMOKE_LIMIT="${AF_SMOKE_LIMIT:-100}" \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-af3_mcq_order_smoke_a40}"
fi
