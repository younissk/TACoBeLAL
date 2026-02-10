#!/bin/bash
# Example SLURM script for MCQ-ORDER text-only open-source LLM runs on 1x A40.

#SBATCH --job-name=mcq-order-text-llm
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8.0-gcc-12.2.0-bplw5nu || true
fi

# Optional shared cache dirs (recommended on clusters):
# export HF_HOME=/path/to/shared/hf_cache
# export TRANSFORMERS_CACHE=/path/to/shared/hf_cache
# export WANDB_API_KEY=your_wandb_key
# export HF_TOKEN=your_hf_token  # required for gated models (e.g., Meta Llama)

# qwen (default) or llama
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
# Set FULL_BENCHMARK=1 to run without --limit
FULL_BENCHMARK="${FULL_BENCHMARK:-0}"

# Install runtime deps and ensure dataset exists.
make install-llm
make install-tracking
make download-dataset
make build-mcq-dataset

if [ "${FULL_BENCHMARK}" = "1" ]; then
  export LOCAL_LIMIT=
fi

if [ "${MODEL_FAMILY}" = "llama" ]; then
  make eval-mcq-order-llama \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-llama_mcq_order_a40}"
else
  make eval-mcq-order-qwen \
    WANDB_PROJECT="${WANDB_PROJECT:-tacobelal}" \
    WANDB_ENTITY="${WANDB_ENTITY:-}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_mcq_order_a40}"
fi
