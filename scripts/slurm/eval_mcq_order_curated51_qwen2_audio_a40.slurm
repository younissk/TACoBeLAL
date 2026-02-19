#!/bin/bash
# SLURM script for MCQ-ORDER curated 51-sample run with Qwen2-Audio.
#
# Example:
# sbatch --export=ALL,CURATED_DATASET=data/<your_curated_51>.jsonl scripts/slurm/eval_mcq_order_curated51_qwen2_audio_a40.slurm

#SBATCH --job-name=mcq-order-cur51-qwen2-audio
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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

CURATED_DATASET="${CURATED_DATASET:-data/mcq_event_timeline_curated_51.jsonl}"
BENCH_SAMPLES="${BENCH_SAMPLES:-51}"
BENCH_PREPARE_DATA="${BENCH_PREPARE_DATA:-0}"
BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS:-0}"
BENCH_WANDB="${BENCH_WANDB:-1}"
BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT:-tacobelal}"
BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY:-}"
BENCH_WANDB_RUN_NAME="${BENCH_WANDB_RUN_NAME:-mcq_order_cur51_qwen2_audio_${SLURM_JOB_ID}}"
BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY:-50}"
BENCH_ARGS_USER="${BENCH_ARGS:-}"

if [ ! -f "${CURATED_DATASET}" ]; then
  echo "ERROR: curated dataset not found: ${CURATED_DATASET}" >&2
  exit 1
fi

BENCH_ARGS_COMBINED="--dataset ${CURATED_DATASET}"
if [ -n "${BENCH_ARGS_USER}" ]; then
  BENCH_ARGS_COMBINED="${BENCH_ARGS_COMBINED} ${BENCH_ARGS_USER}"
fi

make run-benchmark \
  BENCH_TASK=mcq-order \
  BENCH_MODEL=qwen2-audio \
  BENCH_SAMPLES="${BENCH_SAMPLES}" \
  BENCH_USE_AUDIO=1 \
  BENCH_PREPARE_DATA="${BENCH_PREPARE_DATA}" \
  BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS}" \
  BENCH_WANDB="${BENCH_WANDB}" \
  BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT}" \
  BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY}" \
  BENCH_WANDB_RUN_NAME="${BENCH_WANDB_RUN_NAME}" \
  BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY}" \
  BENCH_ARGS="${BENCH_ARGS_COMBINED}"
