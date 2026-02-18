#!/bin/bash
# Dedicated SLURM launcher for MCQ safety-check runs on 1x A40.
#
# Examples (one model per job):
# sbatch --export=ALL,BENCH_MODEL=qwen2-audio scripts/slurm/eval_mcq_safety_a40.slurm
# sbatch --export=ALL,BENCH_MODEL=qwen2-5-omni scripts/slurm/eval_mcq_safety_a40.slurm
# sbatch --export=ALL,BENCH_MODEL=voxtral scripts/slurm/eval_mcq_safety_a40.slurm
# sbatch --export=ALL,BENCH_MODEL=audioflamingo scripts/slurm/eval_mcq_safety_a40.slurm

#SBATCH --job-name=mcq-safety-bench
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

BENCH_TASK="${BENCH_TASK:-mcq-safety}"
BENCH_MODEL="${BENCH_MODEL:-qwen2-audio}"
BENCH_SAMPLES="${BENCH_SAMPLES:-0}"
BENCH_USE_AUDIO="${BENCH_USE_AUDIO:-1}"
BENCH_PREPARE_DATA="${BENCH_PREPARE_DATA:-1}"
BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS:-0}"
BENCH_WANDB="${BENCH_WANDB:-1}"
BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT:-tacobelal}"
BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY:-}"
BENCH_WANDB_RUN_NAME="${BENCH_WANDB_RUN_NAME:-${BENCH_TASK}_${BENCH_MODEL}_${SLURM_JOB_ID}}"
BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY:-50}"
BENCH_ARGS="${BENCH_ARGS:-}"

make run-benchmark \
  BENCH_TASK="${BENCH_TASK}" \
  BENCH_MODEL="${BENCH_MODEL}" \
  BENCH_SAMPLES="${BENCH_SAMPLES}" \
  BENCH_USE_AUDIO="${BENCH_USE_AUDIO}" \
  BENCH_PREPARE_DATA="${BENCH_PREPARE_DATA}" \
  BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS}" \
  BENCH_WANDB="${BENCH_WANDB}" \
  BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT}" \
  BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY}" \
  BENCH_WANDB_RUN_NAME="${BENCH_WANDB_RUN_NAME}" \
  BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY}" \
  BENCH_ARGS="${BENCH_ARGS}"
