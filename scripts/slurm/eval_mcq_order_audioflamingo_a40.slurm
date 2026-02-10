#!/bin/bash
# Example SLURM script for MCQ-ORDER Audio Flamingo smoke/full runs on 1x A40.

#SBATCH --job-name=mcq-order-af3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Optional cluster cache dirs (recommended on shared storage):
# export HF_HOME=/path/to/shared/hf_cache
# export TRANSFORMERS_CACHE=/path/to/shared/hf_cache

# 1) Prepare code/data/repo once (skip if already done)
make setup-from-scratch

# 2) Run smoke test (100 examples by default)
make eval-mcq-order-audioflamingo-smoke

# 3) Run full benchmark (uncomment when ready)
# make eval-mcq-order-audioflamingo-full
