#!/bin/bash
# SLURM script for MCQ-ORDER Audio Flamingo no-audio run (1000 samples) on 1x A40.

#SBATCH --job-name=mcq-order-audioflamingo-no-audio
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
  module use /opt/sw/zen/spack-0.19.0/share/spack/modules/linux-almalinux8-zen3
  module load cuda/11.8.0-gcc-12.2.0-bplw5nu || true
  module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6 || true
fi

make install-tracking
make download-dataset
make extract-audio
make build-mcq-dataset
make eval-mcq-order-audioflamingo-no-audio-smoke AF_SMOKE_LIMIT=1000
