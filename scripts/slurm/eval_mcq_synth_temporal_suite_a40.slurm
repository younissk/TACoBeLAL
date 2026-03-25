#!/bin/bash
# Short SLURM sweep for the temporal trivial benchmark suite on 1x A40.
#
# Examples:
# sbatch scripts/slurm/eval_mcq_synth_temporal_suite_a40.slurm
# sbatch --export=ALL,MODELS="random llm-qwen qwen2-audio" scripts/slurm/eval_mcq_synth_temporal_suite_a40.slurm

#SBATCH --job-name=mcq-synth-temporal-suite
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

TRIVIAL_SYNTH_SCENES="${TRIVIAL_SYNTH_SCENES:-100}"
SYNTH_SEED="${SYNTH_SEED:-7}"
SYNTH_GENERATOR_VERSION="${SYNTH_GENERATOR_VERSION:-synth-v1}"
BENCH_SAMPLES="${BENCH_SAMPLES:-100}"
BENCH_USE_AUDIO="${BENCH_USE_AUDIO:-1}"
BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS:-0}"
BENCH_WANDB="${BENCH_WANDB:-1}"
BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT:-tacobelal}"
BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY:-}"
BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY:-50}"
BENCH_ARGS="${BENCH_ARGS:-}"

BENCHMARKS="${BENCHMARKS:-mcq-synth-pitch-order-trivial mcq-synth-loudness-order-trivial mcq-synth-duration-order-trivial mcq-synth-count-beeps-trivial mcq-synth-gap-trivial mcq-synth-pattern-pitch-trivial mcq-synth-dog-car-order-trivial}"
MODELS="${MODELS:-random llm-qwen qwen2-audio audioflamingo}"

for bench_task in ${BENCHMARKS}; do
  case "${bench_task}" in
    mcq-synth-pitch-order-trivial) build_target="build-mcq-synth-pitch-order-trivial" ;;
    mcq-synth-loudness-order-trivial) build_target="build-mcq-synth-loudness-order-trivial" ;;
    mcq-synth-duration-order-trivial) build_target="build-mcq-synth-duration-order-trivial" ;;
    mcq-synth-count-beeps-trivial) build_target="build-mcq-synth-count-beeps-trivial" ;;
    mcq-synth-gap-trivial) build_target="build-mcq-synth-gap-trivial" ;;
    mcq-synth-pattern-pitch-trivial) build_target="build-mcq-synth-pattern-pitch-trivial" ;;
    mcq-synth-dog-car-order-trivial) build_target="build-mcq-synth-dog-car-order-trivial" ;;
    *)
      echo "Unsupported benchmark task: ${bench_task}" >&2
      exit 1
      ;;
  esac
  make "${build_target}" \
    TRIVIAL_SYNTH_SCENES="${TRIVIAL_SYNTH_SCENES}" \
    SYNTH_SEED="${SYNTH_SEED}" \
    SYNTH_GENERATOR_VERSION="${SYNTH_GENERATOR_VERSION}"
done

for bench_task in ${BENCHMARKS}; do
  for bench_model in ${MODELS}; do
    run_name="${bench_task}_${bench_model}_${SLURM_JOB_ID}"
    make run-benchmark \
      BENCH_TASK="${bench_task}" \
      BENCH_MODEL="${bench_model}" \
      BENCH_SAMPLES="${BENCH_SAMPLES}" \
      BENCH_USE_AUDIO="${BENCH_USE_AUDIO}" \
      BENCH_PREPARE_DATA=0 \
      BENCH_INSTALL_DEPS="${BENCH_INSTALL_DEPS}" \
      BENCH_WANDB="${BENCH_WANDB}" \
      BENCH_WANDB_PROJECT="${BENCH_WANDB_PROJECT}" \
      BENCH_WANDB_ENTITY="${BENCH_WANDB_ENTITY}" \
      BENCH_WANDB_RUN_NAME="${run_name}" \
      BENCH_WANDB_LOG_EVERY="${BENCH_WANDB_LOG_EVERY}" \
      BENCH_ARGS="${BENCH_ARGS}"
  done
done
