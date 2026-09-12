#!/usr/bin/env bash
#SBATCH --job-name=fawkes-harness
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
# Usage: hpc submit hpc/harness.sh <mode> [extra harness args...]
# Runs the LFW harness for the v2 cloaker in <mode> against the three held-out evaluators,
# then re-evaluates the same cloaks after JPEG-75 re-encoding. Each mode uses its own workdir.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export UV_CACHE_DIR=/scratch/work/zalessi1/.uv-cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mode="${1:?mode}"; shift
work="eval/work/${mode}"
ev=(--evaluator buffalo_l --evaluator antelopev2 --evaluator adaface_vit_b)
echo "=== v2 ${mode} $(date)"
uv run python eval/harness.py --cloaker v2 --mode "${mode}" --batch-size 16 --workdir "${work}" "${ev[@]}" "$@"
echo "=== v2 ${mode} jpeg75 $(date)"
uv run python eval/harness.py --cloaker v2 --mode "${mode}" --batch-size 16 --workdir "${work}" --jpeg 75 --skip-cloak "${ev[@]}" "$@"
echo "=== done $(date)"
