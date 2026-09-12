#!/usr/bin/env bash
#SBATCH --job-name=fawkes-sweep
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --output=out/%x_%j.out
# Parameter sweep of the v2 cloaker on the LFW harness, one GPU job. Every configuration shares
# eval/work/sweep so the clean photos' embeddings are computed once (sqlite cache).
#   hpc submit hpc/sweep.sh            # all configurations below
#   hpc submit hpc/sweep.sh A C        # a subset, by label
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export UV_CACHE_DIR=/scratch/work/zalessi1/.uv-cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
EV=(--evaluator buffalo_l --evaluator antelopev2 --evaluator adaface_vit_b)

# label | mode | extra harness arguments (cloak overrides and flags)
CONFIGS=(
  "A|mid|"
  "B|mid|--cloak-arg self_weight=1.0"
  "C|mid|--cloak-arg self_weight=1.0 --cloak-arg laggard=0.1"
  "D|mid|--cloak-arg self_weight=1.0 --cloak-arg steps=120 --cloak-arg stop_cos=0.99"
  "E|mid|--cloak-arg self_weight=1.0 --cloak-arg eps=16"
  "F|high|--cloak-arg self_weight=1.0"
  "G|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99"
  "H|high|--cloak-arg self_weight=2.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --cloak-arg laggard=0.1"
  "I|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --cloak-arg eps=20 --cloak-arg dssim_budget=0.02"
  "J|mid|--cloak-arg self_weight=1.0 --shared-target"
  "K|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --shared-target"
)
want=("$@")
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label mode extra <<<"$cfg"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $label "* ]]; then continue; fi
  echo "=== $label $mode $extra  $(date)"
  # shellcheck disable=SC2086
  uv run python eval/harness.py --cloaker v2 --mode "$mode" --batch-size 16 --tag "$label" \
    --workdir eval/work/sweep "${EV[@]}" $extra
done
echo "=== done $(date)"
