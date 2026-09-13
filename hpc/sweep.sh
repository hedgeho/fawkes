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
EV=(--evaluator buffalo_l --evaluator antelopev2 --evaluator adaface_vit_b --evaluator lvface_t)

# label | mode | extra harness arguments (cloak overrides and flags) | "jpeg" to also evaluate a JPEG-75 copy
# Round 1 (2026-09-12, results in eval/results/*_[A-K].json): the residual penalty (self_weight)
# is what makes the probe fail; laggard weighting and a larger eps help a little more.
CONFIGS=(
  "A|mid||"
  "B|mid|--cloak-arg self_weight=1.0|"
  "C|mid|--cloak-arg self_weight=1.0 --cloak-arg laggard=0.1|"
  "D|mid|--cloak-arg self_weight=1.0 --cloak-arg steps=120 --cloak-arg stop_cos=0.99|"
  "E|mid|--cloak-arg self_weight=1.0 --cloak-arg eps=16|"
  "F|high|--cloak-arg self_weight=1.0|"
  "G|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99|"
  "H|high|--cloak-arg self_weight=2.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --cloak-arg laggard=0.1|"
  "I|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --cloak-arg eps=20 --cloak-arg dssim_budget=0.02|"
  "J|mid|--cloak-arg self_weight=1.0 --shared-target|"
  "K|high|--cloak-arg self_weight=1.0 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --shared-target|"
  # Round 2: stronger residual penalty, mode candidates with JPEG re-encoding
  "L|low|--cloak-arg self_weight=2.0|jpeg"
  "M|low|--cloak-arg self_weight=2.0 --cloak-arg models=adaface_ir101,arcface_r100|"
  "N|mid|--cloak-arg self_weight=2.0 --cloak-arg laggard=0.1 --cloak-arg eps=16|jpeg"
  "O|mid|--cloak-arg self_weight=3.0 --cloak-arg laggard=0.1 --cloak-arg eps=16|"
  "P|high|--cloak-arg self_weight=3.0 --cloak-arg laggard=0.1 --cloak-arg steps=200 --cloak-arg stop_cos=0.99|jpeg"
  "Q|high|--cloak-arg self_weight=3.0 --cloak-arg laggard=0.1 --cloak-arg steps=200 --cloak-arg stop_cos=0.99 --cloak-arg eps=20 --cloak-arg dssim_budget=0.02|"
  "R|mid|--cloak-arg self_weight=2.0 --cloak-arg laggard=0.1 --cloak-arg eps=16 --cloak-arg steps=120 --cloak-arg stop_cos=0.99|"
  # Round 3: what a fast (no EOT) mode can do at the mid budget; the low mode itself is measured by hpc/harness.sh
  "T|low|--cloak-arg self_weight=1.0 --cloak-arg models=adaface_ir101|"
  "U|low|--cloak-arg self_weight=1.0 --cloak-arg models=adaface_ir101,arcface_r100,lvface_b|"
  "V|low|--cloak-arg self_weight=2.0 --cloak-arg steps=60 --cloak-arg eot_samples=2|"
  # Round 4: the robustness views turned out to drive transfer (V vs the no-EOT low); more of them?
  "W|mid|--cloak-arg eot_samples=5|"
  "X|high|--cloak-arg eot_samples=5|"
  "Y|mid|--cloak-arg eot_samples=5 --cloak-arg steps=90|"
  # Round 5 (2026-09-13): transformer transfer. Second transformer surrogate (LVFace-L), PNA (detach the
  # attention maps in the backward pass), PatchOut-style gradient block dropout; lvface_t added as a
  # fourth evaluator. Z0 repeats the final high mode as the baseline on the new evaluator.
  "Z0|high||"
  "Z1|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l|"
  "Z2|high|--cloak-arg pna=1|"
  "Z3|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg pna=1|"
  "Z4|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg pna=1 --cloak-arg patchout=0.5|"
  "Z5|high|--cloak-arg patchout=0.5|"
  "Z6|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_l|"
  "Z7|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l,lvface_s --cloak-arg pna=1|"
  "Z8|mid|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg pna=1|"
)
want=("$@")
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label mode extra jpeg <<<"$cfg"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $label "* ]]; then continue; fi
  echo "=== $label $mode $extra  $(date)"
  # shellcheck disable=SC2086
  uv run python eval/harness.py --cloaker v2 --mode "$mode" --batch-size 16 --tag "$label" \
    --workdir eval/work/sweep "${EV[@]}" $extra
  if [ "$jpeg" = "jpeg" ]; then
    echo "=== $label jpeg75  $(date)"
    # shellcheck disable=SC2086
    uv run python eval/harness.py --cloaker v2 --mode "$mode" --tag "$label" --jpeg 75 --skip-cloak \
      --workdir eval/work/sweep "${EV[@]}" $extra
  fi
done
echo "=== done $(date)"
