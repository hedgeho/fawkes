#!/usr/bin/env bash
# Parameter sweep of the v2 cloaker on the LFW harness. Every configuration shares eval/work/sweep
# so the clean photos' embeddings are computed once (sqlite cache).
#   eval/sweep.sh            # all configurations below
#   eval/sweep.sh A C        # a subset, by label
# SWEEP_WORKDIR (default eval/work/sweep) selects the workdir; two runs must not share one, as the
# cloaked files are written next to the inputs. Weights must already be in fawkes/model/
# (python -m fawkes.models download); the device follows FAWKES_DEVICE / CUDA availability.
set -euo pipefail
cd "$(dirname "$0")/.."
EV=(--evaluator buffalo_l --evaluator antelopev2 --evaluator adaface_vit_b --evaluator lvface_t)
WORK="${SWEEP_WORKDIR:-eval/work/sweep}"

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
  # Round 3: what a fast (no EOT) mode can do at the mid budget; the low mode itself is measured by eval/run_mode.sh
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
  # Round 6 (2026-09-13): the transformer switches on the four-surrogate ensemble with per-surrogate
  # gradient normalisation (the switches shrink the ViT gradient 10-50x, so without it they drop the
  # ViT out of the update). Z16: the same switches on the original three surrogates.
  "Z9|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1|"
  "Z10|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg pna=1|"
  "Z11|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg pna=1 --cloak-arg tgr=0.25|"
  "Z12|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg pna=1 --cloak-arg sgm=0.6|"
  "Z13|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg pna=1 --cloak-arg token_mask=0.3|"
  "Z14|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg delta_sigma=1.0|"
  "Z15|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg grad_norm=1 --cloak-arg pna=1 --cloak-arg tgr=0.25 --cloak-arg sgm=0.6 --cloak-arg token_mask=0.3|"
  "Z16|high|--cloak-arg grad_norm=1 --cloak-arg pna=1 --cloak-arg tgr=0.25|"
  # Round 7: the round-5 winner (five surrogates, PNA) plus token masking, with and without gradient
  # normalisation; the same recipe as a mid-mode candidate.
  "Z17|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l,lvface_s --cloak-arg pna=1 --cloak-arg token_mask=0.3|jpeg"
  "Z18|high|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l,lvface_s --cloak-arg pna=1 --cloak-arg token_mask=0.3 --cloak-arg grad_norm=1|"
  "Z19|mid|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l,lvface_s --cloak-arg pna=1 --cloak-arg token_mask=0.3|jpeg"
  "Z20|mid|--cloak-arg models=arcface_r100,adaface_ir101,lvface_b,lvface_l --cloak-arg pna=1 --cloak-arg token_mask=0.3|"
  # Round 8 (2026-09-15): visible tint. The cloaks look like smooth reddish / yellowish patches on the
  # skin: SSIM is nearly blind to low-frequency change and the per-channel L-inf bound allows saturated
  # colour. `chroma_eps` clamps the Cb/Cr of the perturbation after every step (0 = luma only),
  # `chroma_weight` penalises its mean chroma magnitude instead. T0 repeats the final mid mode so the
  # new chroma columns have a same-GPU baseline.
  "T0|mid||"
  "T1|mid|--cloak-arg chroma_eps=4|"
  "T2|mid|--cloak-arg chroma_eps=2|jpeg"
  "T3|mid|--cloak-arg chroma_eps=0|jpeg"
  "T4|mid|--cloak-arg chroma_weight=30|"
  "T5|mid|--cloak-arg chroma_weight=100|"
  "T6|mid|--cloak-arg chroma_eps=2 --cloak-arg eps=12|"
  "T7|high|--cloak-arg chroma_eps=2|"
  "T8|high|--cloak-arg chroma_eps=0|"
  # Round 9: the hard bound loses most of the transformer transfer (T1-T3), the soft penalty at weight
  # 30 (T4) halves the visible chroma for a few points. Map the penalty weight, give it more steps,
  # a larger luma bound, and try it in high mode.
  "T9|mid|--cloak-arg chroma_weight=10|"
  "T10|mid|--cloak-arg chroma_weight=20|jpeg"
  "T11|mid|--cloak-arg chroma_weight=50|"
  "T12|mid|--cloak-arg chroma_weight=30 --cloak-arg steps=120|"
  "T13|high|--cloak-arg chroma_weight=30|jpeg"
  "T14|high|--cloak-arg chroma_weight=60|"
  "T15|mid|--cloak-arg chroma_weight=30 --cloak-arg eps=20|"
  # Round 10: weight 20 is nearly free, weight 30 with the luma bound raised to 20 (T15) recovers all
  # of the protection at half the chroma. Same-day high baseline, the eps=20 lever with other weights,
  # in high mode, and under JPEG; T17 repeats T15 for a noise estimate.
  "T16|high||"
  "T17|mid|--cloak-arg chroma_weight=30 --cloak-arg eps=20|jpeg"
  "T18|mid|--cloak-arg chroma_weight=40 --cloak-arg eps=20|"
  "T19|mid|--cloak-arg chroma_weight=20 --cloak-arg eps=20|"
  "T20|high|--cloak-arg chroma_weight=30 --cloak-arg eps=20|jpeg"
  "T21|high|--cloak-arg chroma_weight=60 --cloak-arg eps=20|"
  "T22|mid|--cloak-arg chroma_weight=30 --cloak-arg eps=24|"
  # Round 11 (2026-09-25): postable cloaks. On real photos the decision-12 cloaks read as ageing: dark
  # wrinkle-scale contours around eyes, nose and mouth. Each visibility lever alone on mid: a matched
  # target from the pool (same sex, similar age and skin tone; near or far in feature space), an LPIPS
  # penalty, a texture-masked bound (smooth skin gets eps * eps_floor), a penalty on darkening
  # wrinkle-scale lines, and a sub-pixel warp that carries part of the attack. S0 is the same-day baseline.
  "S0|mid||"
  "S1|mid|--target-strategy near|"
  "S2|mid|--target-strategy far|"
  "S3|mid|--cloak-arg lpips_weight=5|"
  "S4|mid|--cloak-arg lpips_weight=20|"
  "S5|mid|--cloak-arg eps_floor=0.5|"
  "S6|mid|--cloak-arg eps_floor=0.25|"
  "S7|mid|--cloak-arg shade_weight=30|"
  "S8|mid|--cloak-arg shade_weight=100|"
  "S9|mid|--cloak-arg flow_eps=1.0|"
  "S10|mid|--cloak-arg flow_eps=1.0 --cloak-arg eps=16|"
  # Round 12: on the author's photos only the age penalty (genderage in the loss) removes the folds and
  # eye-bag lines; LPIPS, the texture floor and the line penalty leave them, the warp adds its own. The
  # age penalty alone, with the matched near target, pushed younger, stronger, and in high mode.
  "S11|mid|--cloak-arg age_weight=1|"
  "S12|mid|--cloak-arg age_weight=3|"
  "S13|mid|--cloak-arg age_weight=3 --target-strategy near|jpeg"
  "S14|mid|--cloak-arg age_weight=3 --cloak-arg age_margin=-3 --target-strategy near|"
  "S15|mid|--cloak-arg age_weight=10 --target-strategy near|"
  "S16|high|--cloak-arg age_weight=3 --target-strategy near|jpeg"
  "S17|high|--target-strategy near|"
  # Round 13: the far matched target (round 11: 0.98 / 0.98 / 0.74 / 0.98 in mid) still reads as ageing on
  # real photos; with the age penalty it does not. In mid the penalty costs 0.2-0.3 at 60 steps but is
  # nearly free in high (200 steps): far + age penalty with more steps, more budget, a symmetric hinge.
  "R0|mid|--target-strategy far --cloak-arg age_weight=1|"
  "R1|mid|--target-strategy far --cloak-arg age_weight=3|"
  "R2|mid|--target-strategy far --cloak-arg age_weight=3 --cloak-arg age_symmetric=1|"
  "R3|mid|--target-strategy far --cloak-arg age_weight=3 --cloak-arg steps=120 --cloak-arg stop_cos=0.99|"
  "R4|mid|--target-strategy far --cloak-arg age_weight=3 --cloak-arg steps=120 --cloak-arg stop_cos=0.99 --cloak-arg age_symmetric=1|"
  "R5|high|--target-strategy far --cloak-arg age_weight=3|jpeg"
  "R6|high|--target-strategy far --cloak-arg age_weight=10 --cloak-arg age_symmetric=1|"
)
want=("$@")
# SWEEP_STAGE=all (default): cloak and evaluate each configuration in one job, sharing $WORK.
# SWEEP_STAGE=cloak: only cloak, each configuration in $WORK/<label> (the GPU part);
# SWEEP_STAGE=eval: only evaluate those, sharing the embedding cache $WORK/embeddings.sqlite (CPU only).
STAGE="${SWEEP_STAGE:-all}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label mode extra jpeg <<<"$cfg"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $label "* ]]; then continue; fi
  echo "=== $label $mode $extra ($STAGE)  $(date)"
  wd="$WORK"; stage_args=()
  case "$STAGE" in
    cloak) wd="$WORK/$label"; stage_args=(--cloak-only) ;;
    eval) wd="$WORK/$label"; stage_args=(--skip-cloak --cache "$WORK/embeddings.sqlite") ;;
  esac
  # shellcheck disable=SC2086
  uv run python eval/harness.py --cloaker v2 --mode "$mode" --batch-size 16 --tag "$label" \
    --workdir "$wd" "${EV[@]}" "${stage_args[@]}" $extra
  if [ "$jpeg" = "jpeg" ] && [ "$STAGE" != "cloak" ]; then
    echo "=== $label jpeg75  $(date)"
    cache_args=(); [ "$STAGE" = "eval" ] && cache_args=(--cache "$WORK/embeddings.sqlite")
    # shellcheck disable=SC2086
    uv run python eval/harness.py --cloaker v2 --mode "$mode" --tag "$label" --jpeg 75 --skip-cloak \
      --workdir "$wd" "${EV[@]}" "${cache_args[@]}" $extra
  fi
done
echo "=== done $(date)"
