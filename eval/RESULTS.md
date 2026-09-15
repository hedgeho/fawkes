# Harness results, 2026-09-12

All runs: LFW, seed 0, 10 protected + 10 clean identities, 10 train + 5 test photos each, one
target per protected identity unless marked *shared*. Cloaks computed on an A100; the raw JSON
files (`eval/results/`, not tracked) hold the full metrics. Columns are the protection rate on the
three held-out evaluators, then the mean DSSIM inside the face box.

Reading the columns: `self` = `self_weight` (residual-identity penalty), `lag` = `laggard`,
`views` = `eot_samples`, `stop` = `stop_cos`. Mode defaults at the time of each run are in the
git history of `fawkes/protection.py`; the final modes are the last block.

## Round 0: target-only loss, shared target (before the residual penalty)

| mode | buffalo_l | antelopev2 | adaface_vit_b | DSSIM face |
|---|---|---|---|---|
| legacy Fawkes 1.0 mid (TensorFlow) | 0.00 | 0.00 | - | 0.015 |
| low (adaface_ir101, 30 steps, no views, eps 10, budget 0.008) | 0.00 | 0.00 | 0.00 | 0.007 |
| mid (3 surrogates, 60 steps, eps 12, budget 0.012) | 0.04 | 0.10 | 0.00 | 0.012 |
| high (120 steps, eps 16, budget 0.017) | 0.54 | 0.62 | 0.00 | 0.016 |

## Rounds 1 to 4: sweep (`eval/sweep.sh`)

| label | base | changes | buffalo_l | antelopev2 | adaface_vit_b | DSSIM face |
|---|---|---|---|---|---|---|
| A | mid | per-identity targets, no penalty | 0.20 | 0.14 | 0.02 | 0.012 |
| B | mid | self 1 | 0.64 | 0.74 | 0.26 | 0.012 |
| C | mid | self 1, lag 0.1 | 0.60 | 0.80 | 0.24 | 0.013 |
| D | mid | self 1, 120 steps, stop 0.99 | 0.62 | 0.72 | 0.30 | 0.013 |
| E | mid | self 1, eps 16 | 0.74 | 0.86 | 0.30 | 0.013 |
| F | high | self 1 | 0.74 | 0.86 | 0.44 | 0.017 |
| G | high | self 1, 200 steps, stop 0.99 | 0.72 | 0.86 | 0.42 | 0.016 |
| H | high | self 2, lag 0.1, 200 steps, stop 0.99 | 0.80 | 0.92 | 0.52 | 0.016 |
| I | high | self 1, 200 steps, eps 20, budget 0.02 | 0.84 | 0.96 | 0.50 | 0.019 |
| J | mid | self 1, *shared* target | 0.84 | 0.92 | 0.08 | 0.012 |
| K | high | self 1, 200 steps, *shared* target | 0.96 | 0.98 | 0.26 | 0.016 |
| L | low | self 2 (single model, no views) | 0.00 | 0.00 | 0.00 | 0.008 |
| L jpeg | | JPEG 75 | 0.00 | 0.00 | 0.00 | |
| M | low | self 2, + arcface_r100 | 0.04 | 0.16 | 0.00 | 0.008 |
| N | mid | self 2, lag 0.1, eps 16 | 0.70 | 0.82 | 0.28 | 0.012 |
| N jpeg | | JPEG 75 | 0.66 | 0.80 | 0.22 | |
| O | mid | self 3, lag 0.1, eps 16 | 0.32 | 0.34 | 0.22 | 0.012 |
| P | high | self 3, lag 0.1, 200 steps, stop 0.99 | 0.82 | 0.92 | 0.48 | 0.017 |
| P jpeg | | JPEG 75 | 0.80 | 0.90 | 0.50 | |
| Q | high | self 3, lag 0.1, 200 steps, eps 20, budget 0.02 | 0.88 | 0.98 | 0.60 | 0.019 |
| R | mid | self 2, lag 0.1, eps 16, 120 steps, stop 0.99 | 0.78 | 0.86 | 0.34 | 0.013 |
| T | low (eps 16, budget 0.012, 40 steps, no views) | self 1, adaface_ir101 only | 0.06 | 0.10 | 0.00 | 0.011 |
| U | same | self 1, three surrogates | 0.22 | 0.38 | 0.00 | 0.011 |
| V | same | self 2, 60 steps, 2 views | 0.72 | 0.88 | 0.22 | 0.012 |
| W | mid (final) | 5 views | 0.74 | 0.82 | 0.34 | 0.013 |
| X | high (final but 2 views) | 5 views | 0.80 | 0.92 | 0.56 | 0.017 |
| Y | mid (final) | 5 views, 90 steps | 0.68 | 0.80 | 0.40 | 0.013 |

What the sweep says:

- The residual penalty is the decisive change (A to B). Above weight 2 it needs more steps (O).
- A larger L-infinity bound at the same DSSIM budget helps (E, N); the DSSIM budget, not eps, is
  what limits visibility.
- The robustness views drive transfer (T/U versus V), not only JPEG survival; five views help the
  transformer evaluator a little (X).
- Laggard weighting and shared targets move the ResNet and transformer numbers in opposite
  directions; neither is decisive.
- The transformer evaluator never exceeds 0.6. The ensemble contains one transformer; a second
  one is the untested next lever.

## Rounds 5 to 7 (2026-09-13): transformer transfer

A fourth evaluator, LVFace-T (ViT-T, Glint360K), is added. It shares LVFace-B's family and
training set, so it is a weaker held-out test than AdaFace ViT-B; it is there to check that gains
on the transformer are not specific to one model. Cloaks computed on A100 or H100 nodes, so the
seconds per photo are not comparable between rows. New `CloakParams` switches (all off by default):
`pna` detaches the attention maps in the backward pass, `patchout` keeps a random fraction of 8 px
gradient blocks, `tgr` zeroes the extreme-token gradients and scales the rest, `sgm` decays the
residual-branch gradients, `token_mask` drops tokens through the ViT's own mask path on augmented
steps, `delta_sigma` blurs the perturbation, `grad_norm` rescales each surrogate's gradient to
the ensemble mean. `+L`, `+S` = LVFace-L, LVFace-S added to the three surrogates.

| label | base | changes | buffalo_l | antelopev2 | adaface_vit_b | lvface_t | DSSIM face |
|---|---|---|---|---|---|---|---|
| Z0 | high | (final high of 2026-09-12) | 0.82 | 0.92 | 0.52 | 0.78 | 0.017 |
| Z1 | high | +L | 0.84 | 0.96 | 0.62 | 0.88 | 0.017 |
| Z2 | high | pna | 0.84 | 0.94 | 0.56 | 0.76 | 0.017 |
| Z3 | high | +L, pna | 0.88 | 0.98 | 0.66 | 0.86 | 0.017 |
| Z4 | high | +L, pna, patchout 0.5 | 0.76 | 0.90 | 0.50 | 0.82 | 0.018 |
| Z5 | high | patchout 0.5 | 0.76 | 0.82 | 0.40 | 0.66 | 0.018 |
| Z6 | high | L instead of B | 0.86 | 0.86 | 0.48 | 0.76 | 0.017 |
| Z7 | high | +L +S, pna | 0.92 | 0.96 | 0.66 | 0.92 | 0.017 |
| Z8 | mid | +L, pna | 0.78 | 0.88 | 0.38 | 0.82 | 0.012 |
| Z9 | high | +L, grad_norm | 0.82 | 0.90 | 0.58 | 0.84 | 0.017 |
| Z10 | high | +L, grad_norm, pna | 0.86 | 0.94 | 0.60 | 0.86 | 0.017 |
| Z11 | high | +L, grad_norm, pna, tgr 0.25 | 0.80 | 0.92 | 0.32 | 0.76 | 0.017 |
| Z12 | high | +L, grad_norm, pna, sgm 0.6 | 0.84 | 0.92 | 0.62 | 0.90 | 0.016 |
| Z13 | high | +L, grad_norm, pna, token_mask 0.3 | 0.88 | 0.96 | 0.66 | 0.88 | 0.017 |
| Z14 | high | +L, grad_norm, delta_sigma 1.0 | 0.78 | 0.82 | 0.54 | 0.78 | 0.018 |
| Z15 | high | +L, grad_norm, pna, tgr, sgm, token_mask | 0.76 | 0.90 | 0.18 | 0.58 | 0.017 |
| Z16 | high | grad_norm, pna, tgr 0.25 | 0.84 | 0.98 | 0.46 | 0.72 | 0.017 |
| Z17 | high | +L +S, pna, token_mask 0.3 | 0.90 | 0.96 | 0.66 | 0.94 | 0.017 |
| Z17 jpeg | | JPEG 75 | 0.86 | 0.92 | 0.68 | 0.94 | |
| Z18 | high | +L +S, pna, token_mask 0.3, grad_norm | 0.90 | 0.98 | 0.64 | 1.00 | 0.017 |
| Z19 | mid | +L +S, pna, token_mask 0.3 | 0.82 | 0.92 | 0.50 | 0.90 | 0.013 |
| Z19 jpeg | | JPEG 75 | 0.78 | 0.82 | 0.46 | 0.90 | |
| Z20 | mid | +L, pna, token_mask 0.3 | 0.80 | 0.86 | 0.44 | 0.84 | 0.013 |

What these rounds say:

- A second transformer surrogate is the lever (Z0 to Z1: +0.10 on both transformer evaluators,
  ResNets unchanged or up). Replacing LVFace-B by LVFace-L instead of adding it (Z6) gives
  nothing, so it is the number of transformers in the ensemble, not their size. A third one
  (LVFace-S, Z7) adds a little more on every evaluator.
- PNA adds a few points on top of the extra surrogate (Z1 to Z3, Z10 to Z13 with token masking)
  and nothing alone (Z2). Token masking through the ViT's own mask path is a small plus (Z10 to
  Z13); the pixel-grid PatchOut is clearly harmful (Z4, Z5).
- TGR is harmful in every combination (Z11, Z15, Z16): with 24 blocks the zeroed extreme-token
  gradients remove most of the signal the targeted cosine loss needs. SGM is neutral (Z12). The
  blurred perturbation loses on every evaluator at this DSSIM budget (Z14).
- Per-surrogate gradient normalisation neither helps nor hurts with the switches that survive
  (Z17 versus Z18) and costs 20 to 30 percent more, so it stays off. It was needed to test the
  switches fairly: PNA, TGR and SGM shrink the transformer's input gradient 10 to 50 times.
- The gains survive JPEG 75 (Z17, Z19).

## Round 8 (2026-09-15): visible tint

The final cloaks leave smooth reddish / yellowish patches on the skin. Measured on real photos,
the perturbation's mean colour is zero but its low-frequency chroma carries as much energy as its
low-frequency luma: SSIM is nearly blind to a smooth shift, the L-inf bound is per channel, and
the JPEG view rewards chroma that is smooth. Two new `CloakParams` fields: `chroma_eps` clamps
the Cb/Cr of the perturbation after every step (0 = grey cloak), `chroma_weight` penalises its
mean Cb/Cr magnitude. Two new harness columns, computed in the face box on the difference between
the cloaked and the clean photo: *chroma LF* and *luma LF* are the rms of the Cb/Cr and of the Y
component after a Gaussian low-pass with sigma 2 percent of the box (the smooth patches a viewer
sees). T0 repeats the final mid mode on the same GPU as the baseline.

| label | base | changes | buffalo_l | antelopev2 | adaface_vit_b | lvface_t | DSSIM face | chroma LF | luma LF |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | mid | (final mid) | 0.80 | 0.90 | 0.48 | 0.92 | 0.012 | 2.00 | 1.42 |
| T1 | mid | chroma_eps 4 | 0.72 | 0.86 | 0.28 | 0.90 | 0.013 | 1.26 | 1.61 |
| T2 | mid | chroma_eps 2 | 0.74 | 0.86 | 0.18 | 0.90 | 0.013 | 0.73 | 1.85 |
| T2 jpeg | | JPEG 75 | 0.64 | 0.84 | 0.18 | 0.84 | | | |
| T3 | mid | chroma_eps 0 (grey) | 0.62 | 0.78 | 0.10 | 0.84 | 0.013 | 0.09 | 1.98 |
| T3 jpeg | | JPEG 75 | 0.62 | 0.72 | 0.10 | 0.78 | | | |
| T4 | mid | chroma_weight 30 | 0.76 | 0.88 | 0.34 | 0.90 | 0.013 | 0.91 | 1.44 |
| T5 | mid | chroma_weight 100 | 0.70 | 0.84 | 0.14 | 0.86 | 0.013 | 0.26 | 1.52 |
| T6 | mid | chroma_eps 2, eps 12 | 0.60 | 0.76 | 0.14 | 0.82 | 0.013 | 0.71 | 1.58 |
| T7 | high | chroma_eps 2 | 0.84 | 0.92 | 0.40 | 0.96 | 0.017 | 0.79 | 2.48 |
| T8 | high | chroma_eps 0 (grey) | 0.74 | 0.84 | 0.18 | 0.98 | 0.018 | 0.10 | 2.66 |

What round 8 says:

- Colour is doing real work for the transformer evaluator. Removing it entirely (T3) keeps most
  of the ResNet protection but drops AdaFace ViT-B from 0.48 to 0.10; the bound at 2 (T2) is
  hardly better on that evaluator. The luma-only cloak is not an option at this budget.
- The soft penalty is the better lever: at weight 30 (T4) the visible chroma halves for a loss of
  0.04 on the ResNets and 0.14 on the transformer, better on every column than the hard bound at
  4 (T1), which removes less chroma. The penalty leaves colour where the surrogates need it and
  removes it elsewhere; the bound cuts everywhere.
- The lost transfer is not recovered by a smaller luma bound (T6) and not fully by the high mode
  either: high with chroma bound 2 (T7) is back to the ResNet numbers of the unbounded mid and
  0.40 on the transformer, at the high DSSIM budget.

## Round 9 (2026-09-15): the chroma penalty weight

| label | base | changes | buffalo_l | antelopev2 | adaface_vit_b | lvface_t | DSSIM face | chroma LF | luma LF |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | mid | (final mid) | 0.80 | 0.90 | 0.48 | 0.92 | 0.012 | 2.00 | 1.42 |
| T9 | mid | chroma_weight 10 | 0.82 | 0.90 | 0.46 | 0.92 | 0.012 | 1.45 | 1.40 |
| T10 | mid | chroma_weight 20 | 0.80 | 0.90 | 0.42 | 0.92 | 0.013 | 1.13 | 1.42 |
| T10 jpeg | | JPEG 75 | 0.70 | 0.86 | 0.34 | 0.92 | | | |
| T4 | mid | chroma_weight 30 | 0.76 | 0.88 | 0.34 | 0.90 | 0.013 | 0.91 | 1.44 |
| T11 | mid | chroma_weight 50 | 0.72 | 0.84 | 0.18 | 0.92 | 0.013 | 0.62 | 1.49 |
| T12 | mid | chroma_weight 30, 120 steps | 0.78 | 0.88 | 0.40 | 0.88 | 0.013 | 0.93 | 1.68 |
| T15 | mid | chroma_weight 30, eps 20 | 0.84 | 0.94 | 0.42 | 0.92 | 0.012 | 0.93 | 1.53 |
| T13 | high | chroma_weight 30 | 0.82 | 0.96 | 0.52 | 0.94 | 0.016 | 0.92 | 1.89 |
| T13 jpeg | | JPEG 75 | 0.82 | 0.88 | 0.48 | 0.94 | | | |
| T14 | high | chroma_weight 60 | 0.80 | 0.90 | 0.44 | 0.94 | 0.017 | 0.50 | 1.98 |

What round 9 says:

- The penalty trades smoothly: each step of 10 in the weight removes about a quarter of the
  remaining visible chroma. Up to weight 20 the protection is within run-to-run noise of the
  baseline (T9, T10); at 30 the transformer evaluator loses 0.14 (T4); at 50 it collapses (T11).
- More steps do not buy the lost transfer back (T12), a larger luma bound does: weight 30 with
  eps 20 (T15) has half the chroma of the baseline and the same or better protection on every
  evaluator at the same DSSIM. The chroma the penalty removes is replaced by luma detail, which
  the DSSIM budget, not the L-inf bound, was limiting.
- In high mode weight 30 halves the chroma for a loss of 0.18 on the transformer against the
  2026-09-13 high (T13); weight 60 is too much (T14). Round 10 checks the eps 20 lever there.

## Round 10 (2026-09-15): the penalty with a larger luma bound

T16 repeats the final high mode on the same day as the baseline; T17 repeats T15 for a noise
estimate (agreement within 0.04 on every evaluator).

| label | base | changes | buffalo_l | antelopev2 | adaface_vit_b | lvface_t | DSSIM face | chroma LF | luma LF |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | mid | (final mid of 2026-09-13) | 0.80 | 0.90 | 0.48 | 0.92 | 0.012 | 2.00 | 1.42 |
| T19 | mid | chroma_weight 20, eps 20 | 0.88 | 0.92 | 0.44 | 0.94 | 0.012 | 1.17 | 1.51 |
| T17 | mid | chroma_weight 30, eps 20 | 0.88 | 0.92 | 0.40 | 0.90 | 0.012 | 0.93 | 1.53 |
| T17 jpeg | | JPEG 75 | 0.76 | 0.88 | 0.30 | 0.90 | | | |
| T22 | mid | chroma_weight 30, eps 24 | 0.88 | 0.96 | 0.48 | 0.94 | 0.012 | 0.94 | 1.59 |
| T18 | mid | chroma_weight 40, eps 20 | 0.80 | 0.92 | 0.34 | 0.92 | 0.013 | 0.75 | 1.56 |
| T16 | high | (final high of 2026-09-13) | 0.90 | 0.94 | 0.70 | 0.96 | 0.017 | 2.41 | 1.80 |
| T20 | high | chroma_weight 30, eps 20 | 0.92 | 1.00 | 0.62 | 0.98 | 0.017 | 0.93 | 2.11 |
| T20 jpeg | | JPEG 75 | 0.88 | 0.96 | 0.62 | 0.98 | | | |
| T21 | high | chroma_weight 60, eps 20 | 0.88 | 0.92 | 0.54 | 0.96 | 0.017 | 0.48 | 2.20 |

What round 10 says:

- With the luma bound raised the penalty is free. `mid` with weight 30 and eps 24 (T22) has half
  the visible chroma of the baseline and equal or better protection on all four evaluators at
  the same DSSIM; eps 20 (T17) is 0.04 to 0.08 lower on the transformer, eps 24 recovers it.
  The DSSIM budget, not the pixel bound, limits the luma, and the penalty moves the budget from
  colour to luminance detail (luma LF 1.42 to 1.59).
- The same holds in `high`: weight 30 with eps 20 (T20) is within noise of the same-day
  baseline on three evaluators and 0.08 lower on AdaFace ViT-B, at 40 percent of the chroma,
  and holds under JPEG 75. Weight 60 (T21) removes 80 percent of the chroma but costs 0.16 on
  the transformer.
- Under JPEG 75 the mid cloak with the penalty loses more on the transformer (0.30 for T17
  against 0.44 for the old mid) than clean; the final-mode run below measures this for eps 24.
- Weight 40 (T18) is already past the knee. The new defaults are weight 30 with eps 24 in `mid`
  and eps 20 in `high` (decision 12).

## Final modes of 2026-09-13 (`eval/run_mode.sh`, five surrogates in mid and high)

Same protocol; `low` is unchanged from 2026-09-12 (no transformer surrogate) and not re-run. The
high run repeats configuration Z17 with a different GPU allocation; the two agree within 0.04 on
every evaluator, the mid run repeats Z19 within 0.02.

| mode | buffalo_l | antelopev2 | adaface_vit_b | lvface_t | DSSIM face | s/photo A100 |
|---|---|---|---|---|---|---|
| mid | 0.80 | 0.92 | 0.48 | 0.90 | 0.013 | 1.7 |
| mid, JPEG 75 | 0.74 | 0.84 | 0.44 | 0.92 | | |
| high | 0.86 | 0.92 | 0.70 | 0.96 | 0.016 | 4.4 |
| high, JPEG 75 | 0.86 | 0.90 | 0.68 | 0.96 | | |

## Clean gallery (2026-09-13, `--clean-gallery`, same cloaks as the final modes above)

The other threat model: the adversary enrolled the person from clean photos before any cloak
existed and meets a cloaked photo later. Same cloaks, probe trained on the 5 clean test photos per
identity, asked to recognise the 100 cloaked train photos. Evaluated on a CPU from the cluster's
cloaks and embedding caches. The 1:1 columns (every cloaked photo below the 0.3 threshold on every
evaluator) are unchanged by construction.

| mode | buffalo_l | antelopev2 | adaface_vit_b | lvface_t |
|---|---|---|---|---|
| mid | 0.80 | 0.91 | 0.55 | 0.97 |
| mid, JPEG 75 | 0.74 | 0.90 | 0.52 | 0.96 |
| high | 0.89 | 0.93 | 0.71 | 1.00 |
| high, JPEG 75 | 0.88 | 0.89 | 0.71 | 0.99 |

Within noise of the cloaked-gallery numbers on every evaluator (each cell is 100 photos here, 50
there): the cloaks move the embedding far enough that it does not matter which side of the
classifier they land on. Mid on the transformer evaluator is the one cell where the clean gallery
is clearly easier to beat (0.55 against 0.48), which is expected, since a probe fitted on cloaked
photos has partly learned the cloak direction.

## Final modes of 2026-09-12 (`eval/run_mode.sh`, three surrogates)

| mode | buffalo_l | antelopev2 | adaface_vit_b | DSSIM face | s/photo A100 |
|---|---|---|---|---|---|
| low | 0.74 | 0.90 | 0.20 | 0.012 | 1.1 |
| low, JPEG 75 | 0.70 | 0.84 | 0.22 | | |
| mid | 0.74 | 0.84 | 0.32 | 0.012 | 1.0 |
| mid, JPEG 75 | 0.72 | 0.78 | 0.30 | | |
| high | 0.80 | 0.90 | 0.58 | 0.017 | 2.3 |
| high, JPEG 75 | 0.78 | 0.90 | 0.56 | | |

The final high run repeats configuration X with a different GPU allocation; the two agree within
0.02, which is the run-to-run noise of the cloaker on a GPU.
