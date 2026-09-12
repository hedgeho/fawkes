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

## Rounds 1 to 4: sweep (`hpc/sweep.sh`)

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

## Final modes (`hpc/harness.sh`)

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
