# Further architectural improvements for transformer transfer

Threat model: evade *existing, fixed* recognisers. The adversary scrapes cloaked photos, embeds
them with an off-the-shelf model it did not train, and fits a linear or nearest-centroid
classifier. Adaptive retraining against the cloak is out of scope (no cloak survives it, see
`DECISIONS.md`, known limits). The weak spot measured by `eval/harness.py` is transfer to a
held-out vision transformer (AdaFace ViT-B): 0.52 to 0.58 in `high` against 0.80 / 0.90 on
the two IResNet evaluators before this round, 0.66 to 0.70 after it. This document lists what the literature says about that
gap and ranks the options for this pipeline. Written 2026-09-13 from three surveys (transfer to
ViTs, open transformer recognisers, anti-facial-recognition cloaks); every claim carries its
source. What was actually implemented and measured is in `DECISIONS.md` (decision 11) and
`eval/RESULTS.md` (round 5 onward).

## What the surveys established

- **No anti-facial-recognition paper up to mid-2026 measures cloak transfer to a ViT
  recogniser.** GIFT, DiffAM, Adv-Diffusion, Chameleon, FaceCloak, AdvCloak, DPA, Adv-Pruning and
  DivTrackee all use CNN victims (IR152, IRSE50, FaceNet, MobileFace). The one data point is
  Radiya-Dixit and Tramer: a fine-tuned CLIP model cut LowKey's error rate to 16 percent against
  a 29 percent clean baseline ([2106.14851](https://arxiv.org/abs/2106.14851)). Our harness is
  ahead of the published record on this axis, so the numbers below come from ImageNet transfer
  work and have to be re-measured on faces.
- **CNN-crafted perturbations do not transfer to ViTs, ViT-crafted ones transfer to both**
  ([Mahmood et al., ICCV 2021](https://arxiv.org/abs/2104.02610);
  [Naseer et al., ICLR 2022](https://arxiv.org/abs/2106.04169)). The ViT surrogate should carry
  the targeted loss, which is why gradient scale between ensemble members matters (see
  `grad_norm` below).
- **ViTs are more sensitive to low-frequency and phase content** than CNNs
  ([2208.09602](https://arxiv.org/abs/2208.09602)); suppressing the high-frequency energy of the
  perturbation improves ViT transfer by about 11 points in TESSER
  ([2505.19613](https://arxiv.org/abs/2505.19613)).
- **The largest reproduced gains come from changing the surrogate's backward pass**, not the
  input: PNA, TGR, ATT, TESSER each add 10 to 30 points on unseen ViTs at no cost
  ([2109.04176](https://arxiv.org/abs/2109.04176), [2303.15754](https://arxiv.org/abs/2303.15754),
  [ATT, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html)).
- **Surrogate diversity beats optimiser tricks** in the face-specific literature: DPA lifts
  black-box impersonation from 16 to 68 percent on IR152 by adding fine-tuned and
  randomly-initialised checkpoints ([2411.15555](https://arxiv.org/abs/2411.15555)).
- **The adversary's detector is the largest under-modelled factor**: detector choice alone
  degrades transferred attacks by up to 78 percent, interpolation barely matters
  ([2510.17169](https://arxiv.org/abs/2510.17169)). KP-RPE makes the AdaFace ViT robust to
  misalignment by design ([2403.14852](https://arxiv.org/abs/2403.14852)), so landmark jitter
  alone will not close the ViT gap.
- **Impersonation on the surrogate does not imply dodging on the victim** (Adv-Pruning,
  [2401.08903](https://arxiv.org/abs/2401.08903)), which supports keeping the residual-identity
  penalty (decision 10) as a separate term.
- **A tracker that grows its gallery with recognised cloaked photos defeats every method
  tested** (DivTrackee, [2501.06533](https://arxiv.org/abs/2501.06533)); the remedy is diversity
  across one user's cloaked photos, which pulls against the per-user single target.

## Ranked options for this pipeline

Cost is engineering plus run time; gain is the reported number on ViT targets, ImageNet unless
marked. Options 1 to 5 are implemented behind `CloakParams` switches and were measured in sweep rounds
5 to 7 (`eval/RESULTS.md`; decision 11 has the reasoning); the rest are not implemented.

| # | option | mechanism | reported gain | cost | status |
|---|---|---|---|---|---|
| 1 | second transformer surrogate (LVFace-L) | architecture diversity in the ensemble | face-specific: surrogate diversity is the largest lever in DPA | about 1.6x run time with two extra | **adopted**: +0.10 to +0.14 on both transformer evaluators (Z1, Z7) |
| 2 | ViT backward refinements: PNA, TGR, SGM | detach attention maps; zero extreme-token gradients and scale the rest; decay residual-branch gradients | PNA +15, TGR +9 over PNA, SGM +5 to +9 on unseen ViTs | none | measured: PNA +0.04 on top of the extra surrogate (**adopted**); TGR harmful (-0.10 to -0.48, Z11/Z15/Z16); SGM neutral (Z12) |
| 3 | per-surrogate gradient normalisation | rescale each member's input gradient to the ensemble mean before Adam | CSE/CWA: ViT-B 59 to 90 percent from a CNN ensemble ([2303.09105](https://arxiv.org/abs/2303.09105)); needed because option 2 shrinks the ViT gradient 10 to 50 times | one backward per surrogate (about +30 percent) | measured: neutral (Z17 vs Z18), kept available, off by default |
| 4 | token-level PatchOut and pixel-block gradient dropout | drop 20 to 35 percent of tokens through the ViT's own `mask_token` path (in-distribution: LVFace trained with masking); or keep a random subset of 8 px gradient blocks | PatchOut +8 alone, +32 with PNA | none | measured: token masking small plus (**adopted**, 0.3 on augmented steps); pixel-grid PatchOut harmful (Z4, Z5) |
| 5 | low-frequency perturbation prior | optimise a blurred perturbation (`delta_sigma`) or add a DCT high-frequency penalty | TESSER +11 on ViT targets; ViT phase sensitivity ([2208.09602](https://arxiv.org/abs/2208.09602)); also free JPEG robustness | none | measured: blurred delta loses on every evaluator at this budget (Z14); a DCT penalty is untested |
| 6 | spectrum and geometric EOT views | SSA DCT-coefficient scaling, mild per-quadrant scale/rotate (SIA), random resized crop with re-alignment (S4ST) | SSA and SIA are the only CNN-era input methods that reach 50 percent on ViT-B in the 2026 benchmark ([2602.23117](https://arxiv.org/abs/2602.23117)) | one extra view type | not implemented |
| 7 | clean feature mixup (CFM / FTM) | mix intermediate features with those of other faces in the batch during the forward pass | best targeted objective in the benchmark: ViT-B 21.6 vs 2.4 percent for the logit loss ([2305.14846](https://arxiv.org/abs/2305.14846)); FTM +20 more with a ViT surrogate ([2411.15553](https://arxiv.org/abs/2411.15553)) | forward hooks on IResNet stages and ViT blocks | not implemented |
| 8 | stochastic ViT self-augmentation | per step: attention-score scaling, head dropping, MLP token mixing (ViT-EnsembleAttack, [2508.12384](https://arxiv.org/abs/2508.12384)) | ViT ensembles: 70 to 99 percent on unseen ViTs | one extra forward per copy | not implemented |
| 9 | FPR attention-map diversification | multiply pre-softmax attention by a random mask at fixed blocks ([2503.15404](https://arxiv.org/abs/2503.15404)) | 84 vs 77 for TGR from a ViT-B surrogate | none | not implemented |
| 10 | detector-diverse EOT | landmarks from a second detector plus box scale/shift jitter | LowKey 0.66 to 0.90 under a mismatched detector ([2510.17169](https://arxiv.org/abs/2510.17169)) | a second detector in the loop | not implemented |
| 11 | more distinct surrogates and evaluators | timm ViT-S/8 MS1MV3 ([gaunernst](https://huggingface.co/gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3)), CVLface KP-RPE WebFace12M (needs the 5 template keypoints at forward), EdgeFace (hybrid), PETALface Swin WebFace12M (120 px input) as evaluator, WebFace42M ViT-L ONNX as evaluator | DPA: 3 to 5 times black-box success from surrogate diversity | one loader per model | not implemented |
| 12 | feature-map hardening of surrogates (BPFA) | add loss-increasing perturbations to intermediate feature maps each step ([2210.16117](https://arxiv.org/abs/2210.16117)) | part of DPA's 16 to 68 | forward hooks, one extra backward | not implemented |
| 13 | explicit dodging budget (Adv-Pruning) | prune low-magnitude perturbation regions, refill with residual-identity gradient | dodging 2 to 6 percent to over 50 on black-box CNNs | moderate | not implemented |
| 14 | region-weighted budget | larger eps on eyes, nose and mouth (FaceCloak, [2605.19032](https://arxiv.org/abs/2605.19032)) | 85 vs 80 percent Top-1 protection | a landmark-based mask | not implemented |
| 15 | per-user target diversity | rotate targets across a user's photos or penalise similarity between cloaked outputs (DivTrackee) | closes the gallery-growing attack | changes the one-target-per-user design | not implemented |

## Not recommended

- Generative or diffusion cloaks (GIFT, DiffAM, Adv-Diffusion, DiffProtect): visible edits, no
  ViT evidence, abandons the bounded-perturbation premise.
- VDC (dominated by TGR and ATT in the 2026 benchmark), MIG (20 times the cost for mid-pack
  results), FIA / NAA / ILA feature attacks (12 to 20 percent on ViT-B).
- Self-ensemble without token refinement (needs training on a face set).
- Bayesian tuning of augmentation parameters against the evaluator (leaks the held-out model).
- Chroma-only budgets ([2003.00883](https://arxiv.org/abs/2003.00883)) and LPIPS as a penalty
  ([2307.15157](https://arxiv.org/abs/2307.15157)).

## Evaluator caveat

Round 5 adds LVFace-T as a fourth evaluator. It shares LVFace-B's training set, code and
recipe, so it is a weak held-out test; it is there to check that gains on AdaFace ViT-B are not
specific to one model. The most distinct transformer evaluators available with open weights are
PETALface Swin (ArcFace, WebFace12M, [HF](https://huggingface.co/kartiknarayan/PETALface)),
the insightface WebFace42M ViT-L ONNX
([HF mirror](https://huggingface.co/kunkunlin1221/face-recognition_vit-l-pfc0.3-cosface-web42m)),
TransFace-B on MS1MV2 (Google Drive only, needs TransFace's `vit.py` for its `senet` branch) and
CVLface KP-RPE WebFace12M. Adding one of them is the first thing to do before tuning further.

## Sources

- Wei et al., *Towards Transferable Adversarial Attacks on Vision Transformers* (PNA, PatchOut), AAAI 2022. https://arxiv.org/abs/2109.04176
- Zhang et al., *Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization*, CVPR 2023. https://arxiv.org/abs/2303.15754
- Ming et al., *Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning* (ATT), NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html
- TESSER, 2025. https://arxiv.org/abs/2505.19613
- FPR, CVPR 2025. https://arxiv.org/abs/2503.15404
- Wang et al., generalised skip gradient method for ViTs, 2024. https://arxiv.org/abs/2410.08950
- Naseer et al., *On Improving Adversarial Transferability of Vision Transformers*, ICLR 2022. https://arxiv.org/abs/2106.04169
- Mahmood et al., *On the Robustness of Vision Transformers to Adversarial Examples*, ICCV 2021. https://arxiv.org/abs/2104.02610
- Chen et al., *Rethinking Model Ensemble in Transfer-based Adversarial Attacks* (CWA), ICLR 2024. https://arxiv.org/abs/2303.09105
- ViT-EnsembleAttack, ICCV 2025. https://arxiv.org/abs/2508.12384
- Byun et al., *Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup* (CFM), CVPR 2023. https://arxiv.org/abs/2305.14846
- FTM, 2024. https://arxiv.org/abs/2411.15553
- Long et al., *Frequency Domain Model Augmentation* (SSA), ECCV 2022. https://arxiv.org/abs/2207.05382
- SIA, ICCV 2023. https://arxiv.org/abs/2309.14700 ; BSR, CVPR 2024. https://arxiv.org/abs/2308.10299 ; S4ST, 2024. https://arxiv.org/abs/2410.13891
- Transfer-attack benchmark, 2026. https://arxiv.org/abs/2602.23117
- Spectral sensitivity of ViTs, 2022. https://arxiv.org/abs/2208.09602
- DPA, CVPR 2025. https://arxiv.org/abs/2411.15555 ; BPFA, 2022. https://arxiv.org/abs/2210.16117
- Adv-Pruning, ACM MM 2024. https://arxiv.org/abs/2401.08903
- Preprocessing mismatch in face-recognition attacks, 2025. https://arxiv.org/abs/2510.17169
- KP-RPE, CVPR 2024. https://arxiv.org/abs/2403.14852
- FaceCloak, 2026. https://arxiv.org/abs/2605.19032 ; DivTrackee, 2025. https://arxiv.org/abs/2501.06533
- Radiya-Dixit et al., *Data Poisoning Won't Save You From Facial Recognition*, ICLR 2022. https://arxiv.org/abs/2106.14851
- LowKey, ICLR 2021. https://arxiv.org/abs/2101.07922 ; EOLT, 2025. https://arxiv.org/abs/2512.07228
- Open transformer recognisers: LVFace https://huggingface.co/bytedance-research/LVFace ; CVLface https://huggingface.co/minchul ; PETALface https://huggingface.co/kartiknarayan/PETALface ; TransFace https://github.com/DanJun6737/TransFace ; timm-face https://github.com/gau-nernst/timm-face ; EdgeFace https://huggingface.co/Idiap/EdgeFace-Base
