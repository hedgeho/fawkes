# Architectural decisions for Fawkes v2

Short record of *why* the v2 rewrite looks the way it does. Each entry states the decision, the
evidence or argument behind it, and what would change the decision. Dated 2026-09-12.

## 1. Cloak toward a user-supplied target identity, not away from self

**Decision.** The loss pulls each surrogate's embedding of the cloaked face toward the mean
embedding of a few photos of a chosen target person. The old `maximize=True` "move away from own
embedding" mode is removed.

**Evidence.** The Fawkes paper (Shan et al., USENIX Security 2020) defines the method as target
mimicking: every photo of a user is pushed toward the *same* wrong identity so that a model trained
on them learns a consistent wrong class. The released code switched to an untargeted objective that
sends each photo in whichever direction its own gradient points, so a trained model sees a diffuse
class rather than a wrong one and still generalises to the clean face. Untargeted perturbations are
also known to transfer worse between models than targeted ones because they exploit local quirks
of the surrogate. LowKey (Cherepanova et al., ICLR 2021), which uses a coherent objective plus an
ensemble, kept Amazon Rekognition at 2.4 percent rank-50 recognition where Fawkes left 77.5 percent
rank-1.

**Would change it.** Evidence from our harness that an untargeted ensemble loss transfers as well.

## 2. Replace the 2020 surrogates with a modern, diverse ensemble

**Decision.** Three surrogates spanning architecture (IResNet, ViT), loss (ArcFace, AdaFace, LVFace)
and training set (Glint360K, WebFace12M): ArcFace IR-100, AdaFace IR-101, LVFace-B. Evaluators used
to measure protection are kept disjoint from the ensemble.

**Evidence.** Radiya-Dixit and Tramer ("Data Poisoning Won't Save You From Facial Recognition",
ICLR 2022) show that cloaks are fixed at publication time and must defeat all future models; a
model trained after the cloaking tool is released, or trained robustly against it, breaks Fawkes
and LowKey. The old extractors are 2020 models trained on pre-2020 data. The transfer literature
consistently finds that ensembling raises black-box transfer (up to about 31 percent in the
EOLT study, arXiv 2512.07228) and that architectural diversity matters more than ensemble size. No
published work benchmarks Fawkes against AdaFace-IR101 or LVFace class models, so the harness in
`eval/` produces that number. The old extractors remain available for the "before" column only.

**Would change it.** Harness results showing one member contributes nothing; a stronger open model
with a permissive licence.

## 3. Landmark alignment, and optimising through the adversary's alignment

**Decision.** Faces are detected with SCRFD and aligned with the standard ArcFace 5-point similarity
transform. The perturbation lives in photo coordinates on a native-resolution face crop and is
warped into the 112 px template *differentiably* inside the loss, with random landmark jitter.

**Evidence.** The current `align_face.py` discards the five keypoints MTCNN returns and crops the
raw box, then pads it into a mean-coloured square. Every ArcFace-family model, including the
adversary's, expects the aligned template, so the old cloak was optimised in a frame the adversary
never sees and gets resampled by an arbitrary similarity transform before it reaches them. The
"Unlearnable Faces" paper (LPID, arXiv 2607.05996, 2026) models the attacker's crop-and-resize
pipeline differentiably and reports the lowest attacker accuracy of any method, which is direct
evidence that perturbing in the right frame is a first-order effect. Optimising in photo
coordinates also removes the inverse warp that would resample the cloak a second time on output.

## 4. Expectation over transformations: blur, resize, JPEG; not hue

**Decision.** Each optimisation step evaluates the loss on randomly blurred, resized and
JPEG-approximated copies (two samples per step; off in `low` mode).

**Evidence.** LowKey's differentiable blur in the loop is one of the two changes credited with its
large gain over Fawkes. Several follow-ups report JPEG and Gaussian noise degrading Fawkes cloaks.
The EOLT study evaluated 30 transformations and found blur is the bottleneck that must be included
while hue augmentation overfits and *reduces* transfer. Social platforms re-encode uploads as JPEG.

## 5. PyTorch replaces TensorFlow

**Decision.** The optimiser, surrogates and evaluators run on PyTorch (CPU wheel). TensorFlow, Keras
and `mtcnn` are dropped.

**Evidence.** Every strong open recogniser ships as PyTorch weights (insightface `arcface_torch`,
AdaFace, CVLface, LVFace, MagFace, TopoFR). The only live route into TensorFlow is `onnx2tf` on
ONNX exports, which is plausible for plain ResNets but unproven for ViTs and needs per-model
numeric and gradient validation; `onnx-tf` is dead (pins TF 2.8) and `nobuco` is Keras 2 only. The
existing 2020 extractors need a hand-written loader to work on Keras 3 at all. The torch CPU wheel
is about 190 MB versus about 600 MB for TensorFlow, so the install gets smaller.

**Cost.** Rewrite of `differentiator.py` and the face pipeline (about 900 lines total in the
package). The compiled-step speedups from the previous modernization are superseded, not lost:
PyTorch has no retrace cost and the control loop stays on-device.

## 6. InsightFace SCRFD for detection and alignment

**Decision.** `insightface` with `onnxruntime` provides detection and the five landmarks;
alignment uses the exact template points from `insightface/utils/face_align.py`.

**Evidence.** It is the reference implementation most adversaries would use, so our alignment
matches theirs. Its ONNX recognisers (`w600k_r50`, `glintr100`) double as forward-only held-out
evaluators. `mtcnn` 1.0 does return keypoints but depends on TensorFlow; `facenet-pytorch` pins
`torch<2.3`; MediaPipe returns a mouth centre rather than corners and cannot feed the template.

## 7. Projected optimiser with early stopping instead of the penalty scheduler

**Decision.** Adam on the perturbation with an L-infinity projection each step, a fixed DSSIM
penalty weight, and per-face early stopping when all surrogates are within a cosine threshold of
the target or no improvement is seen for ten steps. The `const_diff` doubling scheduler and the
`save_last_on_failed` heuristic are removed.

**Evidence.** Profiling the current code on this machine: 40 compiled steps take 1.45 s but the
same 40 steps through `compute_batch` take 1.89 s, so the per-step host round trip is 23 percent
on CPU and would dominate on a GPU. The scheduler and the 0.3 / 0.015 fallback thresholds are
unexplained constants that exist because an unbounded "away from self" objective has no natural
stopping point; a target loss converges and can stop on its own. Iteration count is the main
latency driver, so a stopping rule is the largest remaining latency win.

## 8. Fix `resize` before measuring anything

**Decision.** `utils.resize` now calls `array_to_img(..., scale=False)`.

**Evidence.** Measured: a crop with values in [100, 140] came back in [0, 255]. With the default
`scale=True`, the protected and original crops in `merge_faces` were stretched by different affine
maps before subtraction, so the pasted cloak had the wrong gain and a uniform colour offset. The
bug is inherited from upstream. Fixing it first makes the "before" harness numbers a fair baseline.

## 9. Public-dataset evaluation harness with held-out evaluators

**Decision.** `eval/harness.py` cloaks the training photos of some LFW identities, trains a linear
probe on embeddings from recognisers *not* in the surrogate ensemble, and reports the protection
rate on clean test photos, with and without JPEG re-encoding.

**Evidence.** Radiya-Dixit and Tramer's central point is that evaluating against the surrogate is
meaningless; only transfer to unseen models matters. LFW is downloadable without credentials via
scikit-learn, so the number is reproducible by anyone. A linear probe on frozen embeddings is how
a low-effort adversary would actually build a recogniser from scraped photos.

## Known limits, stated plainly

- No cloak can defend against a model trained adaptively after the cloaked photos are public
  (Radiya-Dixit and Tramer). The harness measures transfer to today's models, nothing more.
- Most of the strongest weights are research-only; Fawkes states the same use.
- CPU cost per face rises with modern surrogates; `low` mode exists for that reason.
