# Architectural decisions for Fawkes v2

Short record of *why* the v2 rewrite looks the way it does. Each entry states the decision, the
evidence or argument behind it, and what would change the decision. Dated 2026-09-12.

## 1. Cloak toward a user-supplied target identity, not away from self

**Decision.** The loss pulls each surrogate's embedding of the cloaked face toward the mean
embedding of a few photos of a chosen target person. The old `maximize=True` "move away from own
embedding" mode is removed.

**Evidence.** The [Fawkes paper](https://www.usenix.org/system/files/sec20-shan.pdf) (Shan et al., USENIX Security 2020) defines the method as target
mimicking: every photo of a user is pushed toward the *same* wrong identity so that a model trained
on them learns a consistent wrong class. The released code switched to an untargeted objective that
sends each photo in whichever direction its own gradient points, so a trained model sees a diffuse
class rather than a wrong one and still generalises to the clean face. Untargeted perturbations are
also known to transfer worse between models than targeted ones because they exploit local quirks
of the surrogate. [LowKey](https://arxiv.org/abs/2101.07922) (Cherepanova et al., ICLR 2021), which uses a coherent objective plus an
ensemble, kept Amazon Rekognition at 2.4 percent rank-50 recognition where Fawkes left 77.5 percent
rank-1.

**Would change it.** Evidence from our harness that an untargeted ensemble loss transfers as well.

## 2. Replace the 2020 surrogates with a modern, diverse ensemble

**Decision.** Three surrogates spanning architecture (IResNet, ViT), loss (ArcFace, AdaFace, LVFace)
and training set (Glint360K, WebFace12M): ArcFace IR-100, AdaFace IR-101, LVFace-B. Evaluators used
to measure protection are kept disjoint from the ensemble.

**Evidence.** Radiya-Dixit and Tramer (["Data Poisoning Won't Save You From Facial Recognition"](https://arxiv.org/abs/2106.14851),
ICLR 2022) show that cloaks are fixed at publication time and must defeat all future models; a
model trained after the cloaking tool is released, or trained robustly against it, breaks Fawkes
and LowKey. The old extractors are 2020 models trained on pre-2020 data. The transfer literature
consistently finds that ensembling raises black-box transfer (up to about 31 percent in the
[EOLT study](https://arxiv.org/abs/2512.07228), arXiv 2512.07228) and that architectural diversity matters more than ensemble size. No
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
["Unlearnable Faces" paper](https://arxiv.org/abs/2607.05996) (LPID, arXiv 2607.05996, 2026) models the attacker's crop-and-resize
pipeline differentiably and reports the lowest attacker accuracy of any method, which is direct
evidence that perturbing in the right frame is a first-order effect. Optimising in photo
coordinates also removes the inverse warp that would resample the cloak a second time on output.

## 4. Expectation over transformations: blur, resize, JPEG; not hue

**Decision.** Each optimisation step evaluates the loss on randomly blurred, resized and
JPEG-approximated copies, one view per step cycling through the clean alignment and two (five in
`high`) augmented views; on in every mode.

**Evidence.** LowKey's differentiable blur in the loop is one of the two changes credited with its
large gain over Fawkes. Several follow-ups report JPEG and Gaussian noise degrading Fawkes cloaks (e.g. the Florida Tech thesis ["An Assessment of Image-Cloaking Techniques"](https://repository.fit.edu/cgi/viewcontent.cgi?article=1793&context=etd)).
The [EOLT study](https://arxiv.org/abs/2512.07228) evaluated 30 transformations and found blur is the bottleneck that must be included
while hue augmentation overfits and *reduces* transfer. Social platforms re-encode uploads as JPEG.

**Measured after the fact (2026-09-12).** The views do more than make cloaks survive JPEG. The
fast mode with two surrogates and no views gives protection 0.22 / 0.42 / 0.00 on the three
held-out evaluators and drops to 0.10 / 0.18 / 0.00 after JPEG 75; the same mode with the views
gives 0.74 / 0.90 / 0.20 and keeps it under JPEG. The views act as a regulariser that stops the
perturbation from over-fitting the surrogates, the same effect input diversity has in the
transfer-attack literature. Five views per clean step instead of two adds about 0.10 on the
transformer evaluator in `high` at no extra cost per step.

## 5. PyTorch replaces TensorFlow

**Decision.** The optimiser, surrogates and evaluators run on PyTorch (CPU wheel). TensorFlow, Keras
and `mtcnn` are dropped.

**Evidence.** Every strong open recogniser ships as PyTorch weights ([insightface `arcface_torch`](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch),
[AdaFace](https://github.com/mk-minchul/AdaFace), [CVLface](https://github.com/mk-minchul/CVLface), [LVFace](https://github.com/bytedance/LVFace), [MagFace](https://github.com/IrvingMeng/MagFace), [TopoFR](https://github.com/DanJun6737/TopoFR)). The only live route into TensorFlow is [`onnx2tf`](https://github.com/PINTO0309/onnx2tf) on
ONNX exports, which is plausible for plain ResNets but unproven for ViTs and needs per-model
numeric and gradient validation; [`onnx-tf`](https://github.com/onnx/onnx-tensorflow) is dead (pins TF 2.8) and [`nobuco`](https://github.com/AlexanderLutsenko/nobuco) is Keras 2 only. The
existing 2020 extractors need a hand-written loader to work on Keras 3 at all. The torch CPU wheel
is about 190 MB versus about 600 MB for TensorFlow, so the install gets smaller.

**Cost.** Rewrite of `differentiator.py` and the face pipeline (about 900 lines total in the
package). The compiled-step speedups from the previous modernization are superseded, not lost:
PyTorch has no retrace cost and the control loop stays on-device.

## 6. InsightFace SCRFD for detection and alignment

**Decision.** `insightface` with `onnxruntime` provides detection and the five landmarks;
alignment uses the exact template points from [`insightface/utils/face_align.py`](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py).

**Evidence.** It is the reference implementation most adversaries would use, so our alignment
matches theirs. Its ONNX recognisers (`w600k_r50`, `glintr100`) double as forward-only held-out
evaluators. [`mtcnn` 1.0](https://pypi.org/project/mtcnn/) does return keypoints but depends on TensorFlow; [`facenet-pytorch`](https://pypi.org/project/facenet-pytorch/) pins
`torch<2.3`; [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md) returns a mouth centre rather than corners and cannot feed the template.

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
meaningless; only transfer to unseen models matters. [LFW](http://vis-www.cs.umass.edu/lfw/) is downloadable without credentials via
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html), so the number is reproducible by anyone. A linear probe on frozen embeddings is how
a low-effort adversary would actually build a recogniser from scraped photos.

## 10. Penalise the residual similarity to the own face, not only the distance to the target

**Decision.** The loss adds `self_weight` times the positive part of the cosine between the cloaked
face and its own clean embedding, averaged over the surrogates, on top of the target term. A
`laggard` option weights the surrogates by softmax((1 - cos) / laggard) so the one furthest from
the target leads the gradient.

**Evidence.** Measured with the harness on 2026-09-12 (LFW, 10 protected + 10 clean identities,
one target per identity, GPU): the target-only `mid` cloak moves every photo far from its identity
by the 1:1 verification measure (71 percent of cloaked photos below the 0.3 cosine threshold) yet
the linear probe still recognises 80 to 98 percent of the protected people (protection 0.20 / 0.14 /
0.02 on `buffalo_l` / `antelopev2` / `adaface_vit_b`). The probe works from the residual: the mean
cosine of a cloaked photo to its identity's clean centroid is 0.23 to 0.35, against 0.0 for
unrelated people, and a linear classifier only needs that direction. This is the mechanism
Radiya-Dixit and Tramer describe for why an adversary with a better model recovers the identity.
Adding the residual term with weight 1 and nothing else changed takes protection to 0.64 / 0.74 /
0.26 at the same DSSIM (0.012), and the residual cosine drops to 0.06 to 0.20. LowKey's objective is
the same idea in untargeted form (maximise the distance to the original embedding); combining
"toward the target" with "away from self" keeps the coherence across a user's photos that the
target gives while removing the signal the probe uses. Laggard weighting and a larger L-infinity
bound (16 at the same DSSIM budget) each add a few more points; the transformer evaluator remains
the hardest, as the ensemble holds one transformer. The harness assigns one target per protected
identity because that is how independent users behave; a single shared target is kept as an
option and gives higher numbers on the ResNet evaluators and lower on the transformer.

## Known limits, stated plainly

- No cloak can defend against a model trained adaptively after the cloaked photos are public
  (Radiya-Dixit and Tramer). The harness measures transfer to today's models, nothing more.
- Most of the strongest weights are research-only; Fawkes states the same use.
- CPU cost per face rises with modern surrogates; `low` mode exists for that reason.

## Sources

Papers
- Shan et al., *Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models*, USENIX Security 2020. https://www.usenix.org/system/files/sec20-shan.pdf
- Cherepanova et al., *LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition*, ICLR 2021. https://arxiv.org/abs/2101.07922
- Radiya-Dixit, Hong, Carlini, Tramer, *Data Poisoning Won't Save You From Facial Recognition*, ICLR 2022. https://arxiv.org/abs/2106.14851
- Oh, Park, Lee, *Unlearnable Faces: Privacy Protection Surviving Extraction Pipeline* (LPID), 2026. https://arxiv.org/abs/2607.05996
- *Robust Protective Perturbation* / EOLT transformation study, 2025. https://arxiv.org/abs/2512.07228
- Kim et al., *AdaFace: Quality Adaptive Margin for Face Recognition*, CVPR 2022. https://arxiv.org/abs/2204.00964
- You et al., *LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition*, ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/papers/You_LVFace_Progressive_Cluster_Optimization_for_Large_Vision_Models_in_Face_ICCV_2025_paper.pdf
- Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. https://arxiv.org/abs/1801.07698
- Guo et al., *Sample and Computation Redistribution for Efficient Face Detection* (SCRFD), ICLR 2022. https://arxiv.org/abs/2105.04714
- Sun et al., *Transferable Adversarial Facial Images for Privacy Protection*, ACM MM 2024. https://arxiv.org/abs/2408.01428
- Radiya-Dixit's result is also discussed in *An Assessment of Image-Cloaking Techniques* (Florida Tech thesis). https://repository.fit.edu/cgi/viewcontent.cgi?article=1793&context=etd

Code and weights
- InsightFace model zoo and packs: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md ; ONNX mirror https://huggingface.co/public-data/insightface
- InsightFace `arcface_torch` backbones: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- Alignment template: https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
- CVLface hub (AdaFace / ArcFace IR-101, ViT): https://github.com/mk-minchul/CVLface ; https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m
- LVFace: https://github.com/bytedance/LVFace ; https://huggingface.co/bytedance-research/LVFace
- MagFace (Apache-2.0): https://github.com/IrvingMeng/MagFace
- onnx2tf: https://github.com/PINTO0309/onnx2tf ; onnx-tensorflow (unmaintained): https://github.com/onnx/onnx-tensorflow ; nobuco (Keras 2 only): https://github.com/AlexanderLutsenko/nobuco
- Keras 3 legacy HDF5 incompatibility: https://github.com/keras-team/keras/issues/20083
- PyTorch CPU wheels: https://download.pytorch.org/whl/cpu
- LFW via scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
