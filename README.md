Fawkes
------

Fawkes is a privacy protection system developed by researchers at [SANDLab](https://sandlab.cs.uchicago.edu/),
University of Chicago. It adds small, hard-to-see changes ("cloaks") to your photos so that facial
recognition models trained on them learn the wrong face. For the original project see the
[project webpage](https://sandlab.cs.uchicago.edu/fawkes/) and the paper
"[Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models](https://www.shawnshan.com/files/publication/fawkes.pdf)"
(USENIX Security 2020).

This fork is a rewrite of the cloaking pipeline (version 2):

- cloaks push your face toward a **target person you choose**, the method described in the paper,
  instead of just away from itself;
- the perturbation is optimised against an **ensemble of modern face recognisers** (ArcFace IR-100,
  AdaFace IR-101, LVFace ViT-B) through the same **landmark alignment** those recognisers use;
- the optimisation is robust to blur, resizing and JPEG re-encoding;
- protection is **measured** on a public dataset against recognisers that were *not* used to build the
  cloaks (see [Evaluation](#evaluation));
- runs on PyTorch (CPU is fine), TensorFlow is gone.

The reasons behind these choices, with sources, are in [docs/DECISIONS.md](docs/DECISIONS.md).

Copyright
---------
This code is intended only for personal privacy protection or academic research. The pretrained
recogniser weights it downloads are licensed for non-commercial research use only (see
[Model licences](#model-licences)).

Usage
-----

```
fawkes -d ./imgs --target-dir ./target --mode mid
```

or `python3 -m fawkes -d ./imgs -t ./target -m mid`.

Every image in `./imgs` gets a `<name>_cloaked.png` next to it. Options:

* `-d`, `--directory`: the directory with images to protect.
* `-t`, `--target-dir`: a directory with a few photos of the person your cloaked photos should
  resemble to a recogniser. See [Choosing a target](#choosing-a-target).
* `-m`, `--mode`: `low`, `mid` (default) or `high`, the tradeoff between protection strength,
  visible change and run time:

  | mode | surrogates | steps | max pixel change | DSSIM budget | robustness views | s / face, 8-core CPU |
  |---|---|---|---|---|---|---|
  | low | AdaFace IR-101, ArcFace IR-100 | 60 | 16 | 0.012 | blur, resize, JPEG | 35 |
  | mid | + LVFace-B, LVFace-L, LVFace-S (transformers) | 60 | 16 | 0.012 | blur, resize, JPEG | 74 |
  | high | same five | 200 | 16 | 0.017 | blur, resize, JPEG | about 250 (estimated) |

  Every mode also penalises what is left of your own identity in the cloaked face
  (`--self-weight`, see [docs/DECISIONS.md](docs/DECISIONS.md#10-penalise-the-residual-similarity-to-the-own-face-not-only-the-distance-to-the-target)).
  On a CUDA GPU all modes take a second or two per face.

* `--models`: comma-separated surrogate keys overriding the mode (`python -m fawkes.models list`).
* `--steps`, `--eps`, `--th`, `--no-eot`, `--self-weight`: override the mode's steps, pixel bound,
  DSSIM budget, robustness views and residual penalty.
* `--batch-size`: faces optimised together (default 8; larger batches are faster per face).
* `--threads`: CPU threads for PyTorch.
* `--no-align`: the inputs are already 112x112 aligned faces; skip detection.
* `--format`: `png` (default) or `jpg`.
* `--debug`: print per-face statistics.

### Choosing a target

A recogniser trained on your cloaked photos should learn the target's face, not yours. All of your
photos are pushed toward the same target, so use the same `--target-dir` every time; the target
embedding is cached in `<target-dir>/fawkes_target.npz` after the first run.

- Use 3 to 10 photos of one person, each with exactly one clearly visible face. Photos with zero or
  several faces are skipped.
- Pick someone who looks unlike you: a different sex, age or ethnicity moves your embedding
  further. Fawkes warns when your faces already resemble the target under one of its models.
- Public figures work fine; their real photos are labelled as them, not as you.

### Tips

- Run time on an 8-core CPU without a GPU is about 35 s per face in `low` and a minute per face in
  `mid`; batching several faces is faster per face. A CUDA GPU is used automatically when PyTorch
  finds one (install with the `cu128` extra).
- Most of your published photos need to be cloaked for the protection to work; a recogniser trained
  on mostly clean photos of you learns your real face regardless.
- Cloaks survive JPEG re-encoding by design, but do not resize or filter the output more than a
  social network would.

### What this does and does not protect against

Cloaks are fixed once you publish a photo, but recognisers keep improving. Radiya-Dixit et al.
([ICLR 2022](https://arxiv.org/abs/2106.14851)) showed that a model trained *after* a cloaking tool is
released, or trained specifically against it, can defeat any such tool, including the original
Fawkes. The evaluation below measures transfer to strong recognisers of today that were not used to
build the cloaks. It does not, and cannot, promise protection against future or adaptive models.

Evaluation
----------

`eval/harness.py` downloads a subset of [LFW](http://vis-www.cs.umass.edu/lfw/), cloaks the training
photos of some identities (each toward its own target identity), trains a linear classifier on face
embeddings from **held-out** recognisers, and reports the **protection rate**: the fraction of clean
test photos of protected identities that the classifier gets wrong. 10 protected and 10 clean
identities, 10 training and 5 test photos each. The evaluators are the InsightFace packs most
deployments use (`buffalo_l`, a ResNet-50; `antelopev2`, a ResNet-100) and a vision transformer
(AdaFace ViT-B, WebFace4M); none of them is in the surrogate ensemble.

| cloaker | buffalo_l | antelopev2 | AdaFace ViT-B | LVFace-T | DSSIM face box | s / photo |
|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0 | - |
| Fawkes 1.0 (TensorFlow), mid | 0.00 | 0.00 | - | - | 0.015 | 15 (CPU) |
| Fawkes 1.0, mid, JPEG 75 | 0.00 | 0.00 | - | - | 0.015 | - |
| Fawkes 2, low | 0.74 | 0.90 | 0.20 | - | 0.012 | 35 (CPU), 1.1 (A100) |
| Fawkes 2, low, JPEG 75 | 0.70 | 0.84 | 0.22 | - | 0.012 | - |
| Fawkes 2, mid (three surrogates, 2026-09-12) | 0.74 | 0.84 | 0.32 | - | 0.012 | 56 (CPU), 1.0 (A100) |
| Fawkes 2, mid | 0.80 | 0.92 | 0.48 | 0.90 | 0.013 | 74 (CPU), 1.7 (A100) |
| Fawkes 2, mid, JPEG 75 | 0.74 | 0.84 | 0.44 | 0.92 | 0.013 | - |
| Fawkes 2, high (three surrogates, 2026-09-12) | 0.82 | 0.92 | 0.52 | 0.78 | 0.017 | about 190 (CPU), 2.8 (A100) |
| Fawkes 2, high | 0.86 | 0.92 | 0.70 | 0.96 | 0.016 | about 250 (CPU, estimated), 4.4 (A100) |
| Fawkes 2, high, JPEG 75 | 0.86 | 0.90 | 0.68 | 0.96 | 0.016 | - |

The original cloaks move the embeddings a little but every protected identity is still recognised.
Fawkes 2 defeats the classifier for most identities on the ResNet recognisers and, since the
ensemble gained two more transformers (`mid` and `high`), for two thirds of them on the
transformer evaluator; a fourth evaluator, LVFace-T, is a transformer from the same family as
three of the surrogates and is a weaker test.
Under 1:1 verification with the usual 0.3 cosine threshold, every cloaked photo in `mid` and
`high` fails to match its own identity on all three evaluators. The table is for an adversary that
trains on the cloaked photos and meets clean ones; with the sides swapped (`--clean-gallery`: the
adversary already holds clean photos and meets a cloaked one), the same cloaks give 0.80 / 0.91 /
0.55 / 0.97 in `mid` and 0.89 / 0.93 / 0.71 / 1.00 in `high`. Each number is 50 test photos, so
differences below about 0.06 are noise. Every configuration tried on the way to these settings is
in [eval/RESULTS.md](eval/RESULTS.md); the reasoning is in [docs/DECISIONS.md](docs/DECISIONS.md). See [eval/README.md](eval/README.md) for the metrics and
how to run it.

Quick Installation
------------------

From a clone of this repository, with [uv](https://docs.astral.sh/uv/) (Python 3.12, locked
dependencies). PyTorch comes from one of two extras: `cpu` or `cu128` (CUDA 12.8 wheels):

```
uv sync --extra cpu          # or: uv sync --extra cu128
uv run fawkes -d ./imgs -t ./target --mode mid
```

Plain `pip install torch .` also works on Python 3.11 or newer. The face detector (17 MB) and the surrogate
weights (about 930 MB for `mid`, 250 MB for `low`) are downloaded from Hugging Face on first use
into `fawkes/model/`; `python -m fawkes.models download` prefetches them.

Model licences
--------------

| key | model | source | licence |
|---|---|---|---|
| detector | SCRFD-10GF | [InsightFace](https://github.com/deepinsight/insightface) | non-commercial research |
| arcface_r100 | ArcFace IR-100, MS1MV3 | [InsightFace arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) | non-commercial research |
| adaface_ir101 | AdaFace IR-101, WebFace12M | [CVLface](https://github.com/mk-minchul/CVLface) | follows WebFace12M (research) |
| lvface_b | LVFace-B, Glint360K | [ByteDance LVFace](https://github.com/bytedance/LVFace) | non-commercial research |
| lvface_l | LVFace-L (ViT-L), Glint360K | [ByteDance LVFace](https://github.com/bytedance/LVFace) | non-commercial research |
| lvface_s | LVFace-S (ViT-S), Glint360K, optional | [ByteDance LVFace](https://github.com/bytedance/LVFace) | non-commercial research |
| evaluators | w600k_r50, glintr100, AdaFace ViT-B, LVFace-T | InsightFace, CVLface, LVFace | research |

Development
-----------

```
uv sync --extra cpu --extra eval
uv run pytest
```

Tests that need the surrogate weights run only when they are already in `fawkes/model/`. Tests that
need real photos run when `FAWKES_TEST_IMAGES` points to a directory of photos containing faces.

GUI
---

`app/app.py` is a small PyQt5 front end: pick the images, pick the target photo folder, protect.

Academic Research Usage
-----------------------
To protect a class in a dataset, move that label's images to a separate directory and run Fawkes on
it with `--debug`. If the images are already cropped and aligned to the 112x112 ArcFace template,
add `--no-align`. Keep the evaluators in `eval/harness.py` disjoint from the surrogates you cloak
with, or the numbers mean nothing.

### Citation

```
@inproceedings{shan2020fawkes,
  title={Fawkes: Protecting Personal Privacy against Unauthorized Deep Learning Models},
  author={Shan, Shawn and Wenger, Emily and Zhang, Jiayun and Li, Huiying and Zheng, Haitao and Zhao, Ben Y},
  booktitle={Proc. of {USENIX} Security},
  year={2020}
}
```
