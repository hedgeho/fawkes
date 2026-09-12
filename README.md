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
  | low | AdaFace IR-101, ArcFace IR-100 | 40 | 16 | 0.012 | none | 20 |
  | mid | + LVFace-B | 60 | 16 | 0.012 | blur, resize, JPEG | 56 |
  | high | same three | 200 | 16 | 0.017 | blur, resize, JPEG | about 190 |

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

- Run time on an 8-core CPU without a GPU is about 20 s per face in `low` and a minute per face in
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
photos of some identities, trains a linear classifier on face embeddings from **held-out**
recognisers (InsightFace `buffalo_l` and `antelopev2`, the packs most deployments use), and
reports the **protection rate**: the fraction of clean test photos of protected identities that the
classifier gets wrong. 10 protected and 10 clean identities, 10 training and 5 test photos each.

| cloaker | buffalo_l | antelopev2 | cos to own clean face | s / photo (8-core CPU) |
|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.77 | - |
| Fawkes 1.0 (TensorFlow), mid | 0.00 | 0.00 | 0.57 | 15 |
| Fawkes 1.0, mid, JPEG 75 | 0.00 | 0.00 | 0.57 | 15 |
| Fawkes 2, low | see `eval/results/` | | | |
| Fawkes 2, mid | see `eval/results/` | | | |

The original cloaks move the embeddings a little but every protected identity is still recognised.
See [eval/README.md](eval/README.md) for the metrics and how to run it.

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
| evaluators | w600k_r50, glintr100, AdaFace ViT-B | InsightFace, CVLface | research |

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
