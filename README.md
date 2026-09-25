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

Licence
-------
The code in this repository is released under the BSD-3-Clause licence (see [LICENSE](LICENSE));
it is a derivative of the original Fawkes by SANDLAB, University of Chicago, under the same licence.
The network definitions vendored in `fawkes/arch/` come from InsightFace and CVLface (MIT, notices
kept in the file headers).

That licence covers the code only. The pretrained detector and recogniser weights the tool downloads
from Hugging Face are **not** part of this repository and carry their own terms: every one of them is
licensed for non-commercial research use only (see [Model licences](#model-licences)). In practice
this means the tool as a whole may be used for personal privacy protection and academic research,
but not commercially, regardless of the code licence.

Usage
-----

```
fawkes -d ./imgs --mode subtle
```

or `python3 -m fawkes -d ./imgs -m subtle`. Without `--target-dir` Fawkes picks the target for you
(see [Choosing a target](#choosing-a-target)); that needs the target pool, built once with
`python -m fawkes.target_pool build` (requires the `eval` extra for LFW).

Every image in `./imgs` gets a `<name>_cloaked.png` next to it. Options:

* `-d`, `--directory`: the directory with images to protect.
* `-t`, `--target-dir`: a directory with a few photos of the person your cloaked photos should
  resemble to a recogniser (default: picked automatically). See [Choosing a target](#choosing-a-target).
* `--target-strategy`: `far` (default) or `near`, which of the matched pool identities the automatic
  target is.
* `-m`, `--mode`: `low`, `subtle`, `mid` (default) or `high`, the tradeoff between protection
  strength, visible change and run time:

  | mode | surrogates | steps | max pixel change | DSSIM budget | colour penalty | age penalty | s / face, GPU |
  |---|---|---|---|---|---|---|---|
  | low | AdaFace IR-101, ArcFace IR-100 | 60 | 16 | 0.012 | - | - | 0.6 |
  | subtle | + LVFace-B, LVFace-L, LVFace-S (transformers) | 120 | 12 | 0.006 | 30 | 1 | 2.7 |
  | mid | same five | 120 | 24 | 0.012 | 30 | 2 | 3.3 |
  | high | same five | 200 | 20 | 0.017 | 30 | 3 | 8.3 |

  `subtle` is for photos you want to post: at feed size the change is hard to see, and it protects
  less (see [Evaluation](#evaluation)). `mid` and `high` protect more and are visible as a slight
  change of the face's shading. On an 8-core CPU without a GPU, `low` takes about 35 s per face; the
  other modes take several minutes (not measured since the step counts were raised).

  Every mode also penalises what is left of your own identity in the cloaked face
  (`--self-weight`, see [docs/DECISIONS.md](docs/DECISIONS.md#10-penalise-the-residual-similarity-to-the-own-face-not-only-the-distance-to-the-target)).
  `mid` and `high` penalise the colour part of the cloak, which is what shows as reddish or
  yellowish patches on the skin, and spend the DSSIM budget on luma detail instead
  (`--chroma-weight`, [decision 12](docs/DECISIONS.md#12-penalise-the-colour-of-the-cloak-and-raise-the-luma-bound)).
  All but `low` also penalise the apparent ageing of the face: without it the cloak draws folds and
  lines under the eyes that make people look years older (`--age-weight`,
  [decision 13](docs/DECISIONS.md#13-a-matched-automatic-target-an-age-penalty-and-a-subtle-mode)).

* `--models`: comma-separated surrogate keys overriding the mode (`python -m fawkes.models list`).
* `--steps`, `--eps`, `--th`, `--no-eot`, `--self-weight`: override the mode's steps, pixel bound,
  DSSIM budget, robustness views and residual penalty.
* `--chroma-weight`: penalty on the colour (Cb/Cr) part of the cloak (30 in `mid` and `high`,
  0 in `low`). Higher values leave less visible tint and, above about 30, less protection;
  `--chroma-eps` is a hard bound on the colour change instead (0-255 units, `0` for a grey cloak),
  which costs more protection for the same tint (rounds 8 to 10 in [eval/RESULTS.md](eval/RESULTS.md)).
* `--batch-size`: faces optimised together (default 8; larger batches are faster per face).
* `--threads`: CPU threads for PyTorch.
* `--no-align`: the inputs are already 112x112 aligned faces; skip detection.
* `--format`: `png` (default) or `jpg`.
* `--debug`: print per-face statistics.

### Choosing a target

A recogniser trained on your cloaked photos should learn the target's face, not yours.

**Automatic (default).** Fawkes groups the faces in your photos into people and picks a target for
each from a pool of about 420 public identities (LFW): the same apparent sex, a similar age and skin
tone, and, among those, the face least like yours (`--target-strategy far`). Differences in age, sex
or skin tone cost budget without changing identity: they are what made earlier cloaks look older or
tinted. A matched target also protects better (rounds 11 to 16 in [eval/RESULTS.md](eval/RESULTS.md)).
A person Fawkes has cloaked before keeps their target in later runs, so every photo of you pushes
toward the same false identity (remembered in `fawkes/model/target_pool/assignments.json`; delete it
to start over).

**Your own target** (`--target-dir`). All of your photos are pushed toward the same target, so use
the same directory every time; the embedding is cached in `<target-dir>/fawkes_target.npz`.

- Use 3 to 10 photos of one person, each with exactly one clearly visible face. Photos with zero or
  several faces are skipped.
- Pick someone of your sex and roughly your age and skin tone who otherwise looks unlike you; a
  target much older than you makes the cloak draw age.
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
| Fawkes 2, mid (no colour penalty, 2026-09-13) | 0.80 | 0.92 | 0.48 | 0.90 | 0.013 | 74 (CPU), 1.7 (A100) |
| Fawkes 2, mid (no colour penalty), JPEG 75 | 0.74 | 0.84 | 0.44 | 0.92 | 0.013 | - |
| Fawkes 2, mid | 0.86 | 0.94 | 0.46 | 0.94 | 0.013 | 78 (CPU), 1.3 (H100) |
| Fawkes 2, mid, JPEG 75 | 0.82 | 0.88 | 0.38 | 0.92 | 0.013 | - |
| Fawkes 2, high (three surrogates, 2026-09-12) | 0.82 | 0.92 | 0.52 | 0.78 | 0.017 | about 190 (CPU), 2.8 (A100) |
| Fawkes 2, high (no colour penalty, 2026-09-13) | 0.86 | 0.92 | 0.70 | 0.96 | 0.016 | about 250 (CPU, estimated), 4.4 (A100) |
| Fawkes 2, high (no colour penalty), JPEG 75 | 0.86 | 0.90 | 0.68 | 0.96 | 0.016 | - |
| Fawkes 2, high | 0.92 | 0.98 | 0.64 | 0.94 | 0.016 | about 250 (CPU, estimated), 4.2 (A100) |
| Fawkes 2, high, JPEG 75 | 0.90 | 0.96 | 0.64 | 0.94 | 0.016 | - |
| Fawkes 2 (2026-09-25), subtle, automatic target | 0.60 | 0.66 | 0.22 | 0.58 | 0.007 | 2.7 (GPU) |
| Fawkes 2 (2026-09-25), mid, automatic target | 0.88 | 0.88 | 0.70 | 0.86 | 0.013 | 3.3 (GPU) |
| Fawkes 2 (2026-09-25), high, automatic target | 0.96 | 0.96 | 0.78 | 0.98 | 0.017 | 8.3 (GPU) |
| Fawkes 2 (2026-09-25), high, automatic target, JPEG 75 | 0.94 | 0.96 | 0.78 | 0.98 | 0.017 | - |

The original cloaks move the embeddings a little but every protected identity is still recognised.
Fawkes 2 defeats the classifier for most identities on the ResNet recognisers and, since the
ensemble gained two more transformers (`mid` and `high`), for two thirds of them on the
transformer evaluator; a fourth evaluator, LVFace-T, is a transformer from the same family as
three of the surrogates and is a weaker test. The current modes also penalise the colour of the
cloak (decision 12): the smooth reddish and yellowish patches the earlier cloaks left on the skin
are halved (the low-passed chroma of the change inside the face box goes from 2.0 to 0.9 grey
levels) at equal or better protection, except for a few points on the transformer evaluator in
`high` and under JPEG in `mid`.
The 2026-09-25 modes (decision 13) add the matched automatic target and the age penalty; the rows are
the sweep configurations that became the modes (R5, Q3, M4 in eval/RESULTS.md), with the
automatic target instead of the harness's random one. The earlier rows cloak toward a random LFW
identity. `subtle` is the price of a cloak that is hard to see: on the author's photos every setting
that protected more than it also showed, and the strength at which nothing showed protected nothing.
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

### GPU

Optional. The cloaker uses CUDA when torch can see it, else the CPU. Install with the `cu128` extra
(pip users: a CUDA build of torch from [pytorch.org](https://pytorch.org/get-started/locally/)); force
a device with `--device` or `FAWKES_DEVICE`. Without internet at run time, prefetch the weights with
`python -m fawkes.models download` and set `HF_HUB_OFFLINE=1`.

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
