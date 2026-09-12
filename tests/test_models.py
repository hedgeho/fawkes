"""Tests for fawkes.models (surrogate / evaluator zoo).

Fast tests need no downloads. Parity tests run only when a model's weights are already in
``fawkes.models.model_dir()`` or ``FAWKES_TEST_DOWNLOAD=1`` is set (then they download).

Fixture and reference embeddings
--------------------------------
``tests/data/aligned_112.png`` is one LFW photo (scikit-learn ``lfw_funneled``,
``Tony_Blair/Tony_Blair_0001.jpg``) detected with
``insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])``
(``det_size=(320, 320)``, best-scoring face) and aligned with
``insightface.utils.face_align.norm_crop(img, landmark=face.kps, image_size=112)``, written
with ``cv2.imwrite`` (so the PNG on disk is an ordinary RGB image).

Each ``tests/data/ref_<key>.npy`` (raw, un-normalised 512-d float32) was computed ONCE on that
PNG with the *upstream* inference path, never with the wrappers in ``fawkes.models``:

* ``w600k_r50``, ``glintr100``: ``insightface.model_zoo.get_model(path).get_feat(cv2.imread(png))``
  (``ArcFaceONNX``, insightface 2.0, onnxruntime CPU).
* ``arcface_r100`` is arcface_torch ``ms1mv3_arcface_r100_fp16``, NOT the Glint360K R100 the plan
  first named: ``glint360k_cosface_r100_fp16_0.1`` is byte-for-byte the network exported as
  antelopev2 ``glintr100.onnx`` (cosine 1.000000, identical norm 20.6178 on the fixture), so using
  it as a surrogate would make the ``glintr100`` evaluator not held out. Do not swap it back.
  Reference: insightface ``recognition/arcface_torch/inference.py`` at commit
  ``1480e705``: ``inference(weight, "r100", png)``; the script only prints the feature, so its
  module-level ``print`` was replaced by a capture function, and ``torch.load`` was wrapped with
  ``map_location="cpu"`` because the checkpoint was saved from CUDA.
* ``lvface_b``: ``bytedance/LVFace`` ``inference.py`` at commit ``80bd20c8``:
  ``inference(weight, "vit_b_dp005_mask_005", png)`` with the same print capture and
  ``map_location`` wrapper (run with ``uv run --with timm``).
* ``adaface_ir101``, ``adaface_vit_b``: the model repos' own ``wrapper.py`` through
  ``transformers.AutoModel.from_pretrained(path, trust_remote_code=True)`` in a throwaway
  environment (``uv run --with 'transformers<5' --with timm --with omegaconf --with fvcore
  --with torchvision``; transformers 5.x cannot load that wrapper), fed exactly as the model
  card shows: ``PIL.Image.open(png).convert("RGB")`` -> ``ToTensor`` ->
  ``Normalize(mean=[0.5]*3, std=[0.5]*3)``.

Channel order. All six upstream paths feed RGB (arcface_torch and LVFace ``inference.py`` do
``cv2.COLOR_BGR2RGB``; CVLface ``config.json`` says ``color_space: RGB`` and its wrapper does not
flip; ``ArcFaceONNX.get_feat`` uses ``swapRB=True``), and the parity tests reproduce those paths,
which settles the AdaFace BGR-vs-RGB question in favour of RGB for the CVLface checkpoints.
Feeding the BGR-flipped fixture instead lowers the cosine to the reference to 0.81-0.96. A
same-vs-different identity margin on 30 LFW crops was inconclusive (differences under 0.04 and
not consistent in sign), so the wrapper convention rests on the upstream code, not on that.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fawkes import models

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "aligned_112.png"
ALL_KEYS = list(models.SURROGATES) + list(models.EVALUATORS)


def _weights_available(key: str) -> bool:
    return models.weight_path(key).exists() or os.environ.get("FAWKES_TEST_DOWNLOAD") == "1"


def _needs_weights(key: str):
    return pytest.mark.skipif(not _weights_available(key), reason=f"weights for {key} not downloaded")


def _fixture_rgb() -> np.ndarray:
    return np.asarray(Image.open(FIXTURE).convert("RGB"))


def _fixture_tensor() -> torch.Tensor:
    return torch.from_numpy(_fixture_rgb().copy()).permute(2, 0, 1).float().div(255)[None]


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- no downloads

def test_registries_disjoint_and_filled():
    assert set(models.SURROGATES).isdisjoint(models.EVALUATORS)
    assert len(models.SURROGATES) >= 3 and len(models.EVALUATORS) >= 3
    seen_files = set()
    for reg in (models.SURROGATES, models.EVALUATORS):
        for key, s in reg.items():
            assert s.key == key
            for field in ("repo_id", "filename", "kind", "arch", "licence", "note"):
                assert getattr(s, field), f"{key}.{field} empty"
            assert s.kind in ("torch", "onnx")
            assert s.arch == "onnx" if s.kind == "onnx" else s.arch in models._ARCHS
            assert s.embed_dim == 512
            assert s.sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", s.sha256)
            assert (s.repo_id, s.filename) not in seen_files, "same weight file registered twice"
            seen_files.add((s.repo_id, s.filename))


def test_fixture_and_references_exist():
    assert FIXTURE.exists()
    assert _fixture_rgb().shape == (112, 112, 3)
    for key in ALL_KEYS:
        ref = np.load(DATA / f"ref_{key}.npy")
        assert ref.shape == (512,) and np.isfinite(ref).all()
    assert sum(p.stat().st_size for p in DATA.iterdir()) < 1_000_000


def test_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FAWKES_MODEL_DIR", str(tmp_path))
    assert models.model_dir() == tmp_path
    assert models.weight_path("lvface_b") == tmp_path / "lvface_b" / models.SURROGATES["lvface_b"].filename
    monkeypatch.delenv("FAWKES_MODEL_DIR")
    assert models.model_dir().name == "model" and models.model_dir().parent.name == "fawkes"


def test_unknown_keys_rejected():
    with pytest.raises(KeyError):
        models.spec("nope")
    with pytest.raises(KeyError):
        models.load_surrogate("w600k_r50")     # evaluator, not a surrogate
    with pytest.raises(KeyError):
        models.load_evaluator("lvface_b")      # surrogate, not an evaluator


def test_cli_list_runs(capsys):
    assert models.main(["list"]) == 0
    out = capsys.readouterr().out
    for key in ALL_KEYS:
        assert key in out


# --------------------------------------------------------------------------- parity

_cache: dict[str, object] = {}


def _surrogate(key: str):
    if key not in _cache:
        _cache[key] = models.load_surrogate(key)
    return _cache[key]


@pytest.mark.parametrize("key", [pytest.param(k, marks=_needs_weights(k)) for k in models.SURROGATES])
def test_surrogate_matches_upstream(key):
    model = _surrogate(key)
    with torch.no_grad():
        emb = model(_fixture_tensor())
    assert emb.shape == (1, 512) and emb.dtype == torch.float32
    ref = np.load(DATA / f"ref_{key}.npy")
    assert _cos(emb[0].numpy(), ref) > 0.999


@pytest.mark.parametrize("key", [pytest.param(k, marks=_needs_weights(k)) for k in models.EVALUATORS])
def test_evaluator_matches_upstream(key):
    fn = models.load_evaluator(key)
    rgb = _fixture_rgb()
    emb = fn(rgb[None])
    assert emb.shape == (1, 512) and emb.dtype == np.float32
    assert abs(np.linalg.norm(emb[0]) - 1) < 1e-4
    ref = np.load(DATA / f"ref_{key}.npy")
    assert _cos(emb[0], ref) > 0.999
    # batching and 3-d input give the same answer
    both = fn(np.stack([rgb, rgb[:, ::-1]]))
    assert both.shape == (2, 512)
    assert _cos(both[0], emb[0]) > 0.9999
    assert _cos(fn(rgb)[0], emb[0]) > 0.9999


@pytest.mark.parametrize("key", [pytest.param(k, marks=_needs_weights(k)) for k in models.SURROGATES])
def test_surrogate_is_frozen_unit_norm_and_batchable(key):
    model = _surrogate(key)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    model.train()  # must stay in eval mode: BatchNorm statistics are part of the model
    assert not model.training
    x = _fixture_tensor()
    batch = torch.cat([x, x.flip(3), torch.rand_like(x)])
    # NHWC-permuted (channels_last strides) input must work too
    nhwc = batch.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    with torch.no_grad():
        emb = model(batch)
        emb2 = model(nhwc)
    assert emb.shape == (3, 512)
    assert torch.allclose(emb.norm(dim=1), torch.ones(3), atol=1e-4)
    assert torch.allclose(emb, emb2, atol=1e-5)
    assert _cos(emb[0], emb[1]) > 0.7          # mirrored face: same identity
    assert _cos(emb[0], emb[2]) < 0.5          # noise: not
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 100, 100))


@_needs_weights("adaface_ir101")
def test_surrogate_gradient_finite_difference():
    """Autograd gradient w.r.t. input pixels agrees with central finite differences (float64)."""
    torch.manual_seed(0)
    model = _surrogate("adaface_ir101").double()
    x = _fixture_tensor().double()
    target = torch.nn.functional.normalize(torch.randn(1, 512, dtype=torch.float64), dim=1)

    def loss_of(inp):
        return (1 - (model(inp) * target).sum())

    xg = x.clone().requires_grad_(True)
    loss_of(xg).backward()
    grad = xg.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # float64 allows a tiny step, which keeps the central difference on one side of the
    # PReLU kinks that a 1e-3 step (a quarter grey level) crosses somewhere in 65M activations.
    eps = 1e-6
    for (c, i, j) in [(0, 56, 56), (1, 30, 80), (2, 90, 20), (0, 10, 100)]:
        xp = x.clone(); xp[0, c, i, j] += eps
        xm = x.clone(); xm[0, c, i, j] -= eps
        with torch.no_grad():
            fd = (loss_of(xp) - loss_of(xm)).item() / (2 * eps)
        ag = grad[0, c, i, j].item()
        assert abs(fd - ag) <= 1e-6 + 1e-3 * abs(ag), (c, i, j, fd, ag)
    model.float()


@_needs_weights("w600k_r50")
def test_download_is_idempotent_and_verifies(tmp_path, monkeypatch):
    path = models.download("w600k_r50")
    assert path.exists() and path == models.weight_path("w600k_r50")
    assert models.sha256sum(path) == models.EVALUATORS["w600k_r50"].sha256
    # A corrupted file fails verification and is removed.
    monkeypatch.setenv("FAWKES_MODEL_DIR", str(tmp_path))
    bad = models.weight_path("w600k_r50")
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        models.download("w600k_r50", verify=True)
    assert not bad.exists()
