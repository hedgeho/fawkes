"""Smoke tests for fawkes.

Fast tests need no downloads. Tests that need real photos run when FAWKES_TEST_IMAGES points to a
directory of face images; tests that need the surrogate weights run when they are already in
fawkes/model/ (see `python -m fawkes.models download`).
"""
import glob
import os

import numpy as np
import pytest
from PIL import Image

import fawkes
from fawkes import utils
from fawkes.protection import Fawkes, MODES

IMAGE_DIR = os.environ.get("FAWKES_TEST_IMAGES")
needs_images = pytest.mark.skipif(not IMAGE_DIR, reason="FAWKES_TEST_IMAGES not set")


def _random_image(shape=(112, 112, 3), seed=0):
    return np.random.RandomState(seed).uniform(0, 255, shape).astype(np.float32)


def _have_surrogates(keys):
    from fawkes.models import SURROGATES, model_dir
    return all((model_dir() / k / SURROGATES[k].filename).exists() for k in keys)


def test_load_image_applies_exif_orientation(tmp_path):
    img = Image.fromarray(_random_image((20, 40, 3)).astype(np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 270 degrees on load
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif)
    arr = utils.load_image(str(path))
    assert arr.shape == (40, 20, 3)


def test_filter_image_paths_skips_non_images(tmp_path):
    good = tmp_path / "a.png"
    Image.fromarray(_random_image((16, 16, 3)).astype(np.uint8)).save(good)
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "subdir").mkdir()
    paths, images = utils.filter_image_paths(sorted(glob.glob(str(tmp_path / "*"))))
    assert paths == [str(good)]
    assert images[0].shape == (16, 16, 3)


def test_dump_image_roundtrip(tmp_path):
    img = _random_image((20, 30, 3))
    utils.dump_image(img, str(tmp_path / "x.png"))
    back = utils.load_image(str(tmp_path / "x.png"))
    assert np.abs(back - img.round()).max() == 0
    utils.dump_image(img, str(tmp_path / "x.jpg"), format="jpeg")
    assert utils.load_image(str(tmp_path / "x.jpg")).shape == (20, 30, 3)


def test_argument_validation_happens_before_any_model_loads(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        Fawkes(mode="min", target_dir=str(tmp_path))
    assert Fawkes(mode="low").target_dir is None  # automatic target from the pool
    with pytest.raises(ValueError, match="not a directory"):
        Fawkes(mode="low", target_dir=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="unknown surrogate"):
        Fawkes(mode="low", target_dir=str(tmp_path), models=["nope"])


def test_named_modes_define_all_parameters():
    from fawkes.models import SURROGATES
    for name, params in MODES.items():
        assert {"models", "steps", "eps", "dssim_budget", "eot_samples", "stop_cos"} <= set(params), name
        assert all(m in SURROGATES for m in params["models"]), name


def test_mode_overrides_are_applied(tmp_path):
    f = Fawkes(mode="mid", target_dir=str(tmp_path), steps=3, eps=5.0, dssim_budget=0.02, eot_samples=0,
               models=["adaface_ir101"])
    assert f.params.steps == 3 and f.params.eps == 5.0 and f.params.dssim_budget == 0.02
    assert f.params.eot_samples == 0 and f.model_keys == ["adaface_ir101"]


def test_run_protection_without_images_returns_3(tmp_path):
    (tmp_path / "t").mkdir()
    assert Fawkes(mode="low", target_dir=str(tmp_path / "t")).run_protection([]) == 3


def test_version_is_v2():
    assert fawkes.__version__.startswith("2.")


@needs_images
@pytest.mark.skipif(not _have_surrogates(MODES["low"]["models"]), reason="low-mode surrogate not downloaded")
def test_end_to_end_low_mode(tmp_path):
    """Detect, build a target from the photos themselves, cloak, and write outputs."""
    from fawkes.detect import Detector
    paths = [p for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "*"))) if "_cloaked" not in p]
    assert paths, "no images in FAWKES_TEST_IMAGES"
    work = tmp_path / "imgs"
    work.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    # a real face as the target: crop the first detected face out of the first photo
    img = utils.load_image(paths[0])
    face = Detector().detect(img)[0]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    m = int(0.4 * (x2 - x1))
    utils.dump_image(img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m], str(target / "t.png"))
    utils.dump_image(img, str(work / "a.png"))

    # the target is the same person, so disable early stopping to force a perturbation; no robustness
    # views, so that every one of the three steps is a clean step that records its perturbation
    protector = Fawkes(mode="low", target_dir=str(target), steps=3, batch_size=2, stop_cos=1.01, eot_samples=0)
    assert protector.run_protection([str(work / "a.png")], debug=True) == 1
    out = utils.load_image(str(work / "a_cloaked.png"))
    assert out.shape == img.shape
    diff = np.abs(out - img)
    assert diff.max() > 0 and diff.max() <= MODES["low"]["eps"] + 1
    assert (target / "fawkes_target.npz").exists()
    # second run reuses the saved target and is deterministic
    protector2 = Fawkes(mode="low", target_dir=str(target), steps=3, batch_size=2, stop_cos=1.01, eot_samples=0)
    assert protector2.run_protection([str(work / "a.png")]) == 1
    np.testing.assert_array_equal(utils.load_image(str(work / "a_cloaked.png")), out)
