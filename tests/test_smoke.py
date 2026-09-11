"""Smoke tests for fawkes.

Fast tests need no downloads. Tests that need the feature-extractor models
(about 300 MB) run only if the models are already in fawkes/model/ or
FAWKES_TEST_DOWNLOAD=1 is set. Face-detection tests need real photos; point
FAWKES_TEST_IMAGES at a directory of face images to enable them.
"""
import glob
import os

import numpy as np
import pytest
from PIL import Image

import fawkes
from fawkes import utils
from fawkes.align_face import align, aligner
from fawkes.protection import Fawkes

MODEL_DIR = os.path.join(os.path.dirname(fawkes.__file__), "model")
HAVE_MODELS = (os.path.exists(os.path.join(MODEL_DIR, "extractor_2.h5"))
               or os.environ.get("FAWKES_TEST_DOWNLOAD") == "1")
IMAGE_DIR = os.environ.get("FAWKES_TEST_IMAGES")

needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="extractor models not downloaded")
needs_images = pytest.mark.skipif(not IMAGE_DIR, reason="FAWKES_TEST_IMAGES not set")


def _random_image(shape=(112, 112, 3), seed=0):
    return np.random.RandomState(seed).uniform(0, 255, shape).astype(np.float32)


def _smooth_image():
    """A smooth gradient with a soft blob; SSIM behaves sensibly on it, noise does not."""
    y, x = np.mgrid[0:112, 0:112].astype(np.float32) / 111.0
    img = np.stack([60 + 150 * x, 80 + 120 * y, 200 - 100 * (x * y)], -1)
    img += 25 * np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2) / 0.02)[..., None]
    return np.clip(img, 0, 255)


def test_get_ends_centres_window():
    assert utils.get_ends(10, 4) == (3, 7)
    assert utils.get_ends(5, 5) == (0, 5)


def test_resize_changes_shape_and_keeps_range():
    out = utils.resize(_random_image((40, 60, 3)), (112, 112))
    assert out.shape == (112, 112, 3)
    assert out.min() >= 0 and out.max() <= 255


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


def test_faces_no_align_uses_whole_image():
    img = _random_image((80, 120, 3))
    faces = utils.Faces(["x.png"], [img], aligner=None, verbose=0, no_align=True)
    assert faces.cropped_faces.shape == (1, 112, 112, 3)
    protected = np.clip(faces.cropped_faces + 10, 0, 255)
    merged, missing = faces.merge_faces(protected, faces.cropped_faces)
    assert missing == []
    assert merged[0].shape == img.shape, "cloak must be merged back at the original size"
    assert 0 <= merged[0].min() and merged[0].max() <= 255
    assert 0 < np.abs(merged[0] - img).mean() <= 10


def test_hash_file_auto_detects_algorithm(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"fawkes")
    md5 = utils._hash_file(str(f), "md5")
    assert utils.validate_file(str(f), md5, algorithm="auto")
    sha = utils._hash_file(str(f), "sha256")
    assert utils.validate_file(str(f), sha, algorithm="auto")


def test_sanitize_legacy_config_strips_groups():
    cfg = {"layers": [
        {"class_name": "Functional", "config": {"layers": [
            {"class_name": "DepthwiseConv2D", "config": {"groups": 1, "kernel_size": [3, 3]}}]}},
        {"class_name": "Conv2D", "config": {"groups": 1}},
    ]}
    utils._sanitize_legacy_config(cfg)
    assert "groups" not in cfg["layers"][0]["config"]["layers"][0]["config"]
    assert cfg["layers"][1]["config"]["groups"] == 1


def test_faces_marks_images_without_a_face():
    noise = _random_image((160, 160, 3), seed=3)
    faces = utils.Faces(["noise.png"], [noise], aligner(), verbose=0)
    assert len(faces.cropped_faces) == 0
    assert faces.images_without_face == [0]


def test_mode_validation_happens_before_any_model_loads():
    with pytest.raises(ValueError, match="custom"):
        Fawkes(gpu=None, mode="custom", th=0.01)
    with pytest.raises(ValueError, match="mode must be one of"):
        Fawkes(gpu=None, mode="min")


def test_named_modes_define_all_parameters():
    from fawkes.protection import MODES
    for name, params in MODES.items():
        assert set(params) == {"th", "max_step", "lr", "sd", "extractors"}, name


@needs_images
def test_align_detects_faces_and_merge_roundtrips():
    paths = [p for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "*"))) if "_cloaked" not in p]
    paths, images = utils.filter_image_paths(paths)
    assert paths, "no images found in FAWKES_TEST_IMAGES"
    cropped, boxes = align(images[0], aligner())
    assert len(cropped) >= 1
    assert all(c.shape[0] >= 30 and c.shape[1] >= 30 for c in cropped)

    faces = utils.Faces(paths[:1], images[:1], aligner(), verbose=0)
    merged, missing = faces.merge_faces(faces.cropped_faces, faces.cropped_faces)
    assert missing == []
    np.testing.assert_allclose(merged[0], images[0], atol=1e-3)


@needs_models
@pytest.mark.parametrize("name", ["extractor_2", "extractor_0"])
def test_extractors_load_and_embed(name):
    extractor = utils.load_extractor(name)
    emb = np.asarray(extractor(_random_image((2, 112, 112, 3))))
    assert emb.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert np.isfinite(emb).all()


@needs_models
def test_custom_mode_uses_given_parameters():
    protector = Fawkes(gpu=None, mode="custom", th=0.02, max_step=7, lr=3, sd=1e5)
    assert (protector.th, protector.max_step, protector.lr, protector.sd) == (0.02, 7, 3, 1e5)
    assert len(protector.feature_extractors_ls) == 2


@needs_models
def test_end_to_end_no_align(tmp_path):
    src = tmp_path / "face.png"
    Image.fromarray(_smooth_image().astype(np.uint8)).save(src)

    protector = Fawkes(gpu=None, mode="low")
    protector.max_step = 3
    status = protector.run_protection([str(src)], batch_size=1, format="png", no_align=True)
    assert status == 1

    out = tmp_path / "face_cloaked.png"
    assert out.exists()
    cloaked = np.asarray(Image.open(out).convert("RGB")).astype(np.float32)
    original = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    assert cloaked.shape == original.shape
    diff = np.abs(cloaked - original)
    assert diff.max() > 0, "no cloak was applied"
    assert diff.max() <= 20, "cloak exceeds the per-pixel clip range"
