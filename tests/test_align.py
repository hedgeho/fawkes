import glob
import os

import cv2
import numpy as np
import pytest
import torch

from fawkes import align

IMAGE_DIR = os.environ.get("FAWKES_TEST_IMAGES")
needs_images = pytest.mark.skipif(not IMAGE_DIR, reason="FAWKES_TEST_IMAGES not set")

# five plausible landmarks of a tilted face in a 300x400 photo
KPS = np.array([[120.0, 150.0], [190.0, 140.0], [160.0, 190.0], [130.0, 230.0], [195.0, 222.0]], np.float32)


def _smooth_photo(h=400, w=300, seed=0):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([100 + 80 * np.sin(x / 23) + 20 * np.cos(y / 17),
                    120 + 60 * np.cos(x / 31 + y / 41),
                    90 + 70 * np.sin(y / 19) * np.cos(x / 29)], -1)
    img += rng.uniform(-8, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def test_estimate_norm_matches_insightface():
    from insightface.utils import face_align
    ours = align.estimate_norm(KPS)
    theirs = face_align.estimate_norm(KPS, 112)
    np.testing.assert_allclose(ours, theirs, atol=1e-3)


def test_estimate_norm_lands_landmarks_on_template():
    m = align.estimate_norm(KPS)
    projected = align.apply_affine(m, KPS)
    # least-squares fit: every landmark within a few pixels of its template point
    assert np.abs(projected - align.ARCFACE_DST).max() < 6


def test_warp_to_template_matches_cv2():
    photo = _smooth_photo()
    m = align.estimate_norm(KPS)
    ref = cv2.warpAffine(photo, m, (112, 112), borderValue=0.0)
    x = torch.from_numpy(photo).permute(2, 0, 1)[None]
    ours = align.warp_to_template(x, m)[0].permute(1, 2, 0).numpy()
    diff = np.abs(ours - ref)
    assert diff.max() <= 1.5, diff.max()
    assert diff.mean() < 0.2, diff.mean()


def test_warp_is_differentiable_and_batched():
    photo = _smooth_photo()
    x = torch.from_numpy(photo).permute(2, 0, 1)[None].repeat(2, 1, 1, 1).requires_grad_(True)
    m = np.stack([align.estimate_norm(KPS), align.estimate_norm(KPS + 5)])
    out = align.warp_to_template(x, m)
    assert out.shape == (2, 3, 112, 112)
    out.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_make_crop_keeps_alignment_consistent():
    photo = _smooth_photo()
    bbox = np.array([100, 120, 220, 260], np.float32)
    crop = align.make_crop(photo, bbox, KPS, max_side=448)
    assert crop.scale == 1.0
    x0, y0, x1, y1 = crop.box
    assert x0 <= 100 and y0 <= 120 and x1 >= 220 and y1 >= 260, "crop must contain the detector box"
    # warping the crop with its own matrix gives the same template image as warping the photo
    from_photo = cv2.warpAffine(photo, align.estimate_norm(KPS), (112, 112), borderValue=0.0)
    from_crop = cv2.warpAffine(crop.image, crop.matrix, (112, 112), borderValue=0.0)
    assert np.abs(from_photo - from_crop).max() < 1e-2


def test_make_crop_downscales_large_faces():
    photo = _smooth_photo(h=1200, w=900)
    bbox = np.array([100, 100, 800, 900], np.float32)
    kps = KPS * 3
    crop = align.make_crop(photo, bbox, kps, max_side=448)
    assert max(crop.image.shape[:2]) == 448
    assert crop.scale < 1.0
    from_photo = cv2.warpAffine(photo, align.estimate_norm(kps), (112, 112), borderValue=0.0)
    from_crop = cv2.warpAffine(crop.image, crop.matrix, (112, 112), borderValue=0.0)
    assert np.abs(from_photo - from_crop).mean() < 3.0  # resampling differences only


def test_paste_back_identity_and_delta():
    photo = _smooth_photo()
    bbox = np.array([100, 120, 220, 260], np.float32)
    crop = align.make_crop(photo, bbox, KPS)
    same = align.paste_back(photo, crop, crop.image)
    np.testing.assert_array_equal(same, photo)
    cloaked = np.clip(crop.image + 3, 0, 255)
    out = align.paste_back(photo, crop, cloaked)
    x0, y0, x1, y1 = crop.box
    inside = out[y0:y1, x0:x1] - photo[y0:y1, x0:x1]
    assert 2.0 < inside.mean() <= 3.0
    outside = np.array(out)
    outside[y0:y1, x0:x1] = photo[y0:y1, x0:x1]
    np.testing.assert_array_equal(outside, photo)


def test_paste_back_upsamples_delta():
    photo = _smooth_photo(h=1200, w=900)
    crop = align.make_crop(photo, np.array([100, 100, 800, 900], np.float32), KPS * 3, max_side=448)
    out = align.paste_back(photo, crop, np.clip(crop.image + 4, 0, 255))
    x0, y0, x1, y1 = crop.box
    assert abs((out[y0:y1, x0:x1] - photo[y0:y1, x0:x1]).mean() - 4) < 0.5


def test_template_crop_is_identity_for_112():
    img = _smooth_photo(112, 112)
    crop = align.template_crop(img)
    warped = cv2.warpAffine(img, crop.matrix, (112, 112), borderValue=0.0)
    assert np.abs(warped - img).max() < 1e-2


@needs_images
def test_detector_finds_faces_with_landmarks():
    from fawkes.detect import Detector
    from fawkes.utils import load_image
    paths = [p for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "*"))) if "_cloaked" not in p]
    img = load_image(paths[0])
    faces = Detector().detect(img)
    assert faces, "no face detected"
    assert faces[0].kps.shape == (5, 2)
    crop = align.make_crop(img, faces[0].bbox, faces[0].kps)
    aligned = cv2.warpAffine(crop.image, crop.matrix, (112, 112))
    assert aligned.shape == (112, 112, 3) and aligned.std() > 10
