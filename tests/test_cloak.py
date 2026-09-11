import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fawkes import align, cloak

torch.set_num_threads(2)


class DummySurrogate(nn.Module):
    """Small fixed random conv net with a normalised 64-d output; stands in for a recogniser."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 8, 5, stride=4)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.fc = nn.Linear(16 * 13 * 13, 64)
        for prm in self.parameters():
            prm.data = torch.randn(prm.shape, generator=g) * 0.1
            prm.requires_grad_(False)

    def forward(self, x):
        x = F.relu(self.conv1((x - 0.5) * 2))
        x = F.relu(self.conv2(x))
        return F.normalize(self.fc(x.flatten(1)), dim=1)


def _face_photo(h=200, w=180, seed=0):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([120 + 60 * np.sin(x / 15 + seed), 110 + 50 * np.cos(y / 13), 100 + 40 * np.sin((x + y) / 21)], -1)
    img += rng.uniform(-5, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


KPS = np.array([[60.0, 80.0], [120.0, 80.0], [90.0, 115.0], [65.0, 150.0], [115.0, 150.0]], np.float32)
BBOX = np.array([40, 40, 140, 180], np.float32)


def _crops(n):
    return [align.make_crop(_face_photo(seed=i), BBOX, KPS + i) for i in range(n)]


def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity
    a = _face_photo()
    b = np.clip(a + np.random.RandomState(1).normal(0, 6, a.shape), 0, 255).astype(np.float32)
    ref = structural_similarity(a, b, data_range=255, channel_axis=-1, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    ta = torch.from_numpy(a).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b).permute(2, 0, 1)[None]
    ours = float(cloak.ssim(ta, tb)[0])
    assert abs(ours - ref) < 0.01, (ours, ref)
    assert abs(float(cloak.dssim(ta, tb, torch.ones(1, 1, *a.shape[:2]))[0]) - (1 - ref) / 2) < 0.005


def test_eot_transforms_are_differentiable_and_bounded():
    x = torch.rand(2, 3, 112, 112, requires_grad=True)
    y = cloak.jpeg_approx(cloak.resize_round_trip(cloak.gaussian_blur(x, 1.0), 0.8), 75.0)
    assert y.shape == x.shape and float(y.detach().min()) >= 0 and float(y.detach().max()) <= 1
    y.sum().backward()
    assert float(x.grad.abs().sum()) > 0
    # JPEG at high quality is close to identity, at low quality it is not
    x = torch.rand(1, 3, 112, 112)
    assert float((cloak.jpeg_approx(x, 95.0) - x).abs().mean()) < float((cloak.jpeg_approx(x, 20.0) - x).abs().mean())


def test_cloak_moves_embeddings_toward_target_within_budget():
    surrogates = {"a": DummySurrogate(1), "b": DummySurrogate(2)}
    params = cloak.CloakParams(steps=40, lr=2.0, eps=10.0, dssim_budget=0.02, eot_samples=1,
                               stop_cos=0.99, patience=40, batch_size=4)
    cl = cloak.Cloaker(surrogates, params)
    crops = _crops(3)
    target_crop = align.make_crop(_face_photo(seed=9), BBOX, KPS)
    targets = {k: v[0] for k, v in cl.embed([target_crop]).items()}
    before = cl.embed(crops)
    result = cl.cloak(crops, targets)

    assert len(result.images) == 3
    for i, (img, crop) in enumerate(zip(result.images, crops)):
        assert img.shape == crop.image.shape
        assert np.abs(img - crop.image).max() <= params.eps + 1e-3
        assert result.dssim[i] <= params.dssim_budget * 1.05 + 1e-6
    for j, k in enumerate(surrogates):
        before_cos = (before[k] * targets[k][None]).sum(1)
        assert (result.cos[:, j] > before_cos + 0.05).all(), (k, before_cos, result.cos[:, j])


def test_cloak_early_stops_when_target_reached():
    surrogates = {"a": DummySurrogate(3)}
    crops = _crops(2)
    cl = cloak.Cloaker(surrogates, cloak.CloakParams(steps=50, eot_samples=0, stop_cos=0.0, patience=50))
    targets = {k: v[0] for k, v in cl.embed(crops[:1]).items()}
    # a cosine threshold of 0 is met on the first step for anything remotely similar
    result = cl.cloak(crops, targets)
    assert (result.steps <= 2).all(), result.steps


def test_cloak_is_deterministic_for_a_seed():
    surrogates = {"a": DummySurrogate(4)}
    crops = _crops(1)
    targets = {k: v[0] for k, v in cloak.Cloaker(surrogates).embed(_crops(2)[1:]).items()}
    p = cloak.CloakParams(steps=5, eot_samples=1, seed=3)
    r1 = cloak.Cloaker(surrogates, p).cloak(crops, targets)
    r2 = cloak.Cloaker(surrogates, p).cloak(crops, targets)
    np.testing.assert_array_equal(r1.images[0], r2.images[0])
