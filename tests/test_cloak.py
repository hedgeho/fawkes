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


def test_self_penalty_lowers_similarity_to_own_face():
    surrogates = {"a": DummySurrogate(1), "b": DummySurrogate(2)}
    common = dict(steps=40, lr=2.0, eps=10.0, dssim_budget=0.02, eot_samples=0, stop_cos=1.01, patience=40, batch_size=4)
    crops = _crops(2)
    target_crop = align.make_crop(_face_photo(seed=9), BBOX, KPS)
    residual = {}
    for self_weight in (0.0, 3.0):
        cl = cloak.Cloaker(surrogates, cloak.CloakParams(self_weight=self_weight, **common))
        targets = {k: v[0] for k, v in cl.embed([target_crop]).items()}
        before = cl.embed(crops)
        result = cl.cloak(crops, targets)
        after = cl.embed([align.FaceCrop(box=c.box, scale=c.scale, image=img, kps=c.kps)
                          for c, img in zip(crops, result.images)])
        residual[self_weight] = np.mean([(before[k] * after[k]).sum(axis=1).mean() for k in surrogates])
        assert (result.dssim <= 0.02 * 1.05 + 1e-4).all()
    assert residual[3.0] < residual[0.0]


def test_laggard_weighting_runs_and_reaches_target():
    surrogates = {"a": DummySurrogate(1), "b": DummySurrogate(2)}
    params = cloak.CloakParams(steps=30, lr=2.0, eps=10.0, dssim_budget=0.02, eot_samples=1, laggard=0.1,
                               self_weight=1.0, stop_cos=1.01, patience=30, batch_size=4)
    cl = cloak.Cloaker(surrogates, params)
    crops = _crops(2)
    targets = {k: v[0] for k, v in cl.embed([align.make_crop(_face_photo(seed=9), BBOX, KPS)]).items()}
    result = cl.cloak(crops, targets)
    assert result.cos.shape == (2, 2) and (result.cos > 0.3).all()


class TinyViT(nn.Module):
    """Two-block VisionTransformer on the fixture size, random weights; exercises the PNA switch."""

    def __init__(self, seed=0):
        super().__init__()
        from fawkes.arch.vit import VisionTransformer
        torch.manual_seed(seed)
        self.net = VisionTransformer(img_size=112, patch_size=16, num_classes=64, embed_dim=32, depth=2,
                                     num_heads=4, norm_layer="ln", mask_ratio=0.0)
        for prm in self.parameters():
            prm.requires_grad_(False)
        self.eval()  # BatchNorm1d in the feature head must use running statistics

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x):
        return F.normalize(self.net((x - 0.5) * 2), dim=1)


def test_pna_changes_gradient_not_forward():
    from fawkes.arch.vit import Attention
    model = TinyViT()
    x = torch.rand(2, 3, 112, 112)
    target = F.normalize(torch.randn(1, 64), dim=1)

    def grad_and_out(pna):
        for m in model.modules():
            if isinstance(m, Attention):
                m.pna = pna
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        (1 - (out * target).sum(dim=1)).sum().backward()
        return xg.grad.clone(), out.detach()

    g0, o0 = grad_and_out(False)
    g1, o1 = grad_and_out(True)
    assert torch.allclose(o0, o1)
    assert torch.isfinite(g1).all() and g1.abs().sum() > 0
    assert not torch.allclose(g0, g1)
    # the cloaker sets the flag on every attention module from CloakParams
    cl = cloak.Cloaker({"vit": model}, cloak.CloakParams(pna=True))
    assert all(m.pna for m in model.modules() if isinstance(m, Attention))
    cloak.Cloaker({"vit": model}, cloak.CloakParams(pna=False))
    assert not any(m.pna for m in model.modules() if isinstance(m, Attention))


def test_block_dropout_mask_shape_and_rate():
    rng = np.random.RandomState(0)
    grad = torch.ones(3, 3, 100, 90)
    m = cloak._block_dropout_mask(grad, 0.5, rng)
    assert m.shape == (3, 1, 100, 90)
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert 0.35 < m.mean().item() < 0.65
    # constant within each 8x8 cell
    assert torch.equal(m[:, :, :8, :8], m[:, :, :1, :1].expand(-1, -1, 8, 8))
    assert cloak._block_dropout_mask(grad, 1.0, rng).min() == 1.0


def test_patchout_cloak_runs_and_moves_toward_target():
    surrogates = {"a": DummySurrogate(1)}
    crops = _crops(2)
    targets = {k: v[0] for k, v in cloak.Cloaker(surrogates).embed(_crops(3)[2:]).items()}
    params = cloak.CloakParams(steps=30, lr=2.0, eps=10.0, dssim_budget=0.02, eot_samples=1, patchout=0.5,
                               stop_cos=0.99, patience=30)
    before = cloak.Cloaker(surrogates, params).embed(crops)["a"] @ targets["a"]
    res = cloak.Cloaker(surrogates, params).cloak(crops, targets)
    assert (res.cos[:, 0] > before + 0.05).all()
    assert (res.dssim <= 0.02 * 1.05 + 1e-6).all()


@pytest.mark.parametrize("params", [dict(tgr=0.25), dict(sgm=0.6), dict(pna=True, tgr=0.25, sgm=0.6)])
def test_vit_backward_switches_change_gradient_not_forward(params):
    model = TinyViT()
    x = torch.rand(2, 3, 112, 112)
    target = F.normalize(torch.randn(1, 64), dim=1)

    def run(p):
        cloak.Cloaker({"vit": model}, cloak.CloakParams(**p))  # applies the switches to the modules
        xg = x.clone().requires_grad_(True)
        out = model(xg)
        (1 - (out * target).sum(dim=1)).sum().backward()
        return out.detach(), xg.grad.clone()

    o0, g0 = run({})
    o1, g1 = run(params)
    assert torch.allclose(o0, o1)
    assert torch.isfinite(g1).all() and g1.abs().sum() > 0 and not torch.allclose(g0, g1)
    run({})  # switches reset to the defaults
    assert all(m.tgr == 0 for m in model.modules() if hasattr(m, "tgr"))
    assert all(m.sgm == 1.0 for m in model.modules() if hasattr(m, "sgm"))


def test_token_mask_only_on_augmented_steps_and_changes_forward():
    model = TinyViT()
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        o0 = model(x)
        model.net.adv_mask_ratio = 0.3
        torch.manual_seed(0)
        o1 = model(x)
        model.net.adv_mask_ratio = 0.0
        assert torch.allclose(o0, model(x))
    assert not torch.allclose(o0, o1)
    cl = cloak.Cloaker({"vit": model}, cloak.CloakParams(token_mask=0.3, steps=4, eot_samples=1))
    assert model.net.adv_mask_ratio == 0.0
    crops = _crops(1)
    targets = {"vit": F.normalize(torch.randn(64), dim=0).numpy()}
    cl.cloak(crops, targets)
    assert model.net.adv_mask_ratio == 0.0  # reset after cloaking


def test_blurred_delta_is_smooth_and_within_bounds():
    surrogates = {"a": DummySurrogate(1)}
    crops = _crops(1)
    targets = {k: v[0] for k, v in cloak.Cloaker(surrogates).embed(_crops(2)[1:]).items()}
    common = dict(steps=20, lr=2.0, eps=8.0, dssim_budget=0.05, eot_samples=0, stop_cos=1.0, patience=20)
    sharp = cloak.Cloaker(surrogates, cloak.CloakParams(**common)).cloak(crops, targets)
    smooth = cloak.Cloaker(surrogates, cloak.CloakParams(delta_sigma=1.5, **common)).cloak(crops, targets)

    def high_freq(res):
        d = res.images[0] - crops[0].image
        return float(np.abs(np.diff(d, axis=0)).mean() + np.abs(np.diff(d, axis=1)).mean())

    assert np.abs(smooth.images[0] - crops[0].image).max() <= 8.0 + 1e-3
    assert high_freq(smooth) < 0.7 * high_freq(sharp)


def test_grad_norm_equalises_surrogate_contributions():
    class Scaled(nn.Module):
        def __init__(self, inner, k):
            super().__init__()
            self.inner, self.k = inner, k

        def forward(self, x):
            # same embedding as `inner`, but a much smaller input gradient (as PNA/TGR do)
            return self.inner(x * self.k + (x - x * self.k).detach())

    base = DummySurrogate(1)
    surrogates = {"strong": DummySurrogate(2), "weak": Scaled(base, 0.01)}
    crops = _crops(1)
    targets = {k: v[0] for k, v in cloak.Cloaker(surrogates).embed(_crops(2)[1:]).items()}
    common = dict(steps=25, lr=2.0, eps=10.0, dssim_budget=0.03, eot_samples=0, stop_cos=1.0, patience=25)
    plain = cloak.Cloaker(surrogates, cloak.CloakParams(**common)).cloak(crops, targets)
    normed = cloak.Cloaker(surrogates, cloak.CloakParams(grad_norm=True, **common)).cloak(crops, targets)
    # with normalisation the weak surrogate gets its full say and ends closer to the target
    assert normed.cos[0, 1] > plain.cos[0, 1]
    assert normed.dssim[0] <= 0.03 * 1.05 + 1e-6
    # the function itself: each surrogate's rescaled gradient has the mean of the raw norms
    delta = torch.zeros(1, 3, 40, 40, requires_grad=True)
    terms = torch.stack([(delta * 3).sum().view(1), (delta * 0.01).flatten().pow(2).sum().view(1) + 0.02 * delta.sum().view(1)], dim=1)
    penalty = 0.0 * delta.sum().view(1)
    g = cloak.Cloaker._normalised_grad(terms, penalty, delta, torch.ones(1))
    raw = [torch.autograd.grad(terms[:, j].sum(), delta, retain_graph=True)[0].norm() for j in range(2)]
    assert raw[0] > 100 * raw[1]
    assert abs(g.norm().item() - sum(r.item() for r in raw)) < 1e-4  # two parallel gradients, each rescaled to the mean


def test_grad_norm_matches_plain_gradient_for_one_surrogate():
    surrogates = {"a": DummySurrogate(1)}
    crops = _crops(2)
    targets = {k: v[0] for k, v in cloak.Cloaker(surrogates).embed(_crops(3)[2:]).items()}
    p = dict(steps=6, eot_samples=0, seed=1)
    r1 = cloak.Cloaker(surrogates, cloak.CloakParams(**p)).cloak(crops, targets)
    r2 = cloak.Cloaker(surrogates, cloak.CloakParams(grad_norm=True, **p)).cloak(crops, targets)
    assert np.allclose(r1.images[0], r2.images[0])
