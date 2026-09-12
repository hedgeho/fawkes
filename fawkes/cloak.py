"""Cloak generation: optimise a bounded perturbation on face crops so that every surrogate
recogniser sees the target identity, through the adversary's own alignment step.

Surrogates are torch modules taking (N,3,112,112) RGB in [0,1] (template-aligned) and returning
L2-normalised embeddings. Crops are `align.FaceCrop`s; the perturbation lives at crop resolution.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from fawkes.align import TEMPLATE_SIZE, estimate_norm, warp_to_template


@dataclass
class CloakParams:
    steps: int = 60
    lr: float = 2.0  # Adam step in [0, 255] pixel units
    eps: float = 12.0  # L-inf bound on the perturbation, [0, 255] units
    dssim_budget: float = 0.012  # mean DSSIM allowed on the crop
    dssim_weight: float = 100.0  # penalty per unit of DSSIM above the budget
    eot_samples: int = 2  # augmented views cycled between clean steps, 0 disables EOT
    kps_jitter: float = 2.0  # px, random landmark offsets (detector disagreement)
    blur_sigma_max: float = 1.5
    resize_min: float = 0.7
    jpeg_quality: tuple = (60, 95)
    stop_cos: float = 0.75  # stop a face once every surrogate is this close to the target
    patience: int = 10  # stop a face when its clean loss has not improved for this many clean steps
    seed: int = 0
    batch_size: int = 8


@dataclass
class CloakResult:
    images: list  # cloaked crops, (h,w,3) float32 [0,255], same order as the input
    cos: np.ndarray  # (N, n_models) final cosine to the target per surrogate, clean alignment
    dssim: np.ndarray  # (N,)
    steps: np.ndarray  # (N,) steps run per face
    seconds: float
    models: list = field(default_factory=list)


# ----------------------------------------------------------------------------- image metrics

def _gaussian_window(size=11, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, max_val=255.0, window=None):
    """Per-image SSIM of (N,3,H,W) tensors, gaussian window 11/1.5, as in Wang et al. 2004.

    Matches skimage's structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False) averaged over channels.
    """
    if window is None:
        window = _gaussian_window().to(x)
    c = x.shape[1]
    w = window.expand(c, 1, -1, -1)
    pad = window.shape[-1] // 2

    def filt(t):
        return F.conv2d(t, w, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x ** 2
    syy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def dssim(x, y, mask=None):
    """(1 - SSIM) / 2 per image; if `mask` (N,1,H,W) is given the SSIM map is averaged over it."""
    if mask is None:
        return (1 - ssim(x, y)) / 2
    window = _gaussian_window().to(x)
    c = x.shape[1]
    w = window.expand(c, 1, -1, -1)

    def filt(t):
        return F.conv2d(t, w, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x ** 2
    syy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    pad = window.shape[-1] // 2
    m = mask[:, :, pad:-pad, pad:-pad].expand_as(s)
    s = (s * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
    return (1 - s) / 2


# ----------------------------------------------------------------------------- EOT transforms

def gaussian_blur(x, sigma):
    if sigma <= 0.05:
        return x
    radius = max(1, int(math.ceil(3 * sigma)))
    ax = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, radius, radius), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def resize_round_trip(x, factor):
    """Downscale by `factor` and back up, both bilinear with antialiasing on the way down."""
    if factor >= 0.999:
        return x
    h, w = x.shape[-2:]
    small = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)


_Y_TABLE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_C_TABLE = torch.full((8, 8), 99.0)
_C_TABLE[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])


def _dct_matrix():
    n = 8
    m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            m[k, i] = math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= 1 / math.sqrt(2)
    return m * math.sqrt(2 / n)


def jpeg_approx(x, quality):
    """Differentiable JPEG approximation (DiffJPEG-style: YCbCr, 8x8 DCT, quantisation with
    straight-through rounding, no chroma subsampling). x in [0,1], (N,3,H,W), H and W multiples of 8."""
    n, c, h, w = x.shape
    assert h % 8 == 0 and w % 8 == 0, (h, w)
    dev = x.device
    x255 = x * 255.0
    r, g, b = x255[:, 0], x255[:, 1], x255[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    ycc = torch.stack([y, cb, cr], dim=1) - 128.0

    scale = 5000.0 / quality if quality < 50 else 200.0 - 2 * quality
    tables = torch.stack([_Y_TABLE, _C_TABLE, _C_TABLE]).to(dev)
    tables = torch.clamp(torch.floor((tables * scale + 50) / 100), 1, 255)

    d = _dct_matrix().to(dev)
    blocks = ycc.reshape(n, 3, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)  # (n,3,hb,wb,8,8)
    coeffs = d @ blocks @ d.T
    q = coeffs / tables[None, :, None, None]
    q = q + (torch.round(q) - q).detach()
    coeffs = q * tables[None, :, None, None]
    blocks = d.T @ coeffs @ d
    ycc = blocks.permute(0, 1, 2, 4, 3, 5).reshape(n, 3, h, w) + 128.0
    y, cb, cr = ycc[:, 0], ycc[:, 1] - 128, ycc[:, 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    out = torch.stack([r, g, b], dim=1) / 255.0
    return out.clamp(0, 1)


# ----------------------------------------------------------------------------- the cloaker

def _stack_crops(crops):
    """Pad crops to a common size (edge replicate) -> images (N,3,H,W) [0,255], mask (N,1,H,W)."""
    hs = [c.image.shape[0] for c in crops]
    ws = [c.image.shape[1] for c in crops]
    h, w = max(hs), max(ws)
    images = torch.zeros(len(crops), 3, h, w)
    mask = torch.zeros(len(crops), 1, h, w)
    for i, c in enumerate(crops):
        t = torch.from_numpy(np.ascontiguousarray(c.image)).permute(2, 0, 1)
        ch, cw = t.shape[1:]
        t = F.pad(t[None], (0, w - cw, 0, h - ch), mode='replicate')[0]
        images[i] = t
        mask[i, :, :ch, :cw] = 1
    return images, mask


class Cloaker:
    def __init__(self, surrogates, params=None, verbose=False):
        """surrogates: {key: nn.Module} as returned by fawkes.models.load_surrogate."""
        self.surrogates = dict(surrogates)
        self.params = params or CloakParams()
        self.verbose = verbose
        devices = {next(m.parameters()).device for m in self.surrogates.values() if any(True for _ in m.parameters())}
        self.device = devices.pop() if devices else torch.device("cpu")

    @property
    def keys(self):
        return list(self.surrogates)

    @torch.no_grad()
    def embed(self, crops):
        """Clean template embeddings per surrogate: {key: (N, d) float32 array}."""
        out = {k: [] for k in self.surrogates}
        p = self.params
        for start in range(0, len(crops), p.batch_size):
            batch = crops[start:start + p.batch_size]
            images, _ = _stack_crops(batch)
            images = images.to(self.device)
            m = np.stack([c.matrix for c in batch])
            aligned = warp_to_template(images / 255.0, m)
            for k, model in self.surrogates.items():
                out[k].append(model(aligned).cpu().numpy())
        return {k: np.concatenate(v) if v else np.zeros((0, 512), np.float32) for k, v in out.items()}

    def cloak(self, crops, targets):
        """Cloak every crop toward `targets` ({key: (d,) unit vector}). Returns a CloakResult."""
        p = self.params
        torch.manual_seed(p.seed)
        rng = np.random.RandomState(p.seed)
        t0 = time.time()
        order = np.argsort([c.image.shape[0] * c.image.shape[1] for c in crops])
        images = [None] * len(crops)
        cos = np.zeros((len(crops), len(self.surrogates)), np.float32)
        dss = np.zeros(len(crops), np.float32)
        steps = np.zeros(len(crops), np.int32)
        for start in range(0, len(crops), p.batch_size):
            idx = order[start:start + p.batch_size]
            batch = [crops[i] for i in idx]
            out, c, d, s = self._cloak_batch(batch, targets, rng)
            for j, i in enumerate(idx):
                images[i] = out[j]
                cos[i], dss[i], steps[i] = c[j], d[j], s[j]
        return CloakResult(images=images, cos=cos, dssim=dss, steps=steps,
                           seconds=time.time() - t0, models=self.keys)

    def _augment(self, aligned, rng):
        p = self.params
        x = aligned
        x = gaussian_blur(x, float(rng.uniform(0, p.blur_sigma_max)))
        x = resize_round_trip(x, float(rng.uniform(p.resize_min, 1.0)))
        x = jpeg_approx(x, float(rng.uniform(*p.jpeg_quality)))
        return x

    def _feature_loss(self, aligned, target_vecs):
        """(N,) mean over surrogates of 1 - cos, and (N, n_models) cosines."""
        cosines = []
        for k, model in self.surrogates.items():
            emb = model(aligned)
            cosines.append((emb * target_vecs[k][None]).sum(dim=1))
        cosines = torch.stack(cosines, dim=1)
        return (1 - cosines).mean(dim=1), cosines

    def _cloak_batch(self, crops, targets, rng):
        p = self.params
        n = len(crops)
        dev = self.device
        images, mask = _stack_crops(crops)
        images, mask = images.to(dev), mask.to(dev)
        target_vecs = {k: torch.as_tensor(np.asarray(targets[k], np.float32), device=dev) for k in self.surrogates}
        clean_m = np.stack([c.matrix for c in crops])
        delta = torch.zeros_like(images, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=p.lr)

        best_delta = torch.zeros_like(images)
        best_loss = torch.full((n,), float('inf'), device=dev)
        best_cos = torch.zeros(n, len(self.surrogates), device=dev)
        best_dssim = torch.zeros(n, device=dev)
        since_improved = torch.zeros(n, dtype=torch.long, device=dev)
        active = torch.ones(n, dtype=torch.bool, device=dev)
        steps_run = torch.zeros(n, dtype=torch.long, device=dev)

        # Stochastic EOT: each step takes one gradient pass through one view of the batch, cycling
        # through the clean alignment and `eot_samples` augmented views; bookkeeping and early
        # stopping use the clean steps only.
        views = 1 + p.eot_samples
        for step in range(p.steps):
            clean_step = step % views == 0
            x = (images + delta * mask).clamp(0, 255)
            if clean_step:
                aligned = warp_to_template(x / 255.0, clean_m)
            else:
                jitter = rng.uniform(-p.kps_jitter, p.kps_jitter, size=(n, 5, 2)).astype(np.float32)
                m = np.stack([estimate_norm(c.kps + jitter[i]) for i, c in enumerate(crops)])
                aligned = self._augment(warp_to_template(x / 255.0, m), rng)
            feat_loss, cosines = self._feature_loss(aligned, target_vecs)
            d = dssim(x, images, mask)
            penalty = p.dssim_weight * F.relu(d - p.dssim_budget)
            loss = ((feat_loss + penalty) * active.float()).sum()

            opt.zero_grad()
            loss.backward()
            delta.grad.mul_(active.float().view(-1, 1, 1, 1))
            opt.step()
            with torch.no_grad():
                delta.clamp_(-p.eps, p.eps)
                delta.mul_(mask)
                steps_run[active] = step + 1
                if clean_step:
                    # best-so-far bookkeeping on the clean alignment, within budget only
                    within = d.detach() <= p.dssim_budget * 1.05
                    improved = active & within & (feat_loss.detach() < best_loss)
                    best_loss = torch.where(improved, feat_loss.detach(), best_loss)
                    best_cos[improved] = cosines.detach()[improved]
                    best_dssim[improved] = d.detach()[improved]
                    # delta was just updated; keep the pre-update state that produced this loss
                    best_delta[improved] = (x - images).detach()[improved]
                    since_improved = torch.where(improved, torch.zeros_like(since_improved), since_improved + 1)
                    reached = within & (cosines.detach().min(dim=1).values >= p.stop_cos)
                    active = active & ~reached & (since_improved < p.patience)

            if self.verbose:
                print("step {:3d} {} loss {:.3f} cos {} dssim {}".format(
                    step + 1, "clean" if clean_step else "aug  ", float(feat_loss.detach().mean()),
                    np.round(cosines.detach().min(dim=1).values.cpu().numpy(), 3),
                    np.round(d.detach().cpu().numpy(), 4)))
            if not active.any():
                break

        never = torch.isinf(best_loss)
        if never.any():
            # no step landed inside the budget: return the last perturbation
            best_delta[never] = (delta.detach() * mask)[never]
            with torch.no_grad():
                x = (images + best_delta).clamp(0, 255)
                _, cosines = self._feature_loss(warp_to_template(x / 255.0, clean_m), target_vecs)
                best_cos[never] = cosines[never]
                best_dssim[never] = dssim(x, images, mask)[never]

        final = (images + best_delta).clamp(0, 255).cpu()
        out = []
        for i, c in enumerate(crops):
            h, w = c.image.shape[:2]
            out.append(final[i, :, :h, :w].permute(1, 2, 0).numpy().astype(np.float32))
        return out, best_cos.cpu().numpy(), best_dssim.cpu().numpy(), steps_run.cpu().numpy()
