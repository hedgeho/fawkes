"""Face alignment: the ArcFace 5-point template, a differentiable warp into it, and crop/paste helpers.

Coordinates are (x, y) pixel positions with pixel centres at integer coordinates, the OpenCV
convention, so that `warp_to_template` reproduces `cv2.warpAffine` on the same matrix.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

TEMPLATE_SIZE = 112
# left eye, right eye, nose, left mouth corner, right mouth corner, for a 112x112 crop
# (insightface/utils/face_align.py, `arcface_dst`).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(src, dst):
    """Least-squares similarity (rotation, uniform scale, translation) mapping src -> dst, as 2x3.

    Umeyama's closed form, the same estimator skimage's SimilarityTransform uses.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = (s * d).sum() / var_src
    t = dst_mean - scale * rot @ src_mean
    m = np.empty((2, 3), dtype=np.float32)
    m[:, :2] = scale * rot
    m[:, 2] = t
    return m


def estimate_norm(kps, image_size=TEMPLATE_SIZE):
    """2x3 matrix taking the 5 landmarks `kps` (5,2) onto the ArcFace template of `image_size`."""
    kps = np.asarray(kps, dtype=np.float32)
    assert kps.shape == (5, 2), kps.shape
    return similarity_transform(kps, ARCFACE_DST * (image_size / TEMPLATE_SIZE))


def norm_crop(img, kps, image_size=TEMPLATE_SIZE):
    """Reference (non-differentiable) aligned crop, identical to insightface's norm_crop."""
    m = estimate_norm(kps, image_size)
    return cv2.warpAffine(img, m, (image_size, image_size), borderValue=0.0)


def apply_affine(m, points):
    """Apply a 2x3 matrix to (K,2) points."""
    points = np.asarray(points, dtype=np.float32)
    return points @ m[:, :2].T + m[:, 2]


def compose(m2, m1):
    """2x3 matrix equal to applying m1 then m2."""
    h1 = np.vstack([m1, [0, 0, 1]])
    h2 = np.vstack([m2, [0, 0, 1]])
    return (h2 @ h1)[:2].astype(np.float32)


def _theta_from_affine(m, in_hw, out_hw):
    """affine_grid theta (2x3, output-normalised -> input-normalised) for a pixel-space 2x3 map m.

    `m` maps input pixel coordinates to output pixel coordinates (cv2.warpAffine semantics);
    normalised coordinates follow align_corners=False, i.e. pixel centre p sits at 2(p+0.5)/size-1.
    """
    h_in, w_in = in_hw
    h_out, w_out = out_hw
    m_h = np.vstack([np.asarray(m, dtype=np.float64), [0, 0, 1]])
    m_inv = np.linalg.inv(m_h)
    # output normalised -> output pixel
    out_norm_to_px = np.array([[w_out / 2, 0, w_out / 2 - 0.5],
                               [0, h_out / 2, h_out / 2 - 0.5],
                               [0, 0, 1]])
    # input pixel -> input normalised
    in_px_to_norm = np.array([[2 / w_in, 0, 1 / w_in - 1],
                              [0, 2 / h_in, 1 / h_in - 1],
                              [0, 0, 1]])
    theta = in_px_to_norm @ m_inv @ out_norm_to_px
    return theta[:2]


def warp_to_template(x, m, image_size=TEMPLATE_SIZE):
    """Differentiably warp a batch of crops into the template.

    x: (N,3,H,W) float tensor; m: (N,2,3) array or tensor of pixel-space maps crop -> template
    (as returned by estimate_norm on crop-space landmarks). Returns (N,3,image_size,image_size).
    Matches cv2.warpAffine(..., flags=INTER_LINEAR, borderValue=0) to within interpolation rounding.
    """
    n, _, h, w = x.shape
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    if m.ndim == 2:
        m = np.broadcast_to(m, (n, 2, 3))
    theta = np.stack([_theta_from_affine(m[i], (h, w), (image_size, image_size)) for i in range(n)])
    theta = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta, (n, 3, image_size, image_size), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


@dataclass
class FaceCrop:
    """A working-resolution crop of one face and everything needed to warp it and paste it back."""
    box: tuple  # (x0, y0, x1, y1) in photo pixels, exclusive upper bounds
    scale: float  # working pixels per photo pixel (<= 1)
    image: np.ndarray  # (h, w, 3) float32 RGB in [0, 255], working resolution
    kps: np.ndarray  # (5, 2) landmarks in working-crop coordinates

    @property
    def matrix(self):
        """2x3 map from working-crop pixels to the template."""
        return estimate_norm(self.kps)


def template_extent(m, pad=0.25, image_size=TEMPLATE_SIZE):
    """Axis-aligned bounding box (x0, y0, x1, y1) in source pixels of the region the template
    samples, padded by `pad` of the template size on every side (room for landmark jitter and blur).
    `m` maps source pixels to the template."""
    m_h = np.vstack([np.asarray(m, dtype=np.float64), [0, 0, 1]])
    inv = np.linalg.inv(m_h)[:2]
    lo, hi = -pad * image_size, (1 + pad) * image_size
    corners = np.array([[lo, lo], [hi, lo], [lo, hi], [hi, hi]], dtype=np.float64)
    src = corners @ inv[:, :2].T + inv[:, 2]
    return src[:, 0].min(), src[:, 1].min(), src[:, 0].max(), src[:, 1].max()


def make_crop(photo, bbox, kps, max_side=448):
    """Cut a face region out of `photo` (HxWx3, RGB, any float/uint8 in [0,255]).

    The region is the union of the detector box and the (padded) area the aligned template samples,
    clipped to the photo, so that warping the crop is equivalent to warping the whole photo.
    If its longer side exceeds `max_side` the crop is downscaled so the perturbation is optimised
    at a resolution close to what recognisers will resample it to; paste_back upsamples it again.
    """
    h, w = photo.shape[:2]
    kps = np.asarray(kps, dtype=np.float32)
    ex0, ey0, ex1, ey1 = template_extent(estimate_norm(kps))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x0 = int(max(0, np.floor(min(x1, ex0))))
    y0 = int(max(0, np.floor(min(y1, ey0))))
    x3 = int(min(w, np.ceil(max(x2, ex1))))
    y3 = int(min(h, np.ceil(max(y2, ey1))))
    region = np.ascontiguousarray(photo[y0:y3, x0:x3], dtype=np.float32)
    scale = min(1.0, max_side / max(region.shape[0], region.shape[1]))
    if scale < 1.0:
        new_w = max(1, int(round(region.shape[1] * scale)))
        new_h = max(1, int(round(region.shape[0] * scale)))
        sx, sy = new_w / region.shape[1], new_h / region.shape[0]
        region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        sx = sy = 1.0
    crop_kps = (kps - [x0, y0]) * [sx, sy]
    return FaceCrop(box=(x0, y0, x3, y3), scale=float(min(sx, sy)), image=region, kps=crop_kps)


def template_crop(image):
    """Treat an already-aligned image as a face crop (for --no-align inputs)."""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape[:2]
    kps = ARCFACE_DST * [w / TEMPLATE_SIZE, h / TEMPLATE_SIZE]
    return FaceCrop(box=(0, 0, w, h), scale=1.0, image=image, kps=kps)


def paste_back(photo, crop, cloaked):
    """Return a float32 copy of `photo` with the perturbation (cloaked - crop.image) added in place.

    The perturbation is resized back to photo resolution if the crop was downscaled.
    """
    out = np.array(photo, dtype=np.float32, copy=True)
    x0, y0, x1, y1 = crop.box
    delta = np.asarray(cloaked, dtype=np.float32) - crop.image
    target_hw = (y1 - y0, x1 - x0)
    if delta.shape[:2] != target_hw:
        delta = cv2.resize(delta, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    out[y0:y1, x0:x1] += delta
    np.clip(out, 0, 255, out=out)
    return out
