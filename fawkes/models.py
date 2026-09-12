"""Model zoo for Fawkes v2: surrogate recognisers (differentiable, torch) and held-out
evaluators (forward only, torch or onnxruntime).

Every model takes a 112x112 face aligned with the ArcFace five-point template
(``insightface.utils.face_align.norm_crop``) and returns a 512-d embedding. The wrappers
returned by :func:`load_surrogate` / :func:`load_evaluator` apply each model's own
preprocessing internally, so callers only ever deal with RGB pixels:

* surrogate: ``forward(x)`` with ``x`` float32 ``(N, 3, 112, 112)``, RGB, values in [0, 1];
  returns L2-normalised float32 ``(N, embed_dim)`` and is differentiable w.r.t. ``x``;
* evaluator: ``fn(imgs)`` with ``imgs`` uint8 ``(N, 112, 112, 3)`` RGB; returns
  L2-normalised float32 ``(N, embed_dim)``, no gradient.

Preprocessing facts, each verified by ``tests/test_models.py`` against the upstream
inference path (cosine > 0.999 on the same aligned crop):

==============  ===============================  ======  =====================
key             upstream                         order   normalisation
==============  ===============================  ======  =====================
arcface_r100    insightface arcface_torch        RGB     (x/255 - 0.5) / 0.5
adaface_ir101   CVLface (mk-minchul)             RGB     (x/255 - 0.5) / 0.5
lvface_b        bytedance/LVFace                 RGB     (x/255 - 0.5) / 0.5
w600k_r50       insightface buffalo_l (ONNX)     RGB     (x - 127.5) / 127.5
glintr100       insightface antelopev2 (ONNX)    RGB     (x - 127.5) / 127.5
adaface_vit_b   CVLface (mk-minchul)             RGB     (x/255 - 0.5) / 0.5
==============  ===============================  ======  =====================

Weights are downloaded with ``huggingface_hub`` into :func:`model_dir` (``fawkes/model/``,
gitignored; override with ``FAWKES_MODEL_DIR``) and checked against a pinned sha256.
Prefetch everything with ``python -m fawkes.models download``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ModelSpec", "SURROGATES", "EVALUATORS", "model_dir", "weight_path", "download",
    "sha256sum", "load_surrogate", "load_evaluator",
]

INPUT_SIZE = 112


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    filename: str
    sha256: str | None
    kind: str        # 'torch' | 'onnx'
    arch: str        # builder name in _ARCHS (torch) or 'onnx'
    embed_dim: int
    licence: str
    note: str


_INSIGHTFACE_LICENCE = (
    "insightface: code MIT; pretrained models non-commercial research only "
    "(https://github.com/deepinsight/insightface#license)"
)
_CVLFACE_LICENCE = (
    "CVLface code MIT; model card says 'follow the license of the training dataset' "
    "(WebFace4M/12M: non-commercial research)"
)

SURROGATES: dict[str, ModelSpec] = {
    "arcface_r100": ModelSpec(
        key="arcface_r100",
        repo_id="camenduru/show",
        filename="models/arcface/ms1mv3_arcface_r100_fp16.pth",
        sha256="a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b",
        kind="torch", arch="iresnet100", embed_dim=512,
        licence=_INSIGHTFACE_LICENCE,
        note=("insightface arcface_torch 'ms1mv3_arcface_r100_fp16' backbone.pth (IResNet-100, MS1MV3, "
              "ArcFace margin). Official distribution is OneDrive/Baidu; this community mirror's sha256 "
              "matches two other independent mirrors (guym-models/ArcFace, marcelohaps/arcface-torch). "
              "The Glint360K R100 checkpoint the plan named ('glint360k_cosface_r100_fp16_0.1', "
              "sha256 5f631718...) is NOT used: it is the very network inside antelopev2 glintr100.onnx "
              "(identical embeddings), which would make the glintr100 evaluator not held out."),
    ),
    "adaface_ir101": ModelSpec(
        key="adaface_ir101",
        repo_id="minchul/cvlface_adaface_ir101_webface12m",
        filename="model.safetensors",
        sha256="2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56",
        kind="torch", arch="adaface_ir101", embed_dim=512,
        licence=_CVLFACE_LICENCE,
        note="AdaFace IR-101 trained on WebFace12M (CVLface release). RGB input per CVLface config.",
    ),
    "lvface_b": ModelSpec(
        key="lvface_b",
        repo_id="bytedance-research/LVFace",
        filename="LVFace-B_Glint360K/LVFace-B_Glint360K.pt",
        sha256="143ded01bf5794aafff91fcd1e3f3408bfbf6b890fc8482274f1c97a82fa7542",
        kind="torch", arch="vit_b_p9", embed_dim=512,
        licence="LVFace code MIT; weights 'non-commercial research purposes only' (LICENSE_CODE.txt / README)",
        note="LVFace-B (ViT-B, patch 9, Glint360K), arcface_torch network name 'vit_b_dp005_mask_005'.",
    ),
}

EVALUATORS: dict[str, ModelSpec] = {
    "w600k_r50": ModelSpec(
        key="w600k_r50",
        repo_id="public-data/insightface",
        filename="models/buffalo_l/w600k_r50.onnx",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        kind="onnx", arch="onnx", embed_dim=512,
        licence=_INSIGHTFACE_LICENCE,
        note="insightface buffalo_l recogniser (IResNet-50, WebFace600K). Same file as the buffalo_l.zip release.",
    ),
    "glintr100": ModelSpec(
        key="glintr100",
        repo_id="DIAMONIK7777/antelopev2",
        filename="glintr100.onnx",
        sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        kind="onnx", arch="onnx", embed_dim=512,
        licence=_INSIGHTFACE_LICENCE,
        note=("insightface antelopev2 recogniser: the ONNX export of arcface_torch "
              "'glint360k_cosface_r100_fp16_0.1' (IResNet-100, Glint360K, CosFace margin). "
              "public-data/insightface has no antelopev2; this mirror's sha256 equals the file inside "
              "the official antelopev2.zip GitHub release and ~/.insightface/models/antelopev2."),
    ),
    "adaface_vit_b": ModelSpec(
        key="adaface_vit_b",
        repo_id="minchul/cvlface_adaface_vit_base_webface4m",
        filename="model.safetensors",
        sha256="5fafd6b7d599a3ede5fac5bd1d01ad05e9e93e89b39b7687d4a3bc93ff2aebc0",
        kind="torch", arch="vit_b_p8", embed_dim=512,
        licence=_CVLFACE_LICENCE,
        note="AdaFace ViT-B (patch 8) trained on WebFace4M (CVLface release, plain ViT, not the KP-RPE variant).",
    ),
}

_ALL: dict[str, ModelSpec] = {**SURROGATES, **EVALUATORS}


def spec(key: str) -> ModelSpec:
    try:
        return _ALL[key]
    except KeyError:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(_ALL)}") from None


# --------------------------------------------------------------------------- files

def model_dir() -> Path:
    """Directory holding downloaded weights (``fawkes/model/`` unless FAWKES_MODEL_DIR is set)."""
    env = os.environ.get("FAWKES_MODEL_DIR")
    return Path(env).expanduser() if env else Path(__file__).resolve().parent / "model"


def weight_path(key: str) -> Path:
    """Where ``download(key)`` puts the file (may not exist yet)."""
    s = spec(key)
    return model_dir() / s.key / s.filename


def sha256sum(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(key: str, verify: bool | None = None) -> Path:
    """Fetch the weights for ``key`` into ``model_dir()`` and return the local path.

    A file that is already present is not re-downloaded. The sha256 is checked after a fresh
    download (or always if ``verify=True``); a mismatch deletes the file and raises.
    """
    from huggingface_hub import hf_hub_download

    s = spec(key)
    dest = weight_path(key)
    fresh = not dest.exists()
    if fresh:
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = Path(hf_hub_download(s.repo_id, s.filename, local_dir=str(model_dir() / s.key)))
        if got.resolve() != dest.resolve():
            raise RuntimeError(f"hf_hub_download put {s.filename} at {got}, expected {dest}")
    if s.sha256 and (verify or (verify is None and fresh)):
        digest = sha256sum(dest)
        if digest != s.sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"sha256 mismatch for {key}: got {digest}, expected {s.sha256}; file removed")
    return dest


# --------------------------------------------------------------------------- torch builders

def _strip_prefix(state: dict, prefix: str) -> dict:
    out = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if len(out) != len(state):
        bad = [k for k in state if not k.startswith(prefix)][:5]
        raise RuntimeError(f"state_dict keys without prefix {prefix!r}: {bad}")
    return out


def _load_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    return _flush_denormals(state)


def _flush_denormals(state: dict) -> dict:
    """Zero weights whose magnitude is below the smallest normal float32.

    The CVLface AdaFace IR-101 checkpoint holds about 206k such values; they make every conv
    produce denormal activations, which x86 handles about 100x slower, so its forward pass
    took 3.7x longer than the same-size arcface_torch IResNet-100. Values under 1.2e-38
    contribute nothing measurable (the fixture embedding is unchanged to 1e-7 in cosine).
    """
    tiny = float(np.finfo(np.float32).tiny)
    for v in state.values():
        if v.is_floating_point():
            v[(v.abs() < tiny) & (v != 0)] = 0
    return state


def _build_iresnet100(path: Path) -> nn.Module:
    from .arch.iresnet import iresnet100
    net = iresnet100()
    net.load_state_dict(_load_state(path), strict=True)
    return net


def _build_adaface_ir101(path: Path) -> nn.Module:
    from .arch.adaface_ir import IR_101
    net = IR_101(input_size=(INPUT_SIZE, INPUT_SIZE), output_dim=512)
    net.load_state_dict(_strip_prefix(_load_state(path), "model.net."), strict=True)
    return net


def _build_vit_b_p9(path: Path) -> nn.Module:
    # arcface_torch / LVFace network name "vit_b_dp005_mask_005"
    from .arch.vit import VisionTransformer
    net = VisionTransformer(img_size=INPUT_SIZE, patch_size=9, num_classes=512, embed_dim=512, depth=24,
                            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05)
    net.load_state_dict(_load_state(path), strict=True)
    return net


def _build_vit_b_p8(path: Path) -> nn.Module:
    # CVLface models/vit config name "base"
    from .arch.vit import VisionTransformer
    net = VisionTransformer(img_size=INPUT_SIZE, patch_size=8, num_classes=512, embed_dim=512, depth=24,
                            mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.0)
    net.load_state_dict(_strip_prefix(_load_state(path), "model.net."), strict=True)
    return net


_ARCHS: dict[str, Callable[[Path], nn.Module]] = {
    "iresnet100": _build_iresnet100,
    "adaface_ir101": _build_adaface_ir101,
    "vit_b_p9": _build_vit_b_p9,
    "vit_b_p8": _build_vit_b_p8,
}


class Recogniser(nn.Module):
    """Wraps a backbone so that it takes RGB [0,1] tensors and returns unit-norm embeddings."""

    def __init__(self, net: nn.Module, spec: ModelSpec, bgr: bool = False,
                 mean: float = 0.5, std: float = 0.5):
        super().__init__()
        self.net = net
        self.spec = spec
        self.bgr = bgr
        self.mean = mean
        self.std = std
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # frozen: never leave eval mode (BatchNorm/Dropout)
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2:] != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"{self.spec.key}: expected (N,3,{INPUT_SIZE},{INPUT_SIZE}), got {tuple(x.shape)}")
        if self.bgr:
            x = x.flip(1)
        # .contiguous(): a permuted NHWC view would otherwise propagate channels_last strides
        # into the backbone, whose Flatten uses .view().
        x = ((x - self.mean) / self.std).contiguous()
        return F.normalize(self.net(x).to(x.dtype), dim=1)


def default_device() -> str:
    """$FAWKES_DEVICE if set, else 'cuda' when available, else 'cpu'."""
    env = os.environ.get("FAWKES_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_torch(key: str, device=None) -> Recogniser:
    s = spec(key)
    if s.kind != "torch":
        raise ValueError(f"{key} is a {s.kind} model, not a torch model")
    net = _ARCHS[s.arch](download(key))
    return Recogniser(net, s).to(device or default_device())


def load_surrogate(key: str, device=None) -> nn.Module:
    """Differentiable recogniser in eval mode with frozen parameters, on `device` (default_device())."""
    if key not in SURROGATES:
        raise KeyError(f"{key!r} is not a surrogate; choose from {sorted(SURROGATES)}")
    return _load_torch(key, device)


# --------------------------------------------------------------------------- evaluators

def _onnx_evaluator(key: str) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    path = download(key)
    so = ort.SessionOptions()
    so.log_severity_level = 3  # the insightface graphs declare batch 1 on the output; batching still works
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    # insightface ArcFaceONNX: (x - 127.5) / 127.5, RGB (blobFromImages swapRB=True on BGR), NCHW.

    def embed(imgs: np.ndarray) -> np.ndarray:
        imgs = _check_uint8(imgs, key)
        blob = (imgs.astype(np.float32) - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))
        feats = sess.run([out_name], {inp.name: blob})[0].astype(np.float32)
        return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)

    return embed


def _torch_evaluator(key: str, device=None) -> Callable[[np.ndarray], np.ndarray]:
    model = _load_torch(key, device)
    dev = next(model.parameters()).device

    def embed(imgs: np.ndarray) -> np.ndarray:
        imgs = _check_uint8(imgs, key)
        x = torch.from_numpy(imgs.astype(np.float32)).permute(0, 3, 1, 2).div_(255.0).to(dev)
        with torch.no_grad():
            return model(x).cpu().numpy()

    return embed


def _check_uint8(imgs: np.ndarray, key: str) -> np.ndarray:
    imgs = np.asarray(imgs)
    if imgs.ndim == 3:
        imgs = imgs[None]
    if imgs.dtype != np.uint8 or imgs.ndim != 4 or imgs.shape[1:] != (INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError(f"{key}: expected uint8 (N,{INPUT_SIZE},{INPUT_SIZE},3) RGB, got {imgs.dtype} {imgs.shape}")
    return imgs


def load_evaluator(key: str, device=None) -> Callable[[np.ndarray], np.ndarray]:
    """Forward-only embedder: uint8 (N,112,112,3) RGB -> L2-normalised float32 (N, embed_dim).

    ONNX evaluators always run on the CPU; torch ones on `device` (default_device())."""
    if key not in EVALUATORS:
        raise KeyError(f"{key!r} is not an evaluator; choose from {sorted(EVALUATORS)}")
    s = EVALUATORS[key]
    return _onnx_evaluator(key) if s.kind == "onnx" else _torch_evaluator(key, device)


# --------------------------------------------------------------------------- CLI

def _cmd_download(keys: list[str], verify: bool) -> int:
    rc = 0
    for key in keys:
        s = spec(key)
        try:
            path = download(key, verify=verify)
        except Exception as exc:  # keep going so one bad mirror does not block the rest
            print(f"{key:14s} FAILED: {exc}", file=sys.stderr)
            rc = 1
            continue
        size = path.stat().st_size / 2**20
        digest = sha256sum(path)
        status = "ok" if s.sha256 in (None, digest) else "SHA256 MISMATCH"
        print(f"{key:14s} {size:8.1f} MiB  sha256={digest}  {status}  {path}")
    return rc


def _cmd_bench(keys: list[str], threads: int, batch_sizes: list[int], repeats: int) -> int:
    torch.set_num_threads(threads)
    print(f"threads={threads}  input {INPUT_SIZE}x{INPUT_SIZE}  (times in seconds, best of {repeats})")
    print(f"{'key':14s} {'params':>10s} {'file MiB':>9s} " + " ".join(f"{'fwd@'+str(b):>9s} {'fwd+bwd@'+str(b):>12s}" for b in batch_sizes))
    for key in keys:
        s = spec(key)
        row = f"{key:14s} "
        if s.kind == "onnx":
            fn = load_evaluator(key)
            row += f"{'-':>10s} {weight_path(key).stat().st_size / 2**20:9.1f} "
            for b in batch_sizes:
                x = np.random.randint(0, 256, (b, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                fn(x)
                t = min(_timed(lambda: fn(x)) for _ in range(repeats))
                row += f"{t:9.3f} {'-':>12s} "
            print(row, flush=True)
            continue
        model = _load_torch(key)
        dev = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        row += f"{n_params:10d} {weight_path(key).stat().st_size / 2**20:9.1f} "
        for b in batch_sizes:
            x = torch.rand(b, 3, INPUT_SIZE, INPUT_SIZE, device=dev)
            with torch.no_grad():
                model(x)
            t_fwd = min(_timed(lambda: _fwd(model, x)) for _ in range(repeats))
            xg = x.clone().requires_grad_(True)
            model(xg).sum().backward()
            t_fb = min(_timed(lambda: _fwd_bwd(model, xg)) for _ in range(repeats))
            row += f"{t_fwd:9.3f} {t_fb:12.3f} "
        print(row, flush=True)
    return 0


def _fwd(model, x):
    with torch.no_grad():
        model(x)


def _fwd_bwd(model, xg):
    xg.grad = None
    model(xg).sum().backward()


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fawkes.models",
                                     description="Manage Fawkes surrogate/evaluator weights.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dl = sub.add_parser("download", help="download weights (default: every surrogate and evaluator)")
    p_dl.add_argument("keys", nargs="*", help=f"model keys; known: {', '.join(_ALL)}")
    p_dl.add_argument("--verify", action="store_true", help="re-hash files that are already present")
    p_ls = sub.add_parser("list", help="print the registries")
    p_b = sub.add_parser("bench", help="time forward and forward+backward passes on CPU")
    p_b.add_argument("keys", nargs="*")
    p_b.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p_b.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p_b.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for group, reg in (("surrogates", SURROGATES), ("evaluators", EVALUATORS)):
            print(f"[{group}]")
            for s in reg.values():
                present = "present" if weight_path(s.key).exists() else "missing"
                print(f"  {s.key:14s} {s.kind:5s} {s.arch:14s} {s.repo_id}/{s.filename}  [{present}]")
                print(f"      licence: {s.licence}")
                print(f"      note: {s.note}")
        return 0
    keys = args.keys or list(_ALL)
    for k in keys:
        spec(k)  # fail early on typos
    if args.cmd == "download":
        return _cmd_download(keys, args.verify)
    return _cmd_bench(keys, args.threads, args.batch, args.repeats)


if __name__ == "__main__":
    sys.exit(main())
