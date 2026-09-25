"""Cloak a folder of real (high-resolution) photos with several configurations and write side-by-side
face crops for looking at, plus LPIPS and the apparent-age shift per configuration.

    python eval/showcase.py --input data/me --out eval/work/showcase \
        --config "base|mid|" --config "near|mid|target=near" --config "floor|mid|eps_floor=0.3"

A config is label|mode|comma-separated name=value overrides; `target=near|far` uses the automatic
target, otherwise --target-dir is used. Output: <out>/<label>/<photo>_cloaked.png, and
<out>/sheets/<photo>_face<k>.jpg with the clean face and one column per configuration, and
<out>/phone/<label>/<photo>.jpg, the whole photo as a feed shows it (1080 px wide, JPEG 85).
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import image_quality, parse_cloak_args, perceptual_quality  # noqa: E402


def parse_config(text):
    label, mode, overrides = (text.split("|") + ["", ""])[:3]
    args = parse_cloak_args([o for o in overrides.split(",") if o])
    return label, mode or "mid", args


def face_boxes(detector, img, n=2, pad=0.35):
    """Up to `n` largest faces as square boxes padded by `pad` of their size."""
    out = []
    for f in detector.detect(img)[:n]:
        x0, y0, x1, y1 = f.bbox
        s = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        out.append(tuple(int(round(v)) for v in (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-dir", default=None)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--tile", type=int, default=512, help="side of each face tile in the sheets")
    args = ap.parse_args(argv)

    from fawkes.detect import Detector
    from fawkes.protection import Fawkes
    from fawkes.utils import load_image

    photos = sorted(f for f in os.listdir(args.input)
                    if "_cloaked" not in f and f.lower().endswith((".jpg", ".jpeg", ".png")))
    configs = [parse_config(c) for c in args.config]
    outputs = {}
    for label, mode, overrides in configs:
        d = os.path.join(args.out, label)
        os.makedirs(d, exist_ok=True)
        paths = []
        for f in photos:
            dst = os.path.join(d, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(dst):
                Image.fromarray(load_image(os.path.join(args.input, f)).astype(np.uint8)).save(dst)
            paths.append(dst)
        strategy = overrides.pop("target", None)
        protector = Fawkes(mode=mode, target_dir=None if strategy else args.target_dir,
                           target_strategy=strategy or "far", remember_targets=False, batch_size=8, **overrides)
        print(f"=== {label}: mode {mode} {overrides} target {strategy or args.target_dir}")
        protector.run_protection(paths)
        pairs = [(p, os.path.splitext(p)[0] + "_cloaked.png") for p in paths]
        q = image_quality(pairs)
        q.update(perceptual_quality(pairs))
        print(f"{label}: DSSIM face {q['dssim_face']:.4f} chroma LF {q['chroma_lf_rms_face']:.2f} "
              f"luma LF {q['luma_lf_rms_face']:.2f} LPIPS {q['lpips_face']} age shift {q['age_shift']}")
        outputs[label] = pairs
        del protector

    detector = Detector()
    sheets = os.path.join(args.out, "sheets")
    os.makedirs(sheets, exist_ok=True)
    first = configs[0][0]
    for j, (clean_p, _) in enumerate(outputs[first]):
        clean = np.asarray(Image.open(clean_p).convert("RGB"))
        for k, box in enumerate(face_boxes(detector, clean)):
            tiles = [("clean", Image.open(clean_p).convert("RGB"))]
            tiles += [(label, Image.open(outputs[label][j][1]).convert("RGB")) for label, _, _ in configs]
            sheet = Image.new("RGB", (args.tile * len(tiles), args.tile + 24), "white")
            draw = ImageDraw.Draw(sheet)
            for t, (label, im) in enumerate(tiles):
                sheet.paste(im.crop(box).resize((args.tile, args.tile), Image.LANCZOS), (t * args.tile, 24))
                draw.text((t * args.tile + 6, 4), label, fill="black")
            name = os.path.splitext(os.path.basename(clean_p))[0]
            sheet.save(os.path.join(sheets, f"{name}_face{k}.jpg"), quality=92)
    print(f"sheets in {sheets}")

    # what a feed shows: the whole photo, 1080 px wide, JPEG 85; flip between the folders to compare
    phone = os.path.join(args.out, "phone")
    for label, pairs in [("clean", [(c, c) for c, _ in outputs[first]])] + list(outputs.items()):
        d = os.path.join(phone, label)
        os.makedirs(d, exist_ok=True)
        for _, cloaked_p in pairs:
            im = Image.open(cloaked_p).convert("RGB")
            im = im.resize((1080, round(im.height * 1080 / im.width)), Image.LANCZOS)
            name = os.path.basename(cloaked_p).replace("_cloaked", "")
            im.save(os.path.join(d, os.path.splitext(name)[0] + ".jpg"), quality=85)
    print(f"phone-size photos in {phone}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
