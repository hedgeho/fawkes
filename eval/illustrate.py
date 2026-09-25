"""The README illustration: a few "posted" photos per person cloaked in every mode, with the output of a
simple face classifier on each: the `faceid` package of the author's `face` project, checked out next to
this repository (buffalo_l embeddings, a reference centroid and a cosine threshold).

    python eval/illustrate.py --work eval/work/illus --out docs/img/modes.jpg

<work>/posted/<person>/ holds the photos the person publishes and <work>/test/<person>/ other, clean
photos of them (for the README: Barack Obama's photos in data/obama, 3 posted and 2 test, and the first
3 and the next 10 LFW photos of Serena Williams, Junichiro Koizumi and Tony Blair). <work>/cloak/<person>/ is the
output of

    python eval/showcase.py --input <work>/posted/<person> --out <work>/cloak/<person> \
        --config "low|low|" --config "subtle|subtle|" --config "mid|mid|" --config "high|high|"

<work>/targets.json maps person -> mode -> the automatic target Fawkes chose (printed by showcase.py),
and <work>/targets/<target>/ holds that identity's LFW photos.

Under every tile:

- the person and the target: the posted photo's cosine similarity to each one's reference (the person's
  clean test photos, the target's LFW photos) and the probability a nearest-centroid classifier over
  every identity in the figure gives them: softmax(64 * cosine), 64 being the logit scale ArcFace-family
  recognisers such as buffalo_l are trained with.
- "trained on these": a reference built from the row's three posted photos, as a scraper would build
  it; how many of the clean test photos match it (the protection Fawkes aims for).

The row labels are the protection rates of the modes in the README's evaluation table.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODES = ["clean", "low", "subtle", "mid", "high"]
PERSONS = ["Barack_Obama", "Serena_Williams", "Junichiro_Koizumi", "Tony_Blair"]
SCALE = 64.0
# protection rate, README evaluation table: (buffalo_l, antelopev2) ResNets, (AdaFace ViT-B, LVFace-T)
# transformers; low is the 2026-09-12 row (random target, LVFace-T not measured)
PROTECTION = {"clean": ((0, 0), (0, 0)), "low": ((0.74, 0.90), (0.20, None)),
              "subtle": ((0.60, 0.66), (0.22, 0.58)), "mid": ((0.88, 0.88), (0.70, 0.86)),
              "high": ((0.96, 0.96), (0.78, 0.98))}


def short(name):
    first, *rest = name.split("_")
    return f"{first[0]}. {rest[-1]}"


def pair(v):
    return " / ".join("-" if x is None else f"{x:.2f}" for x in v)


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def centroid(embs):
    c = np.mean(embs, 0)
    return c / np.linalg.norm(c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="eval/work/illus")
    ap.add_argument("--out", default="docs/img/modes.jpg")
    ap.add_argument("--face-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "face"),
                    help="checkout of the face classifier")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="cosine threshold of the classifier (the value the face project uses)")
    ap.add_argument("--tile", type=int, default=250)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.face_dir))
    from faceid.common import get_app, list_images, load_image

    app = get_app()
    work = Path(args.work)

    def faces(path):
        img, scale = load_image(path)
        return [(f.normed_embedding.astype(np.float32), f.bbox / scale) for f in app.get(img)]

    def embed(path):
        """Embedding of the largest face, as faceid does per image; None if there is no face."""
        fs = faces(path)
        return max(fs, key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))[0] if fs else None

    def score(ref, paths):
        """faceid.classify per image: the best face's similarity to the reference centroid."""
        out = []
        for p in paths:
            s = max((float(e @ ref) for e, _ in faces(p)), default=-1.0)
            out.append(s)
        return np.array(out)

    targets = json.loads((work / "targets.json").read_text())
    gallery = {}  # identity -> reference centroid, for the nearest-centroid probabilities
    for person in PERSONS:
        gallery[person] = centroid([e for e in (embed(p) for p in list_images(work / "test" / person)) if e is not None])
    for name in sorted({t for m in targets.values() for t in m.values()}):
        gallery[name] = centroid([e for e in (embed(p) for p in list_images(work / "targets" / name)) if e is not None])
    names = list(gallery)
    refs = np.stack([gallery[n] for n in names])

    rows = {}  # (person, mode) -> (posted paths, target, cosines and probabilities of the first photo, test scores)
    boxes = {}
    for person in PERSONS:
        tests = list_images(work / "test" / person)
        posted = list_images(work / "posted" / person)
        for mode in MODES:
            d = work / "cloak" / person / ("low" if mode == "clean" else mode)
            paths = [d / (p.stem + (".png" if mode == "clean" else "_cloaked.png")) for p in posted]
            target = targets[person]["mid" if mode == "clean" else mode]
            cos = refs @ embed(paths[0])
            prob = np.exp(SCALE * (cos - cos.max()))
            prob /= prob.sum()
            i, k = names.index(person), names.index(target)
            trained = centroid([e for e in (embed(p) for p in paths) if e is not None])
            found = score(trained, tests)
            rows[person, mode] = (paths, target, (cos[i], prob[i], cos[k], prob[k]), found)
            print(f"{person:18s} {mode:6s} own {cos[i]:.2f} p {prob[i]:.3f}  target {target} {cos[k]:.2f} "
                  f"p {prob[k]:.3f}  top {names[int(np.argmax(cos))]}  trained on these: "
                  f"{(found >= args.threshold).sum()}/{len(tests)} mean {found.mean():.2f}")
        _, box = max(faces(rows[person, "clean"][0][0]), key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))
        x0, y0, x1, y1 = box
        w, h = Image.open(rows[person, "clean"][0][0]).size
        s = min(max(x1 - x0, y1 - y0) * 1.7, w, h)  # a square around the face, kept inside the photo
        cx = min(max((x0 + x1) / 2, s / 2), w - s / 2)
        cy = min(max((y0 + y1) / 2 - 0.05 * s, s / 2), h - s / 2)
        boxes[person] = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)

    t, cap, head, lab = args.tile, 64, 34, 170
    f_head, f_cap, f_lab, f_num = font(19), font(14), font(20), font(14)
    sheet = Image.new("RGB", (lab + t * len(PERSONS), head + (t + cap) * len(MODES)), "white")
    draw = ImageDraw.Draw(sheet)
    green, red, grey = (20, 120, 40), (190, 30, 30), (90, 90, 90)
    draw.text((8, head / 2), "mode, protection", fill="black", font=f_cap, anchor="lm")
    for j, person in enumerate(PERSONS):
        draw.text((lab + j * t + t / 2, head / 2), person.replace("_", " "), fill="black", font=f_head, anchor="mm")
    for i, mode in enumerate(MODES):
        y = head + i * (t + cap)
        resnet, vit = PROTECTION[mode]
        draw.text((10, y + t / 2 - 34), mode, fill="black", font=f_lab)
        draw.text((10, y + t / 2), f"ResNets {pair(resnet)}", fill=grey, font=f_num)
        draw.text((10, y + t / 2 + 20), f"ViTs {pair(vit)}", fill=grey, font=f_num)
        for j, person in enumerate(PERSONS):
            paths, target, (c_own, p_own, c_tgt, p_tgt), tests = rows[person, mode]
            im = Image.open(paths[0]).convert("RGB")
            sheet.paste(im.resize((t, t), Image.LANCZOS, box=boxes[person]), (lab + j * t, y))
            x = lab + j * t + 6
            draw.text((x, y + t + 3), f"{short(person)}: {c_own:.2f}, p {p_own:.0%}",
                      fill=green if p_own > 0.5 else red, font=f_cap)
            draw.text((x, y + t + 22), f"target {short(target)}: {c_tgt:.2f}, p {p_tgt:.0%}",
                      fill=grey, font=f_cap)
            n = int((tests >= args.threshold).sum())
            draw.text((x, y + t + 41), f"trained on these: {n}/{len(tests)} found",
                      fill=green if n > len(tests) / 2 else red, font=f_cap)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
