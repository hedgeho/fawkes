"""The README illustration: a few "posted" photos per person cloaked in every mode, with the output of a
simple face classifier on each: the `faceid` package of the author's `face` project, checked out next to
this repository (buffalo_l embeddings, a reference centroid and a cosine threshold).

    python eval/illustrate.py --work eval/work/illus --out docs/img/modes.jpg

<work>/posted/<person>/ holds the photos the person publishes and <work>/test/<person>/ other, clean
photos of them (for the README: Barack Obama's photos in data/obama, 3 posted and 2 test, and the first
3 and the next 10 LFW photos of Serena Williams and Junichiro Koizumi). <work>/cloak/<person>/ is the
output of

    python eval/showcase.py --input <work>/posted/<person> --out <work>/cloak/<person> \
        --config "low|low|" --config "subtle|subtle|" --config "mid|mid|" --config "high|high|"

Every tile answers two questions with the classifier:

- "this photo": is the posted photo still recognised as the person by a classifier that learnt their
  face from the clean test photos? (what a cloak alone changes; Fawkes does not aim for this)
- "trained on these": a classifier whose reference is the three posted photos of the row, as a scraper
  would build it; how many of the clean test photos does it match? (the protection Fawkes aims for)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODES = ["clean", "low", "subtle", "mid", "high"]
PERSONS = ["Barack_Obama", "Serena_Williams", "Junichiro_Koizumi"]


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
    ap.add_argument("--tile", type=int, default=240)
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

    rows = {}  # (person, mode) -> (posted paths, scores of the posted photos, test scores)
    boxes = {}
    for person in PERSONS:
        tests = list_images(work / "test" / person)
        known = centroid([e for e in (embed(p) for p in tests) if e is not None])
        posted = list_images(work / "posted" / person)
        for mode in MODES:
            d = work / "cloak" / person / ("low" if mode == "clean" else mode)
            paths = [d / (p.stem + (".png" if mode == "clean" else "_cloaked.png")) for p in posted]
            this = score(known, paths)
            trained = centroid([e for e in (embed(p) for p in paths) if e is not None])
            rows[person, mode] = (paths, this, score(trained, tests))
            print(f"{person:18s} {mode:6s} this photo {' '.join(f'{s:.2f}' for s in this)}  "
                  f"trained on these: {(rows[person, mode][2] >= args.threshold).sum()}/{len(tests)} "
                  f"mean {rows[person, mode][2].mean():.2f}")
        _, box = max(faces(rows[person, "clean"][0][0]), key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))
        x0, y0, x1, y1 = box
        w, h = Image.open(rows[person, "clean"][0][0]).size
        s = min(max(x1 - x0, y1 - y0) * 1.7, w, h)  # a square around the face, kept inside the photo
        cx = min(max((x0 + x1) / 2, s / 2), w - s / 2)
        cy = min(max((y0 + y1) / 2 - 0.05 * s, s / 2), h - s / 2)
        boxes[person] = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)

    t, cap, head, lab = args.tile, 46, 34, 92
    f_head, f_cap, f_lab = font(19), font(15), font(18)
    sheet = Image.new("RGB", (lab + t * len(PERSONS), head + (t + cap) * len(MODES)), "white")
    draw = ImageDraw.Draw(sheet)
    green, red = (20, 120, 40), (190, 30, 30)
    for j, person in enumerate(PERSONS):
        draw.text((lab + j * t + t / 2, head / 2), person.replace("_", " "), fill="black", font=f_head, anchor="mm")
    for i, mode in enumerate(MODES):
        y = head + i * (t + cap)
        draw.text((lab / 2, y + t / 2), mode, fill="black", font=f_lab, anchor="mm")
        for j, person in enumerate(PERSONS):
            paths, this, tests = rows[person, mode]
            im = Image.open(paths[0]).convert("RGB")
            sheet.paste(im.resize((t, t), Image.LANCZOS, box=boxes[person]), (lab + j * t, y))
            x = lab + j * t + 6
            ok = this[0] >= args.threshold
            draw.text((x, y + t + 3), f"this photo: {'match' if ok else 'no match'} ({this[0]:.2f})",
                      fill=green if ok else red, font=f_cap)
            n = int((tests >= args.threshold).sum())
            draw.text((x, y + t + 23), f"trained on these: {n}/{len(tests)} found",
                      fill=green if n > len(tests) / 2 else red, font=f_cap)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
