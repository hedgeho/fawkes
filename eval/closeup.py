"""The README close-up: one photo clean and cloaked in every mode, at three scales: the whole face at feed
size, a crop at 100 percent (one image pixel per screen pixel), and the change itself, amplified.

    python eval/closeup.py --dir eval/work/illus/cloak/Barack_Obama \
        --name obama_official_portrait_of_barack_obama --out docs/img/closeup.jpg

<dir>/<mode>/<name>.png is the photo as the cloaker read it and <dir>/<mode>/<name>_cloaked.png its
output (eval/showcase.py writes both). --face and --crop are x0,y0,x1,y1 boxes in image pixels.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODES = ["low", "subtle", "mid", "high"]


def box(s):
    return tuple(int(v) for v in s.split(","))


def font(size):
    for p in ["/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face", type=box, default=(440, 180, 1110, 1100))
    ap.add_argument("--crop", type=box, default=(580, 500, 940, 820))
    ap.add_argument("--feed", type=int, default=240, help="width of the feed-size face")
    ap.add_argument("--gain", type=float, default=4.0, help="amplification of the change")
    a = ap.parse_args()

    clean = Image.open(os.path.join(a.dir, MODES[0], a.name + ".png")).convert("RGB")
    cols = [("clean", clean, None)]
    for m in MODES:
        cl = Image.open(os.path.join(a.dir, m, a.name + "_cloaked.png")).convert("RGB")
        cols.append((m, cl, np.asarray(cl, np.float32) - np.asarray(clean, np.float32)))

    cw = a.crop[2] - a.crop[0]
    ch = a.crop[3] - a.crop[1]
    fw, fh = a.face[2] - a.face[0], a.face[3] - a.face[1]
    feed_h = round(a.feed * fh / fw)
    gap, head, side = 8, 48, 170
    rows = [("feed size", feed_h), ("100% crop", ch), (f"change x{a.gain:g}", ch)]
    W = side + len(cols) * (cw + gap)
    H = head + sum(h + gap for _, h in rows)
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    f_head, f_row = font(32), font(26)

    for i, (label, img, diff) in enumerate(cols):
        x = side + i * (cw + gap)
        d.text((x + cw // 2, head // 2), label, fill="black", font=f_head, anchor="mm")
        y = head
        face = img.crop(a.face).resize((a.feed, feed_h), Image.LANCZOS)
        out.paste(face, (x + (cw - a.feed) // 2, y))
        y += feed_h + gap
        out.paste(img.crop(a.crop), (x, y))
        y += ch + gap
        if diff is None:
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=(128, 128, 128))
        else:
            c = diff[a.crop[1]:a.crop[3], a.crop[0]:a.crop[2]]
            out.paste(Image.fromarray(np.clip(128 + a.gain * c, 0, 255).astype(np.uint8)), (x, y))

    y = head
    for label, h in rows:
        d.text((10, y + h // 2), label, fill="black", font=f_row, anchor="lm")
        y += h + gap
    out.save(a.out, quality=92)


if __name__ == "__main__":
    main()
