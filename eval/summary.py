"""Print one markdown row per harness result file: label, mode, overrides, protection per evaluator,
DSSIM in the face box, tint, LPIPS, apparent-age shift, seconds per photo.  Usage: python eval/summary.py [eval/results/*.json]"""
import glob
import json
import os
import sys


def row(path):
    r = json.load(open(path))
    a, q, ev = r["args"], r["quality"], r["evaluators"]
    label = a.get("tag") or "-"
    if a.get("jpeg"):
        label += f" jpeg{a['jpeg']}"
    if a.get("clean_gallery"):
        label += " clean-gallery"
    overrides = " ".join(a.get("cloak_arg") or []) or "-"
    if a.get("target_strategy", "random") != "random":
        overrides += f" target={a['target_strategy']}"
    prot = " | ".join(f"{ev[k]['protection_rate']:.2f}" for k in ev)
    spp = q.get("seconds_per_photo")
    tint = " | ".join("-" if q.get(k) is None else f"{q[k]:.2f}"
                      for k in ("chroma_lf_rms_face", "luma_lf_rms_face", "lpips_face", "age_shift"))
    return (f"| {label} | {a['mode']} | {overrides} | {prot} | {q['dssim_face']:.4f} | {tint} | "
            f"{'-' if spp is None else f'{spp:.1f}'} |"), list(ev)


def main(paths):
    paths = paths or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*.json")))
    header = None
    for p in sorted(paths):
        line, evs = row(p)
        if header != evs:
            header = evs
            print("| label | mode | overrides | " + " | ".join(evs) + " | DSSIM face | chroma LF | luma LF | LPIPS | age shift | s/photo |")
            print("|---" * (len(evs) + 9) + "|")
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
