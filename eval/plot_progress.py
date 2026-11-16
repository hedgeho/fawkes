"""The README chart: protection against visible change for every released set of modes.

    python eval/plot_progress.py --out docs/img/progress.png

The numbers are the "Final modes" sections and rounds 17 and 19 of eval/RESULTS.md.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# label, mode, buffalo_l, antelopev2, adaface_vit_b, DSSIM face
POINTS = [
    ("Fawkes 1.0", "v1", 0.00, 0.00, None, 0.015),
    ("2026-09-12", "low", 0.74, 0.90, 0.20, 0.012),
    ("current", "low", 0.95, 1.00, 0.42, 0.0124),
    ("2026-09-12", "mid", 0.74, 0.84, 0.32, 0.012),
    ("2026-09-13", "mid", 0.80, 0.92, 0.48, 0.013),
    ("2026-09-15", "mid", 0.86, 0.94, 0.46, 0.013),
    ("current", "mid", 0.93, 0.94, 0.78, 0.0125),
    ("2026-09-12", "high", 0.82, 0.92, 0.52, 0.017),
    ("2026-09-13", "high", 0.86, 0.92, 0.70, 0.016),
    ("2026-09-15", "high", 0.92, 0.98, 0.64, 0.016),
    ("current", "high", 0.96, 0.99, 0.85, 0.0167),
    ("current", "subtle", 0.56, 0.68, 0.26, 0.0067),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"


def panel(ax, title, y_of):
    ax.set_facecolor(SURFACE)
    for mode in ("low", "mid", "high"):
        pts = [p for p in POINTS if p[1] == mode]
        ax.plot([p[5] for p in pts], [y_of(p) for p in pts], color=GREY, lw=1.5, zorder=1)
    for p in POINTS:
        y = y_of(p)
        if y is None:
            continue
        cur = p[0] == "current"
        color = ORANGE if p[1] == "v1" else BLUE if cur else GREY
        ax.scatter(p[5], y, s=70 if cur or p[1] == "v1" else 40, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3)
        if p[1] == "v1":
            ax.annotate("Fawkes 1.0", (p[5], y), xytext=(-8, 8), textcoords="offset points", ha="right",
                        color=INK, fontsize=10)
        elif cur:
            ax.annotate(p[1], (p[5], y), xytext=(8, -4), textcoords="offset points", color=INK, fontsize=11,
                        fontweight="bold")
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlim(0.004, 0.0195)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("visible change (DSSIM inside the face box)", color=INK2, fontsize=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    panel(axes[0], "ResNets (mean of buffalo_l and antelopev2)", lambda p: (p[2] + p[3]) / 2)
    panel(axes[1], "AdaFace ViT-B (the hardest evaluator)", lambda p: p[4])
    axes[0].set_ylabel("protection rate", color=INK2, fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="modes as they ship"),
        plt.Line2D([], [], marker="o", ls="-", color=GREY, ms=6, label="earlier versions, 2026-09-12 to 09-15"),
        plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="Fawkes 1.0, mid (ResNets only)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
