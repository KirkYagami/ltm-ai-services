"""Regenerates the figures in "01 - Feature Engineering - From First Principles.md".

Run from any directory: python path/to/feature_engineering_figures.py
Writes fe_*.png next to this script. Uses the same data and split as listings_demo.py.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from listings_demo import RAW, add_features, logistic, make_pipeline

HERE = Path(__file__).resolve().parent

# Light-surface palette (blue/orange pair validated for CVD and contrast).
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c3c2b7"
CLASS_LABEL = {0: "y = 0: not rented within 7 days", 1: "y = 1: rented within 7 days"}
CLASS_COLOR = {0: BLUE, 1: ORANGE}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(HERE / name, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def load_dev():
    df = pd.read_csv(HERE / "listings.csv")
    X, y = df[RAW], df["left_fast"]
    X_dev, _, y_dev, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    return X_dev, y_dev  # the final test rows are never needed for figures


def fig_scaling_neighbours(X_dev):
    """Same listings, same query: the nearest neighbour depends on the units."""
    d = X_dev[["rent", "distance"]].to_numpy()
    z = StandardScaler().fit_transform(d)
    for q in range(len(d)):
        if not (25_000 < d[q, 0] < 60_000 and 8 < d[q, 1] < 18):
            continue
        raw = np.linalg.norm(d - d[q], axis=1)
        std = np.linalg.norm(z - z[q], axis=1)
        raw[q] = std[q] = np.inf
        a, b = raw.argmin(), std.argmin()
        # want: the raw pick is far away in km (rent barely differs); the standardized pick is close on both
        if abs(d[a, 1] - d[q, 1]) > 12 and abs(d[b, 0] - d[q, 0]) > 2_000 and abs(d[b, 1] - d[q, 1]) < 1.5:
            break
    else:
        raise RuntimeError("no illustrative query found")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    panels = [
        (axes[0], d, "Raw units (₹, km)", a, "rent (₹)"),
        (axes[1], z, "Standardized (training std devs)", b, "rent (standardized)"),
    ]
    for ax, pts, title, own_pick, xlabel in panels:
        gap_rent = abs(d[own_pick, 0] - d[q, 0])
        gap_km = abs(d[own_pick, 1] - d[q, 1])
        xlabel += f"\nline to its nearest neighbour: Δrent = ₹{gap_rent:,.0f}, Δdistance = {gap_km:.1f} km"
        style(ax, grid="both")
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=AXIS, linewidths=0)
        ax.plot(*zip(pts[q], pts[own_pick]), color=INK_2, linewidth=1.2, zorder=2)
        ax.scatter(*pts[a], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(*pts[b], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.scatter(*pts[q], s=120, marker="D", facecolor="none", edgecolor=INK, linewidth=1.8, zorder=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("distance to centre (km)" if pts is d else "distance (standardized)")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    handles = [
        Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.8, markersize=8, label="query listing"),
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label="nearest in raw units"),
        Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=8, label="nearest after standardizing"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fe_scaling_neighbours.png")


def fig_fscore_separation(X_dev, y_dev):
    """Between-class separation versus within-class spread, for a strong and a weak feature."""
    feats = add_features(X_dev)
    scores, _ = f_classif(feats, y_dev)
    fscore = dict(zip(feats.columns, scores))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=False)
    for ax, col, xlabel in [(axes[0], "rent_per_area", "rent per square foot (₹ / month)"),
                            (axes[1], "area", "area (square feet)")]:
        style(ax)
        x = feats[col]
        bins = np.linspace(x.min(), x.max(), 28)
        for c in (0, 1):
            xc = x[y_dev == c]
            ax.hist(xc, bins=bins, histtype="stepfilled", color=CLASS_COLOR[c], alpha=0.35, linewidth=0)
            ax.hist(xc, bins=bins, histtype="step", color=CLASS_COLOR[c], linewidth=1.4)
            ax.axvline(xc.mean(), color=CLASS_COLOR[c], linestyle="--", linewidth=1.6)
        m0, m1 = x[y_dev == 0].mean(), x[y_dev == 1].mean()
        top = ax.get_ylim()[1]
        ax.text(min(m0, m1), top * 0.97, f"mean {min(m0, m1):,.1f} ", ha="right", va="top", fontsize=8.5, color=INK_2)
        ax.text(max(m0, m1), top * 0.97, f" mean {max(m0, m1):,.1f}", ha="left", va="top", fontsize=8.5, color=INK_2)
        f_text = f"{fscore[col]:,.1f}" if fscore[col] >= 10 else f"{fscore[col]:.2f}"
        ax.set_title(f"{col}: F = {f_text}", loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("number of development listings")
    handles = [Patch(facecolor=CLASS_COLOR[c], alpha=0.6, label=CLASS_LABEL[c]) for c in (0, 1)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fe_fscore_separation.png")


def fig_xor():
    """XOR: each column alone has identical class means, yet the pair determines the target."""
    rng = np.random.default_rng(0)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    pts = np.repeat(corners, 30, axis=0) + rng.normal(0, 0.045, (120, 2))
    y = np.repeat(labels, 30)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    style(ax, grid="both")
    for c in (0, 1):
        ax.scatter(*pts[y == c].T, s=30, color=CLASS_COLOR[c], edgecolor=SURFACE, linewidth=0.8)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("The pair decides the class", loc="left")

    style(bx)
    means = {c: [corners[labels == c, j].mean() for j in (0, 1)] for c in (0, 1)}
    width = 0.22
    for c, offset in [(0, -width / 2 - 0.02), (1, width / 2 + 0.02)]:
        bx.bar(np.arange(2) + offset, means[c], width=width, color=CLASS_COLOR[c])
        for j in (0, 1):
            bx.text(j + offset, means[c][j] + 0.02, f"{means[c][j]:.1f}", ha="center", fontsize=9, color=INK_2)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["$x_1$", "$x_2$"])
    bx.set_ylim(0, 1)
    bx.set_ylabel("mean of the column")
    bx.set_title("Each column alone: identical class means", loc="left")
    handles = [Patch(facecolor=CLASS_COLOR[c], label=f"class {c} (y = {c})") for c in (0, 1)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fe_xor.png")


def fig_cv_layout():
    """Where each row goes: one held-out test block, and five CV folds inside the rest."""
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    h, gap = 0.62, 0.38
    rows = ["All 800 listings"] + [f"CV fold {k}" for k in range(1, 6)]

    def block(x0, x1, row, color, text, text_color=INK):
        y0 = len(rows) - 1 - row
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=color, edgecolor=SURFACE, linewidth=2))
        ax.text((x0 + x1) / 2, y0 + h / 2, text, ha="center", va="center", fontsize=8.5, color=text_color)

    block(0, 640, 0, "#9ec5f4", "development data: 640 rows (all choices are made here)")
    block(640, 800, 0, NEUTRAL, "held-out test: 160 rows")
    for k in range(5):
        v0, v1 = k * 128, (k + 1) * 128
        if v0 > 0:
            block(0, v0, k + 1, BLUE, "train", "#ffffff")
        block(v0, v1, k + 1, ORANGE, "validate")
        if v1 < 640:
            block(v1, 640, k + 1, BLUE, "train", "#ffffff")
        block(640, 800, k + 1, GRID, "untouched", INK_2)
    for i, name in enumerate(rows):
        ax.text(-14, len(rows) - 1 - i + h / 2, name, ha="right", va="center", fontsize=9.5, color=INK_2)
    ax.set_xlim(-160, 800)
    ax.set_ylim(-0.1, len(rows) - gap)
    save(fig, "fe_cv_layout.png")


def fig_cv_results(X_dev, y_dev):
    """The 15 validation-fold AUCs behind each row of the results table."""
    candidates = {
        "Dummy (class balance only)": DummyClassifier(strategy="prior"),
        "Raw 4 columns, scaled": make_pipeline(),
        "+ 2 ratio columns, scaled": make_pipeline(engineered=True),
        "+ SelectKBest (keep 3)": make_pipeline(engineered=True, selector=SelectKBest(f_classif, k=3)),
        "+ RFE (keep 3)": make_pipeline(engineered=True,
                                        selector=RFE(logistic(), n_features_to_select=3, step=1)),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    splits = list(cv.split(X_dev, y_dev))
    scores = {name: cross_validate(p, X_dev, y_dev, cv=splits, scoring="roc_auc",
                                   error_score="raise", n_jobs=1)["test_score"]
              for name, p in candidates.items()}

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    style(ax, grid="x")
    ax.spines["left"].set_visible(False)
    offsets = np.tile([-0.09, 0.09, 0.0], 5)  # small fixed vertical stagger so dots do not hide each other
    for i, (name, s) in enumerate(scores.items()):
        y0 = len(scores) - 1 - i
        ax.scatter(s, y0 + offsets, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1,
                   alpha=0.9, zorder=3)
        ax.plot([s.mean()] * 2, [y0 - 0.28, y0 + 0.28], color=INK, linewidth=2.2, zorder=4)
        ax.text(0.995, y0, f"mean {s.mean():.4f}", va="center", ha="right", fontsize=9, color=INK_2)
        print(f"{name:30s} mean AUC {s.mean():.4f}  sd {s.std(ddof=1):.4f}")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(list(scores)[::-1])
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.6, len(scores) - 0.4)
    ax.set_xlabel("ROC AUC on validation folds (higher is better)")
    handles = [Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="one validation fold (15 per pipeline)"),
               Line2D([], [], color=INK, linewidth=2.2, label="mean of the 15 folds")]
    ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    save(fig, "fe_cv_results.png")


if __name__ == "__main__":
    X_dev, y_dev = load_dev()
    fig_scaling_neighbours(X_dev)
    fig_fscore_separation(X_dev, y_dev)
    fig_xor()
    fig_cv_layout()
    fig_cv_results(X_dev, y_dev)
