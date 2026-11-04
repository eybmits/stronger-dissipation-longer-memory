"""Figure style of the paper.

Figures are drawn at their printed width (one column of the Quantum class) and
typeset by LaTeX in Helvetica with sans-serif mathematics. Panels carry a bold letter and a
bold title above the top-left corner. One colour stands for one jump family throughout:
warm colours for families without a protected subspace, cool colours for families with one.
"""
from __future__ import annotations

import os
import shutil

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import ticker

if shutil.which("latex") is None:
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

PT = 1 / 72.27
PAGE = 170.0 / 25.4                 # text width of the Quantum class, two columns (17 cm)
COLUMN = (PAGE - 20 * PT) / 2       # column width (column separation 20 pt)
DESIGN_PAGE, DESIGN_COLUMN = 510.0, 246.0   # widths in pt in which the layouts are written

INK, INK2, INK3, RULE, SHADE = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#d9d9d9", "#f2f2f2"

FAMILY = {
    "separate": "#c8553d",    # local loss
    "exchange": "#e09f3e",    # local loss with exchange
    "gainloss": "#8f2d56",    # local gain and loss
    "dephasing": "#7d8597",   # local dephasing
    "pair": "#2a9d8f",        # pair loss
    "signed": "#7b5ea7",      # one shared channel, mixed phases
    "groups": "#4f86c6",      # two shared channels
    "shared": "#16325c",      # one shared channel, equal phases
}
NAME = {
    "separate": "local", "exchange": "exchange", "gainloss": "gain/loss", "dephasing": "dephasing",
    "pair": "pairwise", "signed": "mixed phases", "groups": "two channels", "shared": "shared",
}
STRONG, GENTLE, ERASED = "#9c6644", "#2d6a4f", "#8c8c8c"
# one set of sizes for every figure: data lines, guide lines, markers, text inside panels
LW, LW_GUIDE, MS, TXT = 1.8, 1.0, 4.0, 7.4
# families without protected operators dashed, dephasing dotted, families with dark states solid
DASH = {"separate": (0, (4, 1.6)), "exchange": (0, (4, 1.6)), "gainloss": (0, (4, 1.6)),
        "dephasing": (0, (1.2, 1.4)), "pair": "-", "signed": "-", "groups": "-", "shared": "-"}

# white for no memory, dark for full memory
RECALL = mcolors.LinearSegmentedColormap.from_list(
    "recall", ["#ffffff", "#fbe7c6", "#f4a259", "#c8553d", "#6b2242", "#21102b"])
SHARE = mcolors.LinearSegmentedColormap.from_list(
    "share", ["#ffffff", "#d7e6f4", "#7fa7d4", "#3a6aa5", "#16325c", "#0b1730"])

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[scaled=0.95]{helvet}\renewcommand{\familydefault}{\sfdefault}"
                           r"\usepackage{amsmath}\usepackage{sansmath}\sansmath"
                           # matplotlib drops glyph 0 of the TeX symbol font (the minus) when it
                           # subsets Type 1 fonts; take the minus from the text font instead
                           r"\DeclareMathSymbol{-}{\mathbin}{operators}{123}",
    "font.family": "sans-serif",
    "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.2,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 7.2,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "axes.grid": False, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4, "xtick.minor.size": 1.9, "ytick.minor.size": 1.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.pad": 2.6, "ytick.major.pad": 2.6, "axes.labelpad": 2.4,
    "lines.linewidth": 1.8, "lines.markersize": 4.0, "lines.markeredgewidth": 0.9, "patch.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": RULE, "legend.fancybox": False,
    "legend.borderpad": 0.35, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
    "axes.unicode_minus": False, "pdf.compression": 9, "savefig.dpi": 300,
})


def tone(c, f):
    """Lighter (f > 0) or darker (f < 0) variant of a colour."""
    rgb = np.array(mcolors.to_rgb(c))
    return mcolors.to_hex(rgb + (1 - rgb) * f if f >= 0 else rgb * (1 + f))


def axes(fig, left, bottom, width, height):
    """Axes placed in points from the lower-left corner of the figure.

    Horizontal positions are given in the design width of the layout and rescaled to the
    printed width, so that text stays at its nominal size."""
    W, H = fig.get_size_inches() / PT
    sx = W / (DESIGN_PAGE if W > 300 else DESIGN_COLUMN)
    return fig.add_axes([left * sx / W, bottom / H, width * sx / W, height / H])


def panel(ax, letter, title="", dx=-22, dy=6):
    """Bold letter and bold title above the top-left corner of the axes (offsets in points)."""
    ax.annotate(rf"\textbf{{{letter}}}\hspace{{0.55em}}\textbf{{{title}}}", xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points", ha="left", va="bottom", fontsize=9)


def decades(ax, axis="x", labels=None):
    """Decade ticks labelled as powers of ten (or plain numbers), unlabelled minor ticks."""
    def fmt(v, _):
        if labels == "plain" or (labels is None and 0.1 <= v <= 100):
            return f"{v:g}"
        return rf"$10^{{{int(round(np.log10(v)))}}}$"
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        a.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        a.set_major_formatter(ticker.FuncFormatter(fmt))
        a.set_minor_formatter(ticker.NullFormatter())


def band(ax, x, rows, color, label=None, lw=None, alpha=0.15, marker=None, ls="-", **kw):
    """Mean over reservoirs with the range between the 16th and 84th percentiles."""
    rows = np.asarray(rows, float)
    mean = np.nanmean(rows, axis=0)
    lo, hi = np.nanpercentile(rows, 16, axis=0), np.nanpercentile(rows, 84, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, mean, color=color, lw=LW if lw is None else lw, label=label, marker=marker, mfc="white", mec=color,
            ls=ls, **kw)
    return mean


def lines(ax, x, rows, color, lw=1.3, thin=0.5, **kw):
    """One thin line per reservoir and their mean, for small sets of reservoirs."""
    rows = np.asarray(rows, float)
    for r in rows:
        ax.plot(x, r, color=tone(color, 0.45), lw=thin)
    mean = np.nanmean(rows, axis=0)
    ax.plot(x, mean, color=color, lw=lw, **kw)
    return mean


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
