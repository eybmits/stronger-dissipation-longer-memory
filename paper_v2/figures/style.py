"""Figure style of the paper.

Every data figure is one column wide (Quantum class, 17 cm text width, two columns) and is built on one grid, in TeX
points: full-width panels are 186 x 76 pt, half-width panels 74 x 76 pt; the left spines of the two columns sit at
38 and 150 pt and every row ends at 224 pt; rows are 40 pt apart. Panel letters sit at the left edge of their
column and titles share one baseline 5.5 pt above the axes; axis labels are aligned within each column and row.
Text is typeset by LaTeX in Helvetica with sans-serif mathematics: panel titles 8.6 pt bold, axis labels 8.2 pt,
tick labels 7.6 pt and everything inside the panels 7.4 pt, as in the TikZ figure 1. One colour stands for one jump
family throughout: warm colours for families without a protected subspace, cool colours for families with one.
"""
from __future__ import annotations

import os
import shutil

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import ticker

if shutil.which("latex") is None:
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

PT = 1 / 72.27
PAGE = 170.0 / 25.4                 # text width of the Quantum class, two columns (17 cm)
COLUMN = (PAGE - 20 * PT) / 2       # column width (column separation 20 pt)

# the grid, in points
L, R = 38.0, 224.0                  # left spine of the first column, right edge of every row
HALF = 74.0                         # width of a half-column panel
X2 = R - HALF                       # left spine of the second column
H = 76.0                            # height of every panel
TOP, TOP_PLAIN, GAP, BOTTOM = 15.0, 7.0, 40.0, 25.0
RISE = 5.5                          # baseline of the panel titles above the axes

# the same colours as figures/fig1_concept.tex
INK, INK2, INK3, RULE, SHADE = "#1a1a1a", "#3a3a3a", "#6f6f6f", "#d9d9d9", "#f2f2f2"
FAMILY = {
    "separate": "#c8553d",    # local loss
    "exchange": "#c9852a",    # local loss with exchange
    "gainloss": "#8f2d56",    # local gain and loss
    "dephasing": "#6b7385",   # local dephasing
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
# one set of sizes for every figure: data lines, guide lines, markers, text inside panels, panel titles
LW, LW_GUIDE, MS, TXT, TITLE = 1.8, 1.0, 4.0, 7.4, 8.6
# families without protected operators dashed, dephasing dotted, families with dark states solid
DASH = {"separate": (0, (4, 1.6)), "exchange": (0, (4, 1.6)), "gainloss": (0, (4, 1.6)),
        "dephasing": (0, (1.2, 1.4)), "pair": "-", "signed": "-", "groups": "-", "shared": "-"}

# white for no memory, dark for full memory
RECALL = mcolors.LinearSegmentedColormap.from_list(
    "recall", ["#ffffff", "#fbe7c6", "#f4a259", "#c8553d", "#6b2242", "#21102b"])

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[scaled=0.95]{helvet}\renewcommand{\familydefault}{\sfdefault}"
                           r"\usepackage{amsmath}\usepackage{sansmath}\sansmath"
                           # matplotlib drops glyph 0 of the TeX symbol font (the minus) when it
                           # subsets Type 1 fonts; take the minus from the text font instead
                           r"\DeclareMathSymbol{-}{\mathbin}{operators}{123}",
    "font.family": "sans-serif",
    "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.2,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": TXT, "legend.title_fontsize": TXT,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "axes.grid": False, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4, "xtick.minor.size": 1.9, "ytick.minor.size": 1.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.pad": 2.6, "ytick.major.pad": 2.6, "axes.labelpad": 2.4,
    "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": 0.9, "patch.linewidth": 0.6,
    "legend.frameon": False, "legend.borderpad": 0.2, "legend.borderaxespad": 0.6, "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.25, "legend.columnspacing": 1.2,
    "axes.unicode_minus": False, "pdf.compression": 9, "savefig.dpi": 300,
})


def tone(c, f):
    """Lighter (f > 0) or darker (f < 0) variant of a colour."""
    rgb = np.array(mcolors.to_rgb(c))
    return mcolors.to_hex(rgb + (1 - rgb) * f if f >= 0 else rgb * (1 + f))


# ------------------------------------------------------------------------------------------------------- grid
def figure(rows, titles=True, below=0.0):
    """A one-column figure with the given number of panel rows (and `below` points under them, for a legend)."""
    top = TOP if titles else TOP_PLAIN
    height = top + rows * H + (rows - 1) * GAP + BOTTOM + below
    fig = plt.figure(figsize=(COLUMN, height * PT))
    fig.grid_top = height - top
    return fig


def place(fig, row, col=None, left=None, right=None):
    """Axes of the panel in `row` (0 at the top): full width for col=None, else column 0 or 1."""
    W, Hf = fig.get_size_inches() / PT
    x0 = X2 if col == 1 else L
    x1 = L + HALF if col == 0 else R
    x0 = x0 if left is None else left
    x1 = x1 if right is None else right
    y1 = fig.grid_top - row * (H + GAP)
    return fig.add_axes([x0 / W, (y1 - H) / Hf, (x1 - x0) / W, H / Hf])


def panel(ax, letter, title, col=0):
    """Bold letter at the left edge of the column and bold title, on one baseline above the axes."""
    fig = ax.figure
    W, Hf = fig.get_size_inches() / PT
    x = 0.0 if col == 0 else X2 - L
    y = ax.get_position().y1 * Hf + RISE
    fig.text(x / W, y / Hf, rf"\textbf{{{letter}}}\hspace{{0.5em}}\textbf{{{title}}}", ha="left", va="baseline",
             fontsize=TITLE)


def align(fig, columns=(), rows=()):
    """Put the y labels of each group of axes (a column) and the x labels of each group (a row) at one distance
    from the axes: the distance the widest or deepest tick labels of the group need, plus the label pad."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = mpl.rcParams["axes.labelpad"] * fig.dpi / 72
    for group in columns:
        need = 0.0
        for ax in group:
            box = ax.get_window_extent(r)
            ext = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
            need = max([need] + [box.x0 - e.x0 for e in ext])
        for ax in group:
            ax.yaxis.set_label_coords(-(need + pad) / ax.get_window_extent(r).width, 0.5)
    for group in rows:
        need = 0.0
        for ax in group:
            box = ax.get_window_extent(r)
            ext = [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
            need = max([need] + [box.y0 - e.y0 for e in ext])
        for ax in group:
            ax.xaxis.set_label_coords(0.5, -(need + pad) / ax.get_window_extent(r).height)


# ------------------------------------------------------------------------------------------------------ ticks
def decades(ax, axis="x", labels="plain", every=1):
    """Decade ticks with unlabelled minor ticks; labels as plain numbers or as powers of ten, on every
    `every`-th decade."""
    def fmt(v, _):
        e = int(round(np.log10(v)))
        if e % every:
            return ""
        return f"{v:g}" if labels == "plain" else rf"$10^{{{e}}}$"
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        a.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        a.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        a.set_major_formatter(ticker.FuncFormatter(fmt))
        a.set_minor_formatter(ticker.NullFormatter())


def log_on_linear(ax, axis, lo, hi, labels):
    """Decade ticks for an axis that shows log10 of a quantity on a linear scale (heat maps)."""
    a = getattr(ax, axis + "axis")
    majors = [e for e in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= e <= hi]
    a.set_major_locator(ticker.FixedLocator(majors))
    a.set_major_formatter(ticker.FixedFormatter(
        [f"{10.0 ** e:g}" if labels == "plain" else rf"$10^{{{e}}}$" for e in majors]))
    minors = [e + np.log10(k) for e in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for k in range(2, 10)]
    a.set_minor_locator(ticker.FixedLocator([m for m in minors if lo <= m <= hi]))
    a.set_minor_formatter(ticker.NullFormatter())


# ------------------------------------------------------------------------------------------------------- data
def band(ax, x, rows, color, label=None, lw=None, alpha=0.15, marker=None, ls="-", **kw):
    """Mean over reservoirs with the range between the 16th and 84th percentiles."""
    rows = np.asarray(rows, float)
    mean = np.nanmean(rows, axis=0)
    lo, hi = np.nanpercentile(rows, 16, axis=0), np.nanpercentile(rows, 84, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, mean, color=color, lw=LW if lw is None else lw, label=label, marker=marker, mfc="white", mec=color,
            ls=ls, **kw)
    return mean


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
