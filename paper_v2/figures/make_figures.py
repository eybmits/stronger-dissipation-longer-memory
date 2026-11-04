#!/usr/bin/env python3
"""Figures 2-6 from paper_v2/data/snapshot.json (written by export_snapshot.py).

Every figure is one column wide and uses the sizes of style.py: data lines st.LW, guide lines st.LW_GUIDE,
markers st.MS and text inside panels st.TXT. Panels span the column (196 pt in design units) or half of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

import style as st

HERE = Path(__file__).resolve().parent
SNAP = json.loads((HERE.parent / "data" / "snapshot.json").read_text())
ORDER = ("separate", "exchange", "gainloss", "dephasing", "pair", "signed", "groups", "shared")
FAR_MAX = 4096.0                      # largest loss rate shown for the slow modes
SCALED = (0, (3.2, 1.1, 0.9, 1.1))    # mixed phases with the input amplitude scaled down


def families():
    return [f for f in ORDER if f in SNAP["memory"]]


def stack(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def logticks(ax, axis, values, labels):
    a = getattr(ax, axis + "axis")
    a.set_major_locator(ticker.FixedLocator(values))
    a.set_major_formatter(ticker.FixedFormatter(labels))
    a.set_minor_locator(ticker.NullLocator())


def text(ax, x, y, s, color, **kw):
    kw.setdefault("fontsize", st.TXT)
    ax.text(x, y, s, color=color, **kw)


# ----------------------------------------------------------------------------------------------- Figure 2
def fig2():
    """Recall at weak and strong loss, memory against loss, slow modes; one column, panels stacked."""
    fig = plt.figure(figsize=(st.COLUMN, 382 * st.PT))
    a = st.axes(fig, 42, 304, 196, 62)
    b = st.axes(fig, 42, 182, 196, 78)
    c = st.axes(fig, 42, 60, 196, 78)

    tau = np.arange(1, 301)
    for fam, label_at in (("separate", (2.25, 0.06)), ("shared", (30, 0.6))):
        rec = SNAP["memory"][fam]
        for g, f in ((0.25, 0.55), (64.0, 0.0)):
            a.plot(tau, rec["curve_mean"][rec["curve_gamma"].index(g)], color=st.tone(st.FAMILY[fam], f))
        text(a, *label_at, st.NAME[fam] + r", $\gamma=64$", st.FAMILY[fam])
    text(a, 12, 0.8, r"both, $\gamma=1/4$", st.INK3)
    a.set_xscale("log")
    logticks(a, "x", [1, 10, 100], ["1", "10", "100"])
    a.set_xlim(1, 300)
    a.set_ylim(0, 1.06)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Delay $\tau$")
    a.set_ylabel(r"Recall $C_\tau$")
    st.panel(a, "a", "Recall at weak and strong loss", dx=-34)

    handles = []
    far = SNAP["review"].get("far", {})
    for fam in families():
        rec = SNAP["memory"][fam]
        x = np.array(rec["gamma"])
        # beyond gamma = 64 the recall of one and two shared channels reaches past the 300 delays summed
        inside = x <= (256 if fam in ("signed", "pair") else 64)
        st.band(b, x[inside], stack(rec["recall"])[:, inside], st.FAMILY[fam], ls=st.DASH[fam])
        xf, rows = x, stack(rec["forget"])
        if fam in far:
            sel = np.array(far[fam]["gamma"]) <= FAR_MAX
            xf = np.r_[x, np.array(far[fam]["gamma"])[sel]]
            rows = np.hstack([rows, stack(far[fam]["forget"])[:, sel]])
        st.band(c, xf, rows, st.FAMILY[fam], ls=st.DASH[fam])
        handles.append(Line2D([], [], color=st.FAMILY[fam], lw=st.LW, ls=st.DASH[fam], label=st.NAME[fam]))
    # mixed phases with the input amplitude scaled as min(1, gamma^(-1/2))
    sc = SNAP["review"]["scaled"]["1"]
    xs = np.array(sc["gamma"])
    for ax, key in ((b, "recall"), (c, "forget")):
        st.band(ax, xs, stack(sc[key]), st.FAMILY["signed"], ls=SCALED, marker="o", ms=st.MS, alpha=0.12)
    b.axhline(45, color=st.INK3, lw=st.LW_GUIDE, ls=(0, (4, 2)))
    text(b, 0.125, 46.2, "45 features", st.INK3, va="bottom")
    b.annotate(r"scaled input", xy=(100, 34.8), xytext=(9, 40), color=st.FAMILY["signed"], fontsize=st.TXT,
               ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=st.FAMILY["signed"], lw=0.8,
                                                         shrinkA=0.5, shrinkB=2.5))
    for ax, top in ((b, 256), (c, FAR_MAX)):
        ax.set_xscale("log")
        st.decades(ax, "x", labels="plain")
        ax.set_xlim(0.1, top)
        ax.set_xlabel(r"Loss rate $\gamma$")
    b.set_ylim(0, 52)
    b.set_yticks([0, 15, 30, 45])
    b.set_ylabel(r"Memory $\sum_\tau C_\tau$")
    c.set_yscale("log")
    st.decades(c, "y")
    c.set_ylim(0.03, 3e4)
    gg = np.geomspace(100, FAR_MAX, 10)
    c.plot(gg, 1.6 * gg, ls=(0, (1, 1.6)), color=st.FAMILY["shared"], lw=st.LW_GUIDE)
    text(c, 150, 900, r"$\propto\gamma$", st.FAMILY["shared"], ha="right")
    gl = np.geomspace(4, 200, 10)
    c.plot(gl, 0.9 / gl, ls=(0, (1, 1.6)), color=st.FAMILY["separate"], lw=st.LW_GUIDE)
    text(c, 5.0, 0.06, r"$\propto 1/\gamma$", st.FAMILY["separate"], ha="right")
    c.set_ylabel(r"Time to forget $t_{\mathrm f}$")
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=2.0, frameon=False, columnspacing=1.2,
               labelspacing=0.3, bbox_to_anchor=(0.54, 0.0))
    st.panel(b, "b", "Memory against loss", dx=-34)
    st.panel(c, "c", "Slow modes", dx=-34)
    st.save(fig, HERE / "fig2_families.pdf")


# ----------------------------------------------------------------------------------------------- Figure 3
def fig3():
    """Protected operators, the averaged step, the dark-state model and the erasure test; two by two."""
    fig = plt.figure(figsize=(st.COLUMN, 292 * st.PT))
    side = 80 * st.COLUMN / st.PT / st.DESIGN_COLUMN      # printed width of panel b, to make it square
    a = st.axes(fig, 72, 174, 62, 100)
    b = st.axes(fig, 156, 274 - side, 80, side)
    c = st.axes(fig, 42, 40, 82, 86)
    d = st.axes(fig, 156, 40, 80, 86)

    groups = ((r"no dark state", ("separate", "exchange", "gainloss")),
              (r"populations only", ("dephasing",)),
              (r"dark subspace", ("pair", "groups", "signed", "shared")))
    yy, rows, heads = 0.0, [], []
    for title, fams in groups:
        heads.append((yy, title))
        yy += 0.95
        for f in fams:
            rows.append((yy, f))
            yy += 1.0
        yy += 0.3
    a.set_xscale("log")
    for k, (yh, title) in enumerate(heads):
        a.axhline(yh - 0.48, color=st.RULE, lw=0.6) if k else None
        text(a, 0.8, yh, rf"\textit{{{title}}}", st.INK3, fontsize=6.6, va="center", ha="left")
    for yi, f in rows:
        v, col = SNAP["kernel"][f], st.FAMILY[f]
        a.plot([0.75, v], [yi, yi], color=st.tone(col, 0.45), lw=2.2, solid_capstyle="round", zorder=1)
        a.plot(v, yi, "o", ms=5.2, mfc="white" if f == "dephasing" else col, mec=col, mew=1.1, zorder=2)
        text(a, v * 1.7, yi, f"${v}$", st.INK2, fontsize=6.8, va="center", ha="left")
    a.set_xlim(0.6, 2000)
    logticks(a, "x", [1, 10, 100, 1000], ["1", "10", "100", r"$10^3$"])
    a.set_yticks([yi for yi, _ in rows])
    a.set_yticklabels([st.NAME[f] for _, f in rows])
    a.tick_params(axis="y", length=0)
    a.set_ylim(yy - 0.2, -1.0)
    a.set_xlabel(r"Protected operators")
    st.panel(a, "a", "What loss leaves alone", dx=-66)

    th = np.linspace(0, 2 * np.pi, 400)
    b.plot(np.cos(th), np.sin(th), color=st.INK3, lw=st.LW_GUIDE)
    maps = SNAP["spectrum"]["maps"]
    for key, col in (("shared_1", st.tone(st.FAMILY["shared"], 0.62)), ("shared_8", st.tone(st.FAMILY["shared"], 0.32)),
                     ("shared_64", st.FAMILY["shared"]), ("separate_8", st.FAMILY["separate"])):
        b.plot(maps[key]["re"], maps[key]["im"], "o", ms=2.6, mfc=col, mec="none", alpha=0.95)
    b.set_xlim(-1.4, 1.4)
    b.set_ylim(-1.4, 1.4)
    b.set_aspect("equal", adjustable="box")
    b.set_xticks([-1, 0, 1])
    b.set_yticks([-1, 0, 1])
    b.set_xlabel(r"Re $\lambda$")
    b.set_ylabel(r"Im $\lambda$", labelpad=0)
    text(b, -1.32, -1.32, r"local, $\gamma=8$", st.FAMILY["separate"], ha="left", va="bottom")
    text(b, -1.32, 1.32, "shared", st.FAMILY["shared"], ha="left", va="top")
    text(b, 1.32, 1.32, r"$\gamma=1,8,64$", st.FAMILY["shared"], ha="right", va="top")
    st.panel(b, "b", "One averaged step", dx=-28)

    red = SNAP["memory"]["shared"]["reduced"]
    tau = np.arange(1, 301)
    for gkey, col in (("4", st.tone(st.FAMILY["shared"], 0.5)), ("64", st.FAMILY["shared"])):
        full, model = np.array(red[gkey]["full"]), np.array(red[gkey]["reduced"])
        c.plot(tau, full, color=col)
        pick = np.unique(np.round(np.geomspace(1, 300, 16)).astype(int)) - 1
        c.plot(tau[pick], model[pick], "o", ms=st.MS, mfc="white", mec=col)
    c.set_xscale("log")
    logticks(c, "x", [1, 10, 100], ["1", "10", "100"])
    c.set_xlim(1, 300)
    c.set_ylim(0, 1.5)
    c.set_yticks([0, 0.5, 1])
    c.set_xlabel(r"Delay $\tau$")
    c.set_ylabel(r"Recall $C_\tau$")
    text(c, 1.6, 0.18, r"$\gamma=4$", st.tone(st.FAMILY["shared"], 0.5))
    text(c, 30, 0.6, r"$\gamma=64$", st.FAMILY["shared"])
    c.legend(handles=[Line2D([], [], color=st.INK2, label="full reservoir"),
                      Line2D([], [], color=st.INK2, ls="none", marker="o", ms=st.MS, mfc="white",
                             label="dark states only")], loc="upper right", handlelength=1.6)
    st.panel(c, "c", "Dark-state model", dx=-34)

    curves = SNAP["removal"]["16"]["curves"]
    tau = np.arange(1, 101)
    spec = (("intact", st.FAMILY["shared"], "-", "intact"),
            ("rotated", st.tone(st.FAMILY["shared"], 0.45), (0, (3.2, 1.4)), "rotated copy erased"),
            ("removed", st.ERASED, (0, (1.0, 1.0)), "dark states erased"))
    handles = []
    for field, col, ls, lab in spec:
        st.band(d, tau, np.array(curves[field]), col, ls=ls, alpha=0.14)
        handles.append(Line2D([], [], color=col, lw=st.LW, ls=ls, label=lab))
    d.set_xscale("log")
    logticks(d, "x", [1, 10, 100], ["1", "10", "100"])
    d.set_xlim(1, 100)
    d.set_ylim(0, 1.62)
    d.set_yticks([0, 0.5, 1])
    d.set_xlabel(r"Delay $\tau$")
    d.set_ylabel(r"Recall $C_\tau$, $\gamma=16$")
    d.legend(handles=handles, loc="upper right", handlelength=1.9, borderpad=0.3, labelspacing=0.2)
    st.panel(d, "d", "Erasure test", dx=-34)
    st.save(fig, HERE / "fig3_dark.pdf")


# ----------------------------------------------------------------------------------------------- Figure 4
def qubit(k):
    chi = k ** 2 / ((k + 2) * (k + 1))
    size = 4 * k / ((k + 2) * (3 * k + 2))
    single = 1 - chi
    order = (1 + chi) / (1 - chi)
    return size, single, order


def order_map(ax, drive, cmap):
    """Share of the memory carried by products of k inputs, per unit ln k, against the loss rate."""
    rec = SNAP["spin"][drive]
    g = np.log10(rec["gamma"])
    grid = np.array(rec["order_grid"])
    dens = np.array(rec["order_density"]).T                  # (order bins, gamma)
    centres = np.log10(np.sqrt(grid[:-1] * grid[1:]))
    gx = np.linspace(g[0], g[-1], 160)
    oy = np.linspace(0, np.log10(4096), 150)
    on_order = np.array([np.interp(oy, centres, dens[:, j], left=dens[0, j], right=0) for j in range(len(g))])
    image = np.array([np.interp(gx, g, on_order[:, i]) for i in range(len(oy))])
    im = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=0.9,
                   extent=(g[0], g[-1], 0, np.log10(4096)), interpolation="bilinear", rasterized=True)
    mean = np.array(rec["heat_mean_order"])[0]
    ax.plot(g, np.log10(mean), color="white", lw=1.6, ls=(0, (3, 1.5)))
    logticks(ax, "x", [2, 3, 4], [r"$10^2$", r"$10^3$", r"$10^4$"])
    logticks(ax, "y", [0, 1, 2, 3], ["1", "10", "100", "1000"])
    ax.set_xlabel(r"Loss rate $\gamma$")
    return im


def fig4():
    """The qubit trade-off on top, the order of the three-spin memory below."""
    fig = plt.figure(figsize=(st.COLUMN, 300 * st.PT))
    a = st.axes(fig, 42, 182, 196, 90)
    b = st.axes(fig, 42, 40, 80, 100)
    c = st.axes(fig, 126, 40, 80, 100)
    cb = st.axes(fig, 210, 40, 5, 100)

    k = np.geomspace(0.01, 100, 400)
    size, single, order = qubit(k)
    a.axvspan(0.01, 0.12, color=st.SHADE, lw=0)
    a.axvspan(8, 100, color=st.SHADE, lw=0)
    a.plot(k, size / size.max(), color=st.GENTLE)
    a.plot(k, single, color=st.INK2, ls=(0, (3, 1.5)))
    a.axvline(2 / np.sqrt(3), color=st.GENTLE, lw=st.LW_GUIDE, ls=(0, (1, 1.6)))
    a.set_xscale("log")
    logticks(a, "x", [0.01, 0.1, 1, 10, 100], [r"$10^{-2}$", "0.1", "1", "10", r"$10^{2}$"])
    a.set_ylim(0, 1.14)
    a.set_xlim(0.01, 100)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Squared turn per memory time $\theta^2/\epsilon$")
    a.set_ylabel(r"Size, single share")
    text(a, 0.035, 0.4, "signal\nvanishes", st.INK2, ha="center")
    text(a, 29, 0.4, "inputs\nmix", st.INK2, ha="center")
    text(a, 3.2, 0.97, "signal size", st.GENTLE, ha="left")
    text(a, 0.0125, 0.88, "single inputs", st.INK2, ha="left", va="top")
    st.panel(a, "a", "One qubit", dx=-34)

    im = order_map(b, "strong", st.RECALL)
    order_map(c, "gentle", st.RECALL)
    b.set_ylabel(r"Inputs mixed, $k$")
    c.set_yticklabels([])
    for ax in (b, c):
        ax.tick_params(which="both", color="white")
    cbar = fig.colorbar(im, cax=cb)
    cbar.set_ticks([0, 0.4, 0.8])
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2.5, width=0.6, labelsize=7.2, direction="out", right=True)
    cbar.set_label(r"Share of memory per $\ln k$", fontsize=7.6, labelpad=2.5)
    st.panel(b, "b", "Fixed amplitude", dx=-34)
    st.panel(c, "c", "Scaled amplitude", dx=-6)
    st.save(fig, HERE / "fig4_writing.pdf")


# ----------------------------------------------------------------------------------------------- Figure 5
def fig5():
    """One reservoir at three reset probabilities against the law exp(-beta p K)."""
    fig = plt.figure(figsize=(st.COLUMN, 136 * st.PT))
    a = st.axes(fig, 42, 34, 196, 92)
    law = SNAP["law"]
    for (p, f), xy in zip((("0.03", 0.55), ("0.01", 0.28), ("0.003", 0.0)), ((40, 0.1), (66, 0.27), (170, 0.34))):
        y = np.array(law["one"][p])
        K = np.arange(len(y))
        col = st.tone(st.INK, f)
        a.plot(K, y, color=col)
        step = max(1, int(round(0.3 / (law["beta"][p] * float(p)))))
        Ks = K[::step]
        a.plot(Ks, np.exp(-law["beta"][p] * float(p) * Ks), "o", ms=st.MS, mfc="white", mec=col)
        text(a, *xy, (r"$p=" if p == "0.003" else "$") + rf"{float(p) * 100:g}\%$", col)
    a.set_xlim(0, 300)
    a.set_ylim(0, 1.04)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Inputs mixed, $K$")
    a.set_ylabel(r"Memory beyond $K$")
    a.legend(handles=[Line2D([], [], color=st.INK2, label="exact moments"),
                      Line2D([], [], color=st.INK2, ls="none", marker="o", ms=st.MS, mfc="white",
                             label=r"law, $e^{-\beta pK}$")], loc="upper right")
    st.save(fig, HERE / "fig5_law.pdf")


# ----------------------------------------------------------------------------------------------- Figure 6
def fig6():
    """Finite-sampling test of the first version: shared channel minus local loss."""
    fig = plt.figure(figsize=(st.COLUMN, 136 * st.PT))
    a = st.axes(fig, 42, 34, 196, 92)
    exact_x = 10 ** 7.95
    col = st.FAMILY["shared"]
    a.axhline(0, color=st.FAMILY["separate"], lw=st.LW_GUIDE, ls=st.DASH["separate"])
    a.axvline(10 ** 7.4, color=st.RULE, lw=0.8)
    for model, ls, f, lab, xy in (("grouped", "-", 0.0, "three grouped settings", (2.6e3, 1.35)),
                                  ("independent", (0, (3.2, 1.4)), 0.45, "each feature separately",
                                   (3.0e5, -1.2))):
        rec = SNAP["v1"][model]["shared"]
        c = st.tone(col, f)
        a.fill_between(rec["budget"], rec["lo"], rec["hi"], color=c, alpha=0.15, lw=0)
        a.plot(rec["budget"], rec["mean"], color=c, ls=ls, marker="o", ms=st.MS - 0.8, mfc=c, mec=c)
        text(a, *xy, lab, c, ha="left")
    m, lo, hi = SNAP["v1"]["grouped"]["shared"]["exact"]
    a.errorbar([exact_x], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=st.MS, mfc="white", mec=col, ecolor=col,
               elinewidth=1.0, capsize=0)
    text(a, 2.3e3, 0.2, "local loss", st.FAMILY["separate"], va="bottom")
    a.set_xscale("log")
    a.set_xlim(1.5e3, 2.4e8)
    a.xaxis.set_major_locator(ticker.FixedLocator([1e4, 1e5, 1e6, 1e7, exact_x]))
    a.xaxis.set_major_formatter(ticker.FixedFormatter([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", "exact"]))
    a.xaxis.set_minor_locator(ticker.FixedLocator(
        [k * 10 ** e for e in range(3, 8) for k in range(2, 10) if 1.5e3 < k * 10 ** e < 10 ** 7.35]))
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.set_ylim(-1.7, 4.3)
    a.set_xlabel(r"Preparations per step")
    a.set_ylabel(r"Shared minus local")
    st.save(fig, HERE / "fig6_sampling.pdf")


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
