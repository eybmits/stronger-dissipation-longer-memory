#!/usr/bin/env python3
"""Figures 2-6 from paper_v2/data/snapshot.json (written by export_snapshot.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def fig2():
    fig = plt.figure(figsize=(st.PAGE, 176 * st.PT))
    a = st.axes(fig, 36, 48, 124, 100)
    c = st.axes(fig, 220, 48, 124, 100)
    d = st.axes(fig, 382, 48, 124, 100)
    tau = np.arange(1, 301)
    for fam, (weak, strong) in (("separate", ((1.25, 0.86), (2.3, 0.05))), ("shared", ((11, 0.86), (30, 0.62)))):
        rec = SNAP["memory"][fam]
        for g, f, (x, y) in ((0.25, 0.55, weak), (64.0, 0.0, strong)):
            col = st.tone(st.FAMILY[fam], f)
            a.plot(tau, rec["curve_mean"][rec["curve_gamma"].index(g)], color=col, lw=1.3)
        a.text(*strong, st.NAME[fam] + r", $\gamma=64$", color=st.FAMILY[fam], fontsize=6.6, ha="left")
    a.text(12, 0.8, r"both, $\gamma=1/4$", color=st.INK3, fontsize=6.6, ha="left")
    a.set_xscale("log")
    logticks(a, "x", [1, 10, 100], ["1", "10", "100"])
    a.set_xlim(1, 300)
    a.set_ylim(0, 1.05)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Delay $\tau$")
    a.set_ylabel(r"Recall $C_\tau$")
    st.panel(a, "a", "Recall at weak and strong loss", dx=-26)

    handles = []
    far = SNAP["review"].get("far", {})
    for fam in families():
        rec = SNAP["memory"][fam]
        x = np.array(rec["gamma"])
        # beyond gamma = 64 the recall of one and two shared channels reaches past the 300 delays summed
        inside = x <= (256 if fam in ("signed", "pair") else 64)
        st.band(c, x[inside], stack(rec["recall"])[:, inside], st.FAMILY[fam], lw=1.3, ls=st.DASH[fam])
        xf, rows = x, stack(rec["forget"])
        if fam in far:
            sel = np.array(far[fam]["gamma"]) <= FAR_MAX
            xf = np.r_[x, np.array(far[fam]["gamma"])[sel]]
            rows = np.hstack([rows, stack(far[fam]["forget"])[:, sel]])
        st.band(d, xf, rows, st.FAMILY[fam], lw=1.3, ls=st.DASH[fam])
        handles.append(Line2D([], [], color=st.FAMILY[fam], lw=1.4, ls=st.DASH[fam], label=st.NAME[fam]))
    # mixed phases with the input amplitude scaled as min(1, gamma^(-1/2))
    sc = SNAP["review"]["scaled"]["1"]
    xs = np.array(sc["gamma"])
    for ax, key in ((c, "recall"), (d, "forget")):
        st.band(ax, xs, stack(sc[key]), st.FAMILY["signed"], lw=1.1, ls=SCALED, marker="o", ms=2.4, alpha=0.12)
    c.axhline(45, color=st.INK3, lw=0.8, ls=(0, (4, 2)))
    c.text(0.13, 45.8, "45 features", color=st.INK3, fontsize=6.4, va="bottom")
    c.annotate(r"scaled input", xy=(90, 34.6), xytext=(4.5, 39.5), color=st.FAMILY["signed"], fontsize=6.4,
               ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=st.FAMILY["signed"], lw=0.5,
                                                         shrinkA=0.5, shrinkB=1.5))
    for ax, top in ((c, 256), (d, FAR_MAX)):
        ax.set_xscale("log")
        st.decades(ax, "x", labels="plain")
        ax.set_xlim(0.1, top)
        ax.set_xlabel(r"Loss rate $\gamma$")
    c.set_ylim(0, 50)
    c.set_ylabel(r"Memory $\sum_\tau C_\tau$")
    d.set_yscale("log")
    st.decades(d, "y")
    d.set_ylim(0.03, 2e4)
    gg = np.geomspace(100, FAR_MAX, 10)
    d.plot(gg, 1.6 * gg, ls=(0, (1, 1.5)), color=st.FAMILY["shared"], lw=0.8)
    d.text(150, 700, r"$\propto\gamma$", color=st.FAMILY["shared"], fontsize=7, ha="right")
    gl = np.geomspace(4, 200, 10)
    d.plot(gl, 0.9 / gl, ls=(0, (1, 1.5)), color=st.FAMILY["separate"], lw=0.8)
    d.text(5.0, 0.07, r"$\propto 1/\gamma$", color=st.FAMILY["separate"], fontsize=7, ha="right")
    d.set_ylabel(r"Time to forget $t_{\mathrm f}$ (steps)")
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, handlelength=1.4, frameon=False,
               columnspacing=1.0, labelspacing=0.25, bbox_to_anchor=((220 + 506) / 2 / 510, 0.0))
    st.panel(c, "b", "Memory against loss", dx=-26)
    st.panel(d, "c", "Slow modes", dx=-26)
    st.save(fig, HERE / "fig2_families.pdf")


def fig3():
    fig = plt.figure(figsize=(st.PAGE, 152 * st.PT))
    a = st.axes(fig, 60, 30, 76, 100)
    side = 100 * st.PAGE / st.PT / st.DESIGN_PAGE        # printed width of panel b, to make it square
    b = st.axes(fig, 166, 130 - side, 100, side)
    c = st.axes(fig, 296, 30, 96, 100)
    d = st.axes(fig, 420, 30, 80, 100)

    groups = ((r"no dark state", ("separate", "exchange", "gainloss")),
              (r"populations only", ("dephasing",)),
              (r"dark subspace", ("pair", "groups", "signed", "shared")))
    yy, rows, heads = 0.0, [], []
    for title, fams in groups:
        heads.append((yy, title))
        yy += 0.9
        for f in fams:
            rows.append((yy, f))
            yy += 1.0
        yy += 0.25
    a.set_xscale("log")
    for k, (yh, title) in enumerate(heads):
        a.axhline(yh - 0.42, color=st.RULE, lw=0.5) if k else None
        a.text(0.78, yh, rf"\textit{{{title}}}", fontsize=6.1, color=st.INK3, va="center", ha="left")
    for yi, f in rows:
        v, col = SNAP["kernel"][f], st.FAMILY[f]
        a.plot([0.75, v], [yi, yi], color=st.tone(col, 0.45), lw=1.8, solid_capstyle="round", zorder=1)
        a.plot(v, yi, "o", ms=5.0, mfc="white" if f == "dephasing" else col, mec=col, mew=1.0, zorder=2)
        lab = {36: r"$36=6^2$", 100: r"$100=10^2$"}.get(v, f"${v}$")
        a.text(v * 1.6, yi, lab, va="center", ha="left", fontsize=6.2, color=st.INK2)
    a.set_xlim(0.65, 9000)
    logticks(a, "x", [1, 10, 100, 1000], ["1", "10", "100", r"$10^3$"])
    a.set_yticks([yi for yi, _ in rows])
    a.set_yticklabels([st.NAME[f] for _, f in rows], fontsize=6.6)
    a.tick_params(axis="y", length=0)
    a.set_ylim(yy - 0.15, -0.95)
    a.set_xlabel(r"Protected operators")
    st.panel(a, "a", "What the loss leaves alone", dx=-55)

    th = np.linspace(0, 2 * np.pi, 400)
    b.plot(np.cos(th), np.sin(th), color=st.INK3, lw=0.6)
    maps = SNAP["spectrum"]["maps"]
    for key, col, ms in (("shared_1", st.tone(st.FAMILY["shared"], 0.62), 2.0),
                         ("shared_8", st.tone(st.FAMILY["shared"], 0.32), 2.0),
                         ("shared_64", st.FAMILY["shared"], 2.0),
                         ("separate_8", st.FAMILY["separate"], 2.4)):
        b.plot(maps[key]["re"], maps[key]["im"], "o", ms=ms, mfc=col, mec="none", alpha=0.95)
    b.set_xlim(-1.36, 1.36)
    b.set_ylim(-1.36, 1.36)
    b.set_aspect("equal", adjustable="box")
    b.set_xticks([-1, 0, 1])
    b.set_yticks([-1, 0, 1])
    b.set_xlabel(r"Re $\lambda$")
    b.set_ylabel(r"Im $\lambda$", labelpad=0)
    b.text(-1.3, -1.3, r"local, $\gamma=8$", color=st.FAMILY["separate"], fontsize=6.4, ha="left", va="bottom")
    b.text(-1.3, 1.3, "shared", color=st.FAMILY["shared"], fontsize=6.4, ha="left", va="top")
    b.text(1.3, 1.3, r"$\gamma=1,8,64$", color=st.FAMILY["shared"], fontsize=6.4, ha="right", va="top")
    st.panel(b, "b", "One averaged step", dx=-20)

    red = SNAP["memory"]["shared"]["reduced"]
    tau = np.arange(1, 301)
    for gkey, col in (("4", st.tone(st.FAMILY["shared"], 0.5)), ("64", st.FAMILY["shared"])):
        full, model = np.array(red[gkey]["full"]), np.array(red[gkey]["reduced"])
        c.plot(tau, full, color=col, lw=1.2)
        pick = np.unique(np.round(np.geomspace(1, 300, 22)).astype(int)) - 1
        c.plot(tau[pick], model[pick], "o", ms=2.6, mfc="white", mec=col, mew=0.7)
    c.set_xscale("log")
    logticks(c, "x", [1, 10, 100], ["1", "10", "100"])
    c.set_xlim(1, 300)
    c.set_ylim(0, 1.02)
    c.set_xlabel(r"Delay $\tau$")
    c.set_ylabel(r"Recall $C_\tau$")
    c.text(2.2, 0.35, r"$\gamma=4$", color=st.tone(st.FAMILY["shared"], 0.5), fontsize=6.8)
    c.text(40, 0.58, r"$\gamma=64$", color=st.FAMILY["shared"], fontsize=6.8)
    c.legend(handles=[Line2D([], [], color=st.INK2, lw=1.1, label="full reservoir"),
                      Line2D([], [], color=st.INK2, ls="none", marker="o", ms=2.6, mfc="white", mew=0.7,
                             label="dark states only")], loc="upper right", fontsize=6.0)
    st.panel(c, "c", "Dark-state model")

    curves = SNAP["removal"]["16"]["curves"]
    tau = np.arange(1, 101)
    spec = (("intact", st.FAMILY["shared"], "-", "intact"),
            ("rotated", st.tone(st.FAMILY["shared"], 0.45), (0, (4, 1.6)), "rotated copy erased"),
            ("removed", st.ERASED, (0, (1.2, 1.2)), "dark states erased"))
    handles = []
    for field, col, ls, lab in spec:
        st.band(d, tau, np.array(curves[field]), col, lw=1.3, ls=ls, alpha=0.16)
        handles.append(Line2D([], [], color=col, lw=1.3, ls=ls, label=lab))
    d.set_xscale("log")
    logticks(d, "x", [1, 10, 100], ["1", "10", "100"])
    d.set_xlim(1, 100)
    d.set_ylim(0, 1.45)
    d.set_yticks([0, 0.5, 1])
    d.set_xlabel(r"Delay $\tau$")
    d.set_ylabel(r"Recall $C_\tau$ at $\gamma=16$")
    d.legend(handles=handles, loc="upper right", fontsize=5.9, handlelength=1.7, borderpad=0.3, labelspacing=0.2)
    st.panel(d, "d", "Erasure test")
    st.save(fig, HERE / "fig3_dark.pdf")


def qubit(k):
    chi = k ** 2 / ((k + 2) * (k + 1))
    size = 4 * k / ((k + 2) * (3 * k + 2))
    single = 1 - chi
    order = (1 + chi) / (1 - chi)
    return size, single, order


def order_curves(ax, drive, colour, gammas):
    """Share of the memory carried by products of k inputs, per unit ln k, at a few loss rates."""
    rec = SNAP["spin"][drive]
    grid = np.array(rec["order_grid"])
    centres = np.sqrt(grid[:-1] * grid[1:])
    dens = np.array(rec["order_density"])
    shades = np.linspace(0.55, 0.0, len(gammas))
    for g, f in zip(gammas, shades):
        ax.plot(centres, dens[rec["gamma"].index(g)], color=st.tone(colour, f), lw=1.2)
    ax.set_xscale("log")
    logticks(ax, "x", [1, 10, 100, 1000], ["1", "10", "100", r"$10^3$"])
    ax.set_xlim(1, 4096)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Inputs mixed, $k$")
    return shades


def fig4():
    """One column: the qubit trade-off on top, the order of the three-spin memory below."""
    fig = plt.figure(figsize=(st.COLUMN, 292 * st.PT))
    a = st.axes(fig, 38, 176, 196, 92)
    b = st.axes(fig, 38, 30, 96, 100)
    c = st.axes(fig, 138, 30, 96, 100)

    k = np.geomspace(0.01, 100, 400)
    size, single, order = qubit(k)
    a.axvspan(0.01, 0.12, color=st.SHADE, lw=0)
    a.axvspan(8, 100, color=st.SHADE, lw=0)
    a.plot(k, size / size.max(), color=st.GENTLE, lw=1.3)
    a.plot(k, single, color=st.INK2, lw=1.0, ls=(0, (3, 1.5)))
    a.set_xscale("log")
    logticks(a, "x", [0.01, 0.1, 1, 10, 100], [r"$10^{-2}$", "0.1", "1", "10", r"$10^{2}$"])
    a.set_ylim(0, 1.12)
    a.set_xlim(0.01, 100)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Squared turn per memory time $\theta^2/\epsilon$")
    a.set_ylabel(r"Size, single-input share")
    a.text(0.034, 0.42, "signal\nvanishes", fontsize=6.4, ha="center", color=st.INK2)
    a.text(30, 0.42, "inputs\nmix", fontsize=6.4, ha="center", color=st.INK2)
    a.text(1.75, 0.93, "signal size", fontsize=6.5, color=st.GENTLE, ha="left")
    a.text(0.0125, 0.86, "single inputs", fontsize=6.5, color=st.INK2, ha="left", va="top")
    a.axvline(2 / np.sqrt(3), color=st.GENTLE, lw=0.5, ls=":")
    st.panel(a, "a", "One qubit", dx=-30)

    gammas = (16.0, 128.0, 1024.0, 8192.0, 65536.0)
    sh = order_curves(b, "strong", st.STRONG, gammas)
    for g, f, x in zip(gammas, sh, (1.75, 6.0, 30, 140, 900)):
        n = int(np.log2(g))
        lab = rf"$\gamma=2^{{{n}}}$" if g == 16 else rf"$2^{{{n}}}$"
        b.text(x, 0.49 if g > 16 else 1.12, lab, color=st.tone(st.STRONG, f), fontsize=6.2, ha="left")
    sh = order_curves(c, "gentle", st.GENTLE, gammas)
    c.text(1.75, 0.83, r"$\gamma=2^4$", color=st.tone(st.GENTLE, sh[0]), fontsize=6.2, ha="left")
    c.text(14, 0.55, r"$2^7$ to $2^{16}$", color=st.GENTLE, fontsize=6.2, ha="left")
    b.set_ylabel(r"Share of memory per $\ln k$")
    c.set_yticklabels([])
    st.panel(b, "b", "Fixed amplitude", dx=-30)
    st.panel(c, "c", "Scaled amplitude", dx=-6)
    st.save(fig, HERE / "fig4_writing.pdf")


def fig5():
    """One reservoir at three reset probabilities against the law exp(-beta p K)."""
    fig = plt.figure(figsize=(st.COLUMN, 128 * st.PT))
    a = st.axes(fig, 40, 30, 190, 86)
    law = SNAP["law"]
    for (p, f), xy in zip((("0.03", 0.55), ("0.01", 0.28), ("0.003", 0.0)), ((14, 0.36), (62, 0.27), (175, 0.33))):
        y = np.array(law["one"][p])
        K = np.arange(len(y))
        col = st.tone(st.INK, f)
        a.plot(K, y, color=col, lw=1.2)
        step = max(1, int(round(0.25 / (law["beta"][p] * float(p)))))
        Ks = K[::step]
        a.plot(Ks, np.exp(-law["beta"][p] * float(p) * Ks), "o", ms=2.6, mfc="white", mec=col, mew=0.7)
        a.text(*xy, (r"$p=" if p == "0.003" else "$") + rf"{float(p) * 100:g}\%$", color=col, fontsize=6.6)
    a.set_xlim(0, 300)
    a.set_ylim(0, 1.03)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"Inputs mixed, $K$")
    a.set_ylabel(r"Memory beyond $K$")
    a.legend(handles=[Line2D([], [], color=st.INK2, lw=1.2, label="exact moments"),
                      Line2D([], [], color=st.INK2, ls="none", marker="o", ms=2.6, mfc="white", mew=0.7,
                             label=r"law, $e^{-\beta pK}$")], loc="upper right", fontsize=6.0)
    st.save(fig, HERE / "fig5_law.pdf")


def fig6():
    """Finite-sampling test of the first version: shared channel minus local loss."""
    fig = plt.figure(figsize=(st.COLUMN, 128 * st.PT))
    a = st.axes(fig, 40, 30, 190, 86)
    exact_x = 10 ** 7.95
    col = st.FAMILY["shared"]
    a.axhline(0, color=st.FAMILY["separate"], lw=0.8, ls=st.DASH["separate"])
    a.axvline(10 ** 7.4, color=st.RULE, lw=0.6)
    for model, ls, f, lab, xy in (("grouped", "-", 0.0, "three grouped settings", (2.6e5, 1.75)),
                                  ("independent", (0, (4, 1.6)), 0.45, "each feature separately",
                                   (4.0e5, -1.15))):
        rec = SNAP["v1"][model]["shared"]
        c = st.tone(col, f)
        a.fill_between(rec["budget"], rec["lo"], rec["hi"], color=c, alpha=0.16, lw=0)
        a.plot(rec["budget"], rec["mean"], color=c, lw=1.3, ls=ls, marker="o", ms=2.4, mfc=c, mec=c)
        a.text(*xy, lab, color=c, fontsize=6.4, ha="left" if model == "grouped" else "left")
    m, lo, hi = SNAP["v1"]["grouped"]["shared"]["exact"]
    a.errorbar([exact_x], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=3.0, mfc="white", mec=col, ecolor=col,
               elinewidth=0.7, capsize=0)
    a.text(2.4e3, 0.15, "local loss", color=st.FAMILY["separate"], fontsize=6.4, va="bottom")
    a.set_xscale("log")
    a.set_xlim(1.5e3, 2.4e8)
    a.xaxis.set_major_locator(ticker.FixedLocator([1e4, 1e5, 1e6, 1e7, exact_x]))
    a.xaxis.set_major_formatter(ticker.FixedFormatter([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", "exact"]))
    a.xaxis.set_minor_locator(ticker.FixedLocator(
        [k * 10 ** e for e in range(3, 8) for k in range(2, 10) if 1.5e3 < k * 10 ** e < 10 ** 7.35]))
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.set_ylim(-1.6, 4.2)
    a.set_xlabel(r"Preparations per step")
    a.set_ylabel(r"Memory, shared minus local")
    st.save(fig, HERE / "fig6_sampling.pdf")


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
