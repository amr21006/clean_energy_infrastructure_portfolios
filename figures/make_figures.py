"""Shared plotting style, helpers and base figure layouts, from the frozen delivery-twin results.

No models are re-fitted. Every value plotted is read from outputs/delivery_twin.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "delivery_twin" / "tables"
RES = ROOT / "outputs" / "delivery_twin" / "results"
FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures" / "base"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

# Colour-blind-safe palette (Okabe-Ito)
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "black": "#000000",
    "grey": "#7F7F7F", "lgrey": "#CFCFCF",
}
STATE_ORDER = ["P", "L", "T", "U", "V", "TS", "O", "C"]
STATE_COL = {
    "P": "#4C4C9E", "L": "#5E8BC4", "T": "#56B4E9", "U": "#E69F00",
    "V": "#D55E00", "TS": "#009E73", "O": "#0B6B3A", "C": "#8B1A1A",
    "OTHER": "#BDBDBD",
}
STATE_LAB = {
    "P": "P  Planned, approvals not initiated",
    "L": "L  Approvals pending",
    "T": "T  Approvals received, not started",
    "U": "U  Under construction ≤ 50%",
    "V": "V  Under construction > 50%",
    "TS": "TS  Complete / testing",
    "O": "O  Commercial operation",
    "C": "C  Cancelled / postponed",
}


def panel_label(ax, txt, dx=-0.085, dy=1.06):
    """Deprecated: retained for Figure 1 only (schematic panels)."""
    ax.text(dx, dy, txt, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")


def subcap(ax, letter, text, y=-0.30):
    """Sub-caption placed below the panel."""
    ax.text(0.5, y, f"({letter}) {text}", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, linespacing=1.25)


# ---------------------------------------------------------------- Figure 1
def _box(ax, x, y, w, h, face, edge, lw=0.9, rad=0.04, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=%.3f" % rad,
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def _arrow(ax, xy1, xy2, color="#333333", lw=1.1, ms=6, z=6):
    """Shaft always longer than the head, so arrows never collapse into a blob."""
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def figure1():
    """Research design and analysis protocol."""
    fig = plt.figure(figsize=(6.5, 6.2))

    # ==================================================== panel (a)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    DARK = ["#33455F", "#1F6F54", "#8A5A20", "#6B2740"]
    TINT = ["#EEF2F7", "#EBF5F1", "#FBF3E8", "#F9EDF0"]
    LIGHT = {"blue": "#E3EDF6", "green": "#E2F0E9", "red": "#F7E6E6", "plum": "#F1E4EA"}

    # four components of a single design, not four sequential stages
    regions = [(80.0, 19.5, "DATA AND COHORT", 0),
               (52.0, 26.0, "THE DELIVERY TWIN:  process representation and study design", 1),
               (27.5, 23.0, "MODEL COMPARISON:  fourteen specifications, four families", 2),
               (0.5, 26.0, "EVALUATION, CLAIM RULES AND GOVERNANCE READ-ACROSS", 3)]
    for y, h, lab, k in regions:
        _box(ax, 0.4, y, 99.2, h, TINT[k], TINT[k], lw=0.0, rad=0.02, z=1)
        ax.text(2.0, y + h - 1.5, lab, fontsize=6.8, fontweight="bold", color=DARK[k],
                va="top", ha="left", zorder=9)

    # ---- stage 1 : four boxes, wide gaps so the arrows have a real shaft
    s1 = [(2.6, 19.6, "EIA-860M archive",
           "129 monthly snapshots\nSep 2015 to May 2026"),
          (26.2, 17.4, "All planned units",
           "10,262 generating units\never reported as planned"),
          (47.6, 24.4, "Prespecified exclusions",
           "Archive burn-in  (−1,630)\nNon-clean technology  (−1,379)\n"
           "Operating at entry  (−81)"),
          (76.0, 21.4, "PRIMARY COHORT",
           "7,172 generating units\n6,713 development clusters")]
    for i, (x, w, hdr, body) in enumerate(s1):
        _box(ax, x, 82.0, w, 11.2, "white", DARK[0], lw=1.6 if i == 3 else 1.0, z=3)
        ax.text(x + w / 2, 91.5, hdr, ha="center", va="center", fontsize=6.4,
                fontweight="bold", color=DARK[0], zorder=5)
        ax.text(x + w / 2, 86.5, body, ha="center", va="center", fontsize=5.9,
                zorder=5, linespacing=1.6)
        if i < 3:
            _arrow(ax, (x + w + 0.5, 87.6), (s1[i + 1][0] - 0.5, 87.6), DARK[0], lw=1.2, ms=6)
    ax.text(50.0, 80.8, "Units grouped by plant, technology group and first reported operation "
            "month; all uncertainty resamples whole clusters",
            ha="center", va="center", fontsize=5.8, style="italic", color="#5A5A5A", zorder=5)
    _arrow(ax, (50.0, 79.7), (50.0, 78.3), DARK[0], lw=1.3, ms=6)

    # ---- stage 2 : boxes pushed clear of the stage title
    s2 = [(2.6, 28.4, "Status harmonisation",
           "9 states  P  L  T  U  V  TS  O  C  Other\n118,141 monthly transitions\n"
           "Equations (1) and (2)"),
          (35.8, 28.4, "Features at the origin month",
           "Reported state, duration, schedule\nrevisions, slippage, reversals,\n"
           "capacity and pipeline position"),
          (69.0, 28.4, "Temporal entry cohorts",
           "Train to Dec 2021  |  Validation 2022\nHeld-out test Jan 2023 to May 2024\n"
           "project-disjoint")]
    for i, (x, w, hdr, body) in enumerate(s2):
        _box(ax, x, 63.6, w, 10.2, "white", DARK[1], lw=1.0, z=3)
        ax.text(x + w / 2, 72.0, hdr, ha="center", va="center", fontsize=6.4,
                fontweight="bold", color=DARK[1], zorder=5)
        ax.text(x + w / 2, 67.4, body, ha="center", va="center", fontsize=5.8,
                zorder=5, linespacing=1.6)
        if i < 2:
            _arrow(ax, (x + w + 0.5, 68.7), (s2[i + 1][0] - 0.5, 68.7), DARK[1], lw=1.2, ms=6)
    ax.text(50.0, 62.4, "Equation (3): landmark definitions and censoring rule",
            ha="center", va="center", fontsize=5.9, style="italic", color="#5A5A5A", zorder=5)

    est = [("ESTIMAND 1", "Next reported state\n(9 classes, one month ahead)",
            "short-cycle progress control", CB["blue"], LIGHT["blue"]),
           ("ESTIMAND 2", "Operation within 12 months\nof first report",
            "schedule assurance", STATE_COL["O"], LIGHT["green"]),
           ("ESTIMAND 3", "Cancellation within 24 months\nof first report",
            "attrition and contingency", STATE_COL["C"], LIGHT["red"])]
    for i, (tag, body, gov, col, fill) in enumerate(est):
        x = 2.6 + i * 33.2
        _box(ax, x, 52.9, 31.6, 8.6, fill, col, lw=1.1, z=3)
        ax.text(x + 15.8, 60.2, tag, ha="center", va="center", fontsize=6.2,
                fontweight="bold", color=col, zorder=5)
        ax.text(x + 15.8, 57.3, body, ha="center", va="center", fontsize=5.9,
                color="#1A1A1A", zorder=5, linespacing=1.55)
        ax.text(x + 15.8, 54.2, gov, ha="center", va="center", fontsize=5.7,
                style="italic", color="#4A4A4A", zorder=5)
        _arrow(ax, (x + 15.8, 52.7), (x + 15.8, 50.7), DARK[1], lw=1.2, ms=6)

    # ---- stage 3
    fam = [("TRANSPARENT\nBASELINES",
            "Empirical Markov chain\nMultinomial logistic\nStatus-schedule logistic\n"
            "Full logistic", CB["grey"]),
           ("TREE AND\nBOOSTING",
            "Random forest\nExtremely randomised trees\nHistogram gradient boosting\n"
            "LightGBM, XGBoost, CatBoost", CB["blue"]),
           ("TABULAR DEEP\nLEARNING",
            "TabM\nDelivery residual network\nCompeting-risk network\n"
            "(+ no-Markov ablation)", CB["orange"]),
           ("SPARSE MoE\nAND POOLING",
            "Delivery-PathMoE\ntop-2 routing, Eq. (4), (5)\nPooled ensemble, Eq. (6)\n"
            "(+ equal-weight ablation)", CB["red"])]
    for i, (hdr, body, col) in enumerate(fam):
        x = 2.6 + i * 24.4
        _box(ax, x, 29.4, 23.0, 16.4, "white", col, lw=1.2, z=3)
        _box(ax, x, 40.6, 23.0, 5.2, col + "26", col, lw=1.2, z=4)
        ax.text(x + 11.5, 43.2, hdr, ha="center", va="center", fontsize=6.0,
                fontweight="bold", color=col, zorder=6, linespacing=1.35)
        ax.text(x + 11.5, 34.9, body, ha="center", va="center", fontsize=5.8,
                color="#1A1A1A", zorder=5, linespacing=1.8)
        _arrow(ax, (x + 11.5, 29.2), (x + 11.5, 26.9), DARK[2], lw=1.2, ms=6)

    # ---- stage 4
    ev = [("Discrimination", "Log loss, Brier score\nMacro-F1, ROC-AUC\nAverage precision\n"
                             "Per-class F1"),
          ("Calibration", "Calibration intercept\nand slope\nDecile calibration\ncurves"),
          ("Decision utility", "Net benefit across\nplausible thresholds\nEquation (8)"),
          ("Uncertainty", "500 cluster-bootstrap\nresamples; paired\ndifferences, Eq. (7)\n"
                          "A win needs a 95%\ninterval excluding zero"),
          ("Prespecified checks", "Data continuity\nSplit disjointness\n"
                                  "Probability-mass\nconservation\nBounded probabilities")]
    for i, (hdr, body) in enumerate(ev):
        x = 2.6 + i * 19.4
        _box(ax, x, 9.4, 17.9, 12.6, "white", DARK[3], lw=1.0, z=3)
        ax.text(x + 8.95, 20.8, hdr, ha="center", va="center", fontsize=6.0,
                fontweight="bold", color=DARK[3], zorder=5)
        ax.text(x + 8.95, 14.5, body, ha="center", va="center", fontsize=5.4,
                color="#1A1A1A", zorder=5, linespacing=1.5)
        _arrow(ax, (x + 8.95, 9.2), (x + 8.95, 7.8), DARK[3], lw=1.2, ms=6)

    out = [("Schedule assurance", "supported by the reported record",
            STATE_COL["O"], LIGHT["green"]),
           ("Project-level attrition triage", "not supported", STATE_COL["C"], LIGHT["red"]),
           ("Portfolio contingency", "the governable margin", DARK[3], LIGHT["plum"])]
    for i, (lab, verdict, col, fill) in enumerate(out):
        x = 2.6 + i * 32.4
        _box(ax, x, 0.9, 30.6, 6.6, fill, col, lw=1.3, z=3)
        ax.text(x + 15.3, 5.6, lab, ha="center", va="center", fontsize=6.1,
                fontweight="bold", color=col, zorder=5)
        ax.text(x + 15.3, 2.9, verdict, ha="center", va="center", fontsize=5.8,
                style="italic", color="#3A3A3A", zorder=5)


    fig.savefig(FIG / "Figure1.png", dpi=500)
    plt.close(fig)
    print("Figure 1 written (research design)")


# ---------------------------------------------------------------- Figure 2
def figure2():
    """The harmonised nine-state delivery process, with observed monthly hazards."""
    fig = plt.figure(figsize=(6.9, 2.9))

    # ==================================================== panel (b)
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(0, 11.4)
    ax2.set_ylim(0.0, 4.8)
    ax2.axis("off")

    CY, BW, BH = 3.20, 0.44, 0.36
    pos = {"P": (1.15, CY), "L": (2.75, CY), "T": (4.35, CY), "U": (5.95, CY),
           "V": (7.55, CY), "TS": (9.15, CY), "C": (3.55, 1.00), "O": (7.55, 1.00)}
    hz = {"P": 96.2, "L": 94.2, "T": 90.6, "U": 86.5, "V": 82.6, "TS": 76.4}
    adv = {"P": 2.9, "L": 4.7, "T": 8.2, "U": 11.1, "V": 6.2}

    ax2.plot([0.55, 4.95], [4.48, 4.48], color=CB["lgrey"], linewidth=3.0, solid_capstyle="butt")
    ax2.text(2.75, 4.58, "Pre-construction approvals", fontsize=6.6, style="italic",
             color="#6A6A6A", ha="center")
    ax2.plot([5.35, 9.75], [4.48, 4.48], color=CB["lgrey"], linewidth=3.0, solid_capstyle="butt")
    ax2.text(7.55, 4.58, "Construction and commissioning", fontsize=6.6, style="italic",
             color="#6A6A6A", ha="center")

    for st, (x, y) in pos.items():
        absorbing = st in ("O", "C")
        ax2.add_patch(FancyBboxPatch((x - BW, y - BH), 2 * BW, 2 * BH,
                                     boxstyle="round,pad=0.02,rounding_size=0.09",
                                     linewidth=1.7 if absorbing else 1.0,
                                     edgecolor=STATE_COL[st],
                                     facecolor=STATE_COL[st] + ("40" if absorbing else "22"),
                                     zorder=4))
        ax2.text(x, y, st, ha="center", va="center", fontsize=10, fontweight="bold",
                 color=STATE_COL[st], zorder=5)

    seq = ["P", "L", "T", "U", "V", "TS"]
    for a_, b_ in zip(seq[:-1], seq[1:]):
        xa, xb = pos[a_][0], pos[b_][0]
        ax2.add_patch(FancyArrowPatch((xa + BW + 0.05, CY), (xb - BW - 0.05, CY),
                                      arrowstyle="-|>", mutation_scale=7, linewidth=1.2,
                                      color="#1A1A1A", zorder=3, shrinkA=0, shrinkB=0))
        ax2.text((xa + xb) / 2, CY + 0.10, "%.1f%%" % adv[a_], ha="center", va="bottom",
                 fontsize=6.3, color="#1A1A1A")
    for st in seq:
        x = pos[st][0]
        ax2.add_patch(FancyArrowPatch((x - 0.17, CY + BH + 0.02), (x + 0.17, CY + BH + 0.02),
                                      connectionstyle="arc3,rad=-1.4", arrowstyle="-|>",
                                      mutation_scale=6, linewidth=0.9, color=CB["grey"],
                                      zorder=3))
        ax2.text(x, CY + 0.92, "%.1f%%" % hz[st], ha="center", va="bottom", fontsize=6.3,
                 color=CB["grey"])
    ax2.text(0.52, CY + 0.98, "persistence", fontsize=6.1, style="italic", color=CB["grey"],
             ha="right", va="bottom")
    ax2.text(0.52, CY + 0.10, "progression", fontsize=6.1, style="italic", color="#1A1A1A",
             ha="right", va="bottom")

    for st, rad in [("P", -0.16), ("L", -0.06), ("T", 0.08), ("U", 0.22)]:
        x = pos[st][0]
        sgn = -1 if x < pos["C"][0] else 1
        ax2.add_patch(FancyArrowPatch((x + sgn * 0.15, CY - BH - 0.02),
                                      (pos["C"][0] + sgn * 0.20, pos["C"][1] + BH + 0.02),
                                      connectionstyle="arc3,rad=%.2f" % rad, arrowstyle="-|>",
                                      mutation_scale=7, linewidth=1.1,
                                      color=STATE_COL["C"], alpha=0.9, zorder=3))
    ax2.text(pos["C"][0], 0.48, "Cancelled / postponed   (absorbing)", ha="center", va="top",
             fontsize=6.5, color=STATE_COL["C"], fontweight="bold")
    ax2.text(pos["C"][0], 0.22, "monthly hazard   P 0.8%   L 0.8%   T 0.6%   U 0.1%",
             ha="center", va="top", fontsize=6.1, color=STATE_COL["C"])

    for st, rad in [("U", -0.20), ("V", 0.0), ("TS", 0.20)]:
        x = pos[st][0]
        sgn = -1 if x < pos["O"][0] else (0 if x == pos["O"][0] else 1)
        ax2.add_patch(FancyArrowPatch((x + sgn * 0.15, CY - BH - 0.02),
                                      (pos["O"][0] + sgn * 0.20, pos["O"][1] + BH + 0.02),
                                      connectionstyle="arc3,rad=%.2f" % rad, arrowstyle="-|>",
                                      mutation_scale=7, linewidth=1.1,
                                      color=STATE_COL["O"], alpha=0.9, zorder=3))
    ax2.text(pos["O"][0], 0.48, "Commercial operation   (absorbing)", ha="center", va="top",
             fontsize=6.5, color=STATE_COL["O"], fontweight="bold")
    ax2.text(pos["O"][0], 0.22, "monthly hazard   U 2.0%   V 10.9%   TS 22.7%",
             ha="center", va="top", fontsize=6.1, color=STATE_COL["O"])

    ax2.add_patch(FancyArrowPatch((pos["U"][0] - 0.20, CY - BH - 0.02),
                                  (pos["T"][0] + 0.20, CY - BH - 0.02),
                                  connectionstyle="arc3,rad=-0.46", arrowstyle="-|>",
                                  mutation_scale=6, linewidth=1.3, color=CB["purple"],
                                  linestyle=(0, (2.2, 1.1)), zorder=6))
    ax2.text(5.15, CY - 0.66, "reversal 0.3%", ha="center", va="center", fontsize=6.1,
             color=CB["purple"], zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.9))


    fig.savefig(FIG / "Figure2.png", dpi=500)
    plt.close(fig)
    print("Figure 2 written (delivery process)")


# ---------------------------------------------------------------- Figure 3
def figure3():
    """Delivery-state dynamics and the descriptive signatures of attrition."""
    tp = pd.read_csv(TAB / "transition_probabilities.csv")
    tm = pd.read_csv(TAB / "transition_metrics.csv")
    obi = pd.read_csv(TAB / "outcomes_by_initial_state.csv")
    sr = pd.read_csv(TAB / "schedule_and_reversal_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    fig.subplots_adjust(hspace=0.72, wspace=0.36)

    # (a) monthly transition matrix
    ax = axes[0, 0]
    m = tp.pivot_table(index="current_state", columns="next_state",
                       values="probability", fill_value=0.0)
    rows = [x for x in STATE_ORDER if x in m.index and x not in ("O", "C")]
    colsx = [x for x in STATE_ORDER if x in m.columns]
    M = m.reindex(index=rows, columns=colsx).values * 100.0
    im = ax.imshow(np.log10(M + 0.01), cmap="viridis", aspect="auto",
                   vmin=np.log10(0.01), vmax=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.05:
                ax.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=5.4,
                        color="white" if np.log10(v + 0.01) < 1.15 else "black")
    ax.set_xticks(range(len(colsx)), colsx, fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlabel("State in month $t+1$")
    ax.set_ylabel("State in month $t$")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_ticks([np.log10(0.01), np.log10(0.1), np.log10(1), np.log10(10), np.log10(100)])
    cb.set_ticklabels(["0", "0.1", "1", "10", "100"])
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
        sp_.set_linewidth(0.4)
    subcap(ax, "a", "Monthly transition probabilities (%),\nlogarithmic colour scale", y=-0.26)

    # (b) exit-hazard decomposition
    ax = axes[0, 1]
    tm2 = tm[tm.current_state.isin(["P", "L", "T", "U", "V", "TS"])].copy()
    tm2["current_state"] = pd.Categorical(tm2.current_state,
                                          ["P", "L", "T", "U", "V", "TS"], ordered=True)
    tm2 = tm2.sort_values("current_state")
    comps = [("advance_probability", "Progress", CB["sky"]),
             ("operation_probability", "Enter operation", STATE_COL["O"]),
             ("reversal_probability", "Reverse", CB["purple"]),
             ("cancellation_probability", "Cancel", STATE_COL["C"])]
    x = np.arange(len(tm2))
    bottom = np.zeros(len(tm2))
    for col, lab, c in comps:
        v = tm2[col].values * 100
        ax.bar(x, v, bottom=bottom, color=c, label=lab, width=0.68,
               edgecolor="white", linewidth=0.3)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.5, "%.1f" % tot, ha="center", fontsize=6.2, color="#333333")
    ax.set_xticks(x, tm2.current_state.astype(str))
    ax.set_xlabel("Reported delivery state")
    ax.set_ylabel("Monthly exit hazard (%)")
    ax.set_ylim(0, max(bottom) * 1.34)
    ax.legend(frameon=False, loc="upper left", fontsize=6.6, handlelength=0.8,
              handletextpad=0.35, labelspacing=0.18)
    subcap(ax, "b", "Decomposition of monthly exit hazards", y=-0.26)

    # (c) terminal outcome by entry state
    ax = axes[1, 0]
    order = ["P", "L", "T", "U", "V", "TS"]
    cats = [("operating_or_later_retired", "Operating", STATE_COL["O"]),
            ("planned", "Still planned", CB["sky"]),
            ("canceled", "Cancelled", STATE_COL["C"]),
            ("unresolved", "Unresolved", CB["lgrey"])]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for key, lab, c in cats:
        vals = []
        for st in order:
            r = obi[(obi.initial_state == st) & (obi.outcome == key)]
            vals.append(float(r.unit_share_within_subgroup.iloc[0]) * 100 if len(r) else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottom, color=c, label=lab, width=0.70,
               edgecolor="white", linewidth=0.3)
        bottom += vals
    for st, xi in zip(order, x):
        r = obi[(obi.initial_state == st) & (obi.outcome == "canceled")]
        if len(r):
            ax.text(xi, 102, "%.1f" % (float(r.unit_share_within_subgroup.iloc[0]) * 100),
                    ha="center", fontsize=6.2, color=STATE_COL["C"])
    ax.set_xticks(x, order)
    ax.set_xlabel("State at first report")
    ax.set_ylabel("Share of units (%)")
    ax.set_ylim(0, 112)
    leg = ax.legend(ncol=2, fontsize=6.4, loc="lower left", handlelength=0.8,
                    handletextpad=0.35, columnspacing=0.6, frameon=True,
                    facecolor="white", edgecolor="none", framealpha=0.90)
    leg.set_zorder(10)
    subcap(ax, "c", "Terminal outcome by entry state\n(cancelled share, %, annotated)", y=-0.26)

    # (d) net schedule slippage, operating vs cancelled
    ax = axes[1, 1]
    techs = ["solar", "storage", "wind"]
    sr2 = sr[sr.technology_group.isin(techs)
             & sr.terminal_category.isin(["operating", "canceled"])]
    ccol = {"operating": STATE_COL["O"], "canceled": STATE_COL["C"]}
    w = 0.34
    for k, tc in enumerate(["operating", "canceled"]):
        d = sr2[sr2.terminal_category == tc].set_index("technology_group").reindex(techs)
        xs = np.arange(3) + (k - 0.5) * w
        ax.bar(xs, d["net_schedule_slip_median_months"].values, width=w, color=ccol[tc],
               label="Operating" if tc == "operating" else "Cancelled",
               edgecolor="white", linewidth=0.3)
        for xi, v, p75 in zip(xs, d["net_schedule_slip_median_months"].values,
                              d["net_schedule_slip_p75_months"].values):
            ax.plot([xi, xi], [v, p75], color="#3A3A3A", linewidth=0.7, zorder=2)
            ax.plot([xi - 0.07, xi + 0.07], [p75, p75], color="#3A3A3A", linewidth=0.7)
            ax.text(xi, v + 1.4, "%.0f" % v, ha="center", va="center", fontsize=6.4,
                    color="#333333", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_xticks(np.arange(3), ["Solar", "Storage", "Wind"])
    ax.set_xlabel("Technology group")
    ax.set_ylabel("Net schedule slippage (months)")
    ax.set_ylim(0, 42)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=0.8,
              handletextpad=0.35, labelspacing=0.18, title="Terminal outcome",
              title_fontsize=6.6)
    subcap(ax, "d", "Median net slippage (bars),\n75th percentile (whiskers)", y=-0.26)

    fig.savefig(FIG / "Figure3.png")
    plt.close(fig)
    print("Figure 3 written")


# ---------------------------------------------------------------- Figure 4
def figure4():
    """Discrimination at the two landmark horizons, with cluster-bootstrap intervals."""
    op = pd.read_csv(TAB / "operation_12m_model_performance.csv")
    ca = pd.read_csv(TAB / "cancellation_24m_model_performance.csv")
    opci = pd.read_csv(TAB / "operation_12m_bootstrap_ci.csv")
    caci = pd.read_csv(TAB / "cancellation_24m_bootstrap_ci.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    fig.subplots_adjust(wspace=1.05)

    ax = axes[0]
    a = opci[opci.metric == "roc_auc"].merge(op[["model", "roc_auc"]], on="model",
                                             how="left").sort_values("roc_auc")
    y = np.arange(len(a))
    ax.errorbar(a.roc_auc.values, y,
                xerr=[a.roc_auc.values - a.ci95_low.values,
                      a.ci95_high.values - a.roc_auc.values],
                fmt="none", elinewidth=0.9, capsize=1.6, ecolor="#5A5A5A")
    for xi, yi, m in zip(a.roc_auc.values, y, a.model):
        ax.plot(xi, yi, "o", ms=3.6, color=_mcol(m), zorder=5)
    ax.set_yticks(y, [NICE[m] for m in a.model], fontsize=6.8)
    ax.set_xlabel("Test ROC-AUC")
    ax.grid(axis="x", linewidth=0.3, color=CB["lgrey"], alpha=0.7)
    ax.set_axisbelow(True)
    subcap(ax, "a", "Operation within 12 months\n(cluster-bootstrap 95% CI)", y=-0.19)

    ax = axes[1]
    b = caci[caci.metric == "average_precision"].merge(
        ca[["model", "average_precision"]], on="model",
        how="left").sort_values("average_precision")
    y = np.arange(len(b))
    ax.errorbar(b.average_precision.values, y,
                xerr=[b.average_precision.values - b.ci95_low.values,
                      b.ci95_high.values - b.average_precision.values],
                fmt="none", elinewidth=0.9, capsize=1.6, ecolor="#5A5A5A")
    for xi, yi, m in zip(b.average_precision.values, y, b.model):
        ax.plot(xi, yi, "o", ms=3.6, color=_mcol(m), zorder=5)
    base = 58 / 1095
    ax.axvline(base, color=CB["black"], linestyle="--", linewidth=0.9)
    ax.text(base + 0.006, len(b) - 0.3, "prevalence\n5.3%", fontsize=6.2,
            color=CB["black"], va="top")
    ax.set_yticks(y, [NICE[m] for m in b.model], fontsize=6.8)
    ax.set_xlabel("Test average precision")
    ax.set_xlim(0.02, 0.29)
    ax.grid(axis="x", linewidth=0.3, color=CB["lgrey"], alpha=0.7)
    ax.set_axisbelow(True)
    subcap(ax, "b", "Cancellation within 24 months\n(cluster-bootstrap 95% CI)", y=-0.19)

    fig.savefig(FIG / "Figure4.png")
    plt.close(fig)
    print("Figure 4 written")


NICE = {
    "empirical_markov": "Empirical Markov", "multinomial_logistic": "Multinomial logistic",
    "status_schedule_logistic": "Status–schedule logistic", "full_logistic": "Full logistic",
    "random_forest": "Random forest", "extra_trees": "Extremely randomised trees",
    "hist_gradient_boosting": "Histogram gradient boosting", "lightgbm_cpu": "LightGBM",
    "catboost_gpu": "CatBoost", "xgboost_gpu": "XGBoost", "tabm_gpu": "TabM",
    "delivery_sparse_moe_gpu": "Delivery-PathMoE (transition)",
    "delivery_path_moe_gpu": "Delivery-PathMoE (rollout)",
    "delivery_state_residual_net_gpu": "Delivery residual network",
    "delivery_competing_risk_net_gpu": "Competing-risk network",
    "delivery_net_no_markov_ablation": "Residual network, no Markov prior",
    "equal_weight_pool_ablation": "Equal-weight pool (ablation)",
    "proposed_delivery_ensemble": "Delivery-state pooled ensemble",
}
BASELINE = {"empirical_markov", "multinomial_logistic", "status_schedule_logistic",
            "full_logistic"}


def _mcol(m):
    if m in BASELINE:
        return CB["grey"]
    if m.startswith("delivery") or m == "proposed_delivery_ensemble":
        return CB["red"]
    if m == "equal_weight_pool_ablation":
        return CB["purple"]
    return CB["blue"]


# ---------------------------------------------------------------- Figure 5
def figure5():
    """The cancellation blind spot: per-class F1 across every specification."""
    pc = pd.read_csv(TAB / "next_state_per_class_performance.csv")
    order_m = ["empirical_markov", "multinomial_logistic", "random_forest", "extra_trees",
               "hist_gradient_boosting", "lightgbm_cpu", "catboost_gpu", "xgboost_gpu",
               "tabm_gpu", "delivery_sparse_moe_gpu", "delivery_state_residual_net_gpu",
               "delivery_net_no_markov_ablation", "equal_weight_pool_ablation",
               "proposed_delivery_ensemble"]
    classes = ["P", "L", "T", "U", "V", "TS", "O", "C"]
    sup = {c: int(pc[(pc.model == "catboost_gpu") & (pc["class"] == c)].support.iloc[0])
           for c in classes}

    Mx = np.zeros((len(order_m), len(classes)))
    for i, m in enumerate(order_m):
        for j, c in enumerate(classes):
            r = pc[(pc.model == m) & (pc["class"] == c)]
            Mx[i, j] = float(r.f1.iloc[0]) if len(r) else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    im = ax.imshow(Mx, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            v = Mx[i, j]
            ax.text(j, i, "0.00" if v == 0 else "%.2f" % v, ha="center", va="center",
                    fontsize=6.4, fontweight="bold" if v == 0 else "normal",
                    color="white" if v == 0 else "#222222")
    ax.add_patch(Rectangle((len(classes) - 1.5, -0.5), 1.0, len(order_m),
                           fill=False, edgecolor="black", linewidth=1.8, zorder=6))
    ax.set_xticks(range(len(classes)),
                  ["%s\n(n=%s)" % (c, format(sup[c], ",")) for c in classes], fontsize=7)
    ax.set_yticks(range(len(order_m)), [NICE[m] for m in order_m], fontsize=7)
    ax.set_xlabel("Realised next state in the held-out test cohort")
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cb.set_label("Per-class $F_1$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
        sp_.set_linewidth(0.4)
    fig.savefig(FIG / "Figure5.png")
    plt.close(fig)
    print("Figure 5 written")


# ---------------------------------------------------------------- Figure 6
def figure6():
    opc = pd.read_csv(TAB / "operation_12m_calibration_bins.csv")
    cac = pd.read_csv(TAB / "cancellation_24m_calibration_bins.csv")
    opd = pd.read_csv(TAB / "operation_12m_decision_curve.csv")
    cad = pd.read_csv(TAB / "cancellation_24m_decision_curve.csv")
    op = pd.read_csv(TAB / "operation_12m_model_performance.csv")
    ca = pd.read_csv(TAB / "cancellation_24m_model_performance.csv")

    show_op = ["proposed_delivery_ensemble", "full_logistic", "tabm_gpu",
               "delivery_sparse_moe_gpu", "catboost_gpu"]
    show_ca = ["delivery_competing_risk_net_gpu", "full_logistic", "xgboost_gpu",
               "tabm_gpu", "proposed_delivery_ensemble"]
    pal = [CB["red"], CB["black"], CB["blue"], CB["green"], CB["orange"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    fig.subplots_adjust(hspace=0.70, wspace=0.32)

    # (a) calibration op12
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], color=CB["grey"], linestyle=":", linewidth=0.9, zorder=1)
    for m, c in zip(show_op, pal):
        d = opc[opc.model == m].sort_values("mean_predicted")
        ax.plot(d.mean_predicted, d.observed_rate, "-o", ms=2.6, linewidth=0.9,
                color=c, label=NICE[m], mfc="white", mew=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.2,
              handletextpad=0.4, labelspacing=0.25)
    subcap(ax, "a", "Calibration: operation within 12 months", y=-0.20)

    # (b) decision curve op12
    ax = axes[0, 1]
    ref = opd[opd.model == show_op[0]].sort_values("threshold")
    ax.plot(ref.threshold, ref.treat_all_net_benefit, color=CB["grey"], linestyle="--",
            linewidth=0.9, label="Commit to all")
    ax.axhline(0, color=CB["grey"], linestyle=":", linewidth=0.9)
    for m, c in zip(show_op, pal):
        d = opd[opd.model == m].sort_values("threshold")
        ax.plot(d.threshold, d.net_benefit, linewidth=1.0, color=c, label=NICE[m])
    ax.set_xlabel("Decision threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.01, 0.5)
    ax.set_ylim(-0.02, 0.45)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.2,
              handletextpad=0.4, labelspacing=0.25)
    subcap(ax, "b", "Decision curve: operation within 12 months", y=-0.20)

    # (c) calibration canc24
    ax = axes[1, 0]
    lim = 0.31
    ax.plot([0, lim], [0, lim], color=CB["grey"], linestyle=":", linewidth=0.9)
    for m, c in zip(show_ca, pal):
        d = cac[cac.model == m].sort_values("mean_predicted")
        ax.plot(d.mean_predicted, d.observed_rate, "-o", ms=2.6, linewidth=0.9,
                color=c, label=NICE[m], mfc="white", mew=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.2,
              handletextpad=0.4, labelspacing=0.25)
    subcap(ax, "c", "Calibration: cancellation within 24 months", y=-0.20)

    # (d) decision curve canc24
    ax = axes[1, 1]
    ref = cad[cad.model == show_ca[0]].sort_values("threshold")
    ax.plot(ref.threshold, ref.treat_all_net_benefit, color=CB["grey"], linestyle="--",
            linewidth=0.9, label="Treat all as at risk")
    ax.axhline(0, color=CB["grey"], linestyle=":", linewidth=0.9)
    for m, c in zip(show_ca, pal):
        d = cad[cad.model == m].sort_values("threshold")
        ax.plot(d.threshold, d.net_benefit, linewidth=1.0, color=c, label=NICE[m])
    ax.set_xlabel("Decision threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.01, 0.30)
    ax.set_ylim(-0.020, 0.045)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right", handlelength=1.2,
              handletextpad=0.4, labelspacing=0.25)
    subcap(ax, "d", "Decision curve: cancellation within 24 months", y=-0.20)

    fig.savefig(FIG / "Figure6.png")
    plt.close(fig)
    print("Figure 6 written")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("\nAll main figures written to", FIG)
