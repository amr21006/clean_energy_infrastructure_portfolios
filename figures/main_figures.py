"""Figures 1 to 5 from the stored analysis outputs.

Reuses the styling and data sources of make_figures.py. Every value is read from
outputs/delivery_twin and outputs/supplementary; no model is re-fitted. Figure 1 is drawn by
fig1_flowchart.py and Figure 4 by fig4_equivalence.py; in Figure 5 the decision-curve
baselines are labelled "Treat all" and "Treat none" and the outcome prevalence is marked.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import make_figures as mf
from make_figures import CB, NICE, STATE_COL, TAB, _arrow, _box, _mcol, subcap

NICE["extra_trees"] = "Extremely randomized trees"
NICE["status_schedule_logistic"] = "Status-and-schedule logistic"

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",  # math symbols in the Times-like face of the text
})


# ---------------------------------------------------------------- Figure 1
def figure1():
    """Research framework; drawn by fig1_flowchart.draw."""
    from fig1_flowchart import draw
    print("Figure 1 written: %d boxes" % draw(OUT))


# ---------------------------------------------------------------- Figure 2
def figure2():
    """Nine-state delivery process; reversal arc drawn above the boxes, label clear."""
    fig = plt.figure(figsize=(6.9, 3.1))
    ax2 = fig.add_subplot(111)
    ax2.set_xlim(0, 11.4)
    ax2.set_ylim(0.0, 5.1)
    ax2.axis("off")

    CY, BW, BH = 3.20, 0.44, 0.36
    pos = {"P": (1.15, CY), "L": (2.75, CY), "T": (4.35, CY), "U": (5.95, CY),
           "V": (7.55, CY), "TS": (9.15, CY), "C": (3.55, 1.00), "O": (7.55, 1.00)}
    hz = {"P": 96.2, "L": 94.2, "T": 90.6, "U": 86.5, "V": 82.6, "TS": 76.4}
    adv = {"P": 2.9, "L": 4.7, "T": 8.2, "U": 11.1, "V": 6.2}

    ax2.plot([0.55, 4.95], [4.82, 4.82], color=CB["lgrey"], linewidth=3.0, solid_capstyle="butt")
    ax2.text(2.75, 4.92, "Pre-construction approvals", fontsize=7.4, style="italic",
             color="#5A5A5A", ha="center")
    ax2.plot([5.35, 9.75], [4.82, 4.82], color=CB["lgrey"], linewidth=3.0, solid_capstyle="butt")
    ax2.text(7.55, 4.92, "Construction and commissioning", fontsize=7.4, style="italic",
             color="#5A5A5A", ha="center")

    for st, (x, y) in pos.items():
        absorbing = st in ("O", "C")
        ax2.add_patch(FancyBboxPatch((x - BW, y - BH), 2 * BW, 2 * BH,
                                     boxstyle="round,pad=0.02,rounding_size=0.09",
                                     linewidth=1.7 if absorbing else 1.0,
                                     edgecolor=STATE_COL[st],
                                     facecolor=STATE_COL[st] + ("40" if absorbing else "22"),
                                     zorder=4))
        ax2.text(x, y, st, ha="center", va="center", fontsize=11, fontweight="bold",
                 color=STATE_COL[st], zorder=5)

    seq = ["P", "L", "T", "U", "V", "TS"]
    for a_, b_ in zip(seq[:-1], seq[1:]):
        xa, xb = pos[a_][0], pos[b_][0]
        ax2.add_patch(FancyArrowPatch((xa + BW + 0.05, CY), (xb - BW - 0.05, CY),
                                      arrowstyle="-|>", mutation_scale=7, linewidth=1.2,
                                      color="#1A1A1A", zorder=3, shrinkA=0, shrinkB=0))
        ax2.text((xa + xb) / 2, CY + 0.10, "%.1f%%" % adv[a_], ha="center", va="bottom",
                 fontsize=7.2, color="#1A1A1A")
    for st in seq:
        x = pos[st][0]
        ax2.add_patch(FancyArrowPatch((x - 0.17, CY + BH + 0.02), (x + 0.17, CY + BH + 0.02),
                                      connectionstyle="arc3,rad=-1.4", arrowstyle="-|>",
                                      mutation_scale=6, linewidth=0.9, color=CB["grey"],
                                      zorder=3))
        ax2.text(x, CY + 0.92, "%.1f%%" % hz[st], ha="center", va="bottom", fontsize=7.2,
                 color="#5A5A5A")
    ax2.text(0.55, CY + 0.98, "persistence", fontsize=7.0, style="italic", color="#5A5A5A",
             ha="right", va="bottom")
    ax2.text(0.55, CY + 0.10, "progression", fontsize=7.0, style="italic", color="#1A1A1A",
             ha="right", va="bottom")

    for st, rad, end in [("P", -0.16, -0.30), ("L", -0.06, -0.10), ("T", 0.08, 0.10),
                         ("U", 0.22, 0.30)]:
        x = pos[st][0]
        sgn = -1 if x < pos["C"][0] else 1
        ax2.add_patch(FancyArrowPatch((x + sgn * 0.15, CY - BH - 0.02),
                                      (pos["C"][0] + end, pos["C"][1] + BH + 0.02),
                                      connectionstyle="arc3,rad=%.2f" % rad, arrowstyle="-|>",
                                      mutation_scale=7, linewidth=1.1,
                                      color=STATE_COL["C"], alpha=0.9, zorder=3))
    ax2.text(3.05, 0.50, "Cancelled or postponed (absorbing)", ha="center", va="top",
             fontsize=7.4, color=STATE_COL["C"], fontweight="bold")
    ax2.text(3.05, 0.20, "per month: P 0.8%, L 0.8%, T 0.6%, U 0.1%",
             ha="center", va="top", fontsize=7.0, color=STATE_COL["C"])

    for st, rad in [("U", -0.20), ("V", 0.0), ("TS", 0.20)]:
        x = pos[st][0]
        sgn = -1 if x < pos["O"][0] else (0 if x == pos["O"][0] else 1)
        ax2.add_patch(FancyArrowPatch((x + sgn * 0.15, CY - BH - 0.02),
                                      (pos["O"][0] + sgn * 0.20, pos["O"][1] + BH + 0.02),
                                      connectionstyle="arc3,rad=%.2f" % rad, arrowstyle="-|>",
                                      mutation_scale=7, linewidth=1.1,
                                      color=STATE_COL["O"], alpha=0.9, zorder=3))
    ax2.text(8.25, 0.50, "Commercial operation (absorbing)", ha="center", va="top",
             fontsize=7.4, color=STATE_COL["O"], fontweight="bold")
    ax2.text(8.25, 0.20, "per month: U 2.0%, V 10.9%, TS 22.7%",
             ha="center", va="top", fontsize=7.0, color=STATE_COL["O"])

    # Reversal U to T drawn between the progression line and the persistence labels.
    ax2.add_patch(FancyArrowPatch((pos["U"][0] - 0.30, CY + BH + 0.02),
                                  (pos["T"][0] + 0.30, CY + BH + 0.02),
                                  connectionstyle="arc3,rad=0.42", arrowstyle="-|>",
                                  mutation_scale=6, linewidth=1.2, color=CB["purple"],
                                  linestyle=(0, (2.2, 1.1)), zorder=6))
    ax2.text(5.15, CY + 0.80, "reversal 0.3%", ha="center", va="center", fontsize=7.0,
             color=CB["purple"], zorder=7)

    fig.savefig(OUT / "Figure2.png", dpi=500)
    plt.close(fig)
    print("Figure 2 written")


# ---------------------------------------------------------------- Figure 3
def figure3():
    """Delivery-state dynamics and descriptive signatures of attrition (legends moved)."""
    tp = pd.read_csv(TAB / "transition_probabilities.csv")
    tm = pd.read_csv(TAB / "transition_metrics.csv")
    obi = pd.read_csv(TAB / "outcomes_by_initial_state.csv")
    sr = pd.read_csv(TAB / "schedule_and_reversal_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    fig.subplots_adjust(hspace=0.80, wspace=0.38)

    ax = axes[0, 0]
    m = tp.pivot_table(index="current_state", columns="next_state",
                       values="probability", fill_value=0.0)
    order = mf.STATE_ORDER
    rows = [x for x in order if x in m.index and x not in ("O", "C")]
    colsx = [x for x in order if x in m.columns]
    M = m.reindex(index=rows, columns=colsx).values * 100.0
    im = ax.imshow(np.log10(M + 0.01), cmap="viridis", aspect="auto",
                   vmin=np.log10(0.01), vmax=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.05:
                ax.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=6.4,
                        color="white" if np.log10(v + 0.01) < 1.15 else "black")
    ax.set_xticks(range(len(colsx)), colsx, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("State in month $t+1$")
    ax.set_ylabel("State in month $t$")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_ticks([np.log10(0.01), np.log10(0.1), np.log10(1), np.log10(10), np.log10(100)])
    cb.set_ticklabels(["0", "0.1", "1", "10", "100"])
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
        sp_.set_linewidth(0.4)
    subcap(ax, "a", "Monthly transition probabilities (%),\nlogarithmic color scale", y=-0.27)

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
        ax.text(xi, tot + 0.5, "%.1f" % tot, ha="center", fontsize=7.2, color="#333333")
    ax.set_xticks(x, tm2.current_state.astype(str))
    ax.set_xlabel("Reported delivery state")
    ax.set_ylabel("Monthly exit hazard (%)")
    ax.set_ylim(0, max(bottom) * 1.36)
    ax.legend(frameon=False, loc="upper left", fontsize=7.6, handlelength=0.8,
              handletextpad=0.35, labelspacing=0.18)
    subcap(ax, "b", "Decomposition of monthly exit hazards", y=-0.27)

    ax = axes[1, 0]
    order6 = ["P", "L", "T", "U", "V", "TS"]
    cats = [("operating_or_later_retired", "Operating", STATE_COL["O"]),
            ("planned", "Still planned", CB["sky"]),
            ("canceled", "Cancelled", STATE_COL["C"]),
            ("unresolved", "Unresolved", CB["lgrey"])]
    x = np.arange(len(order6))
    bottom = np.zeros(len(order6))
    for key, lab, c in cats:
        vals = []
        for st in order6:
            r = obi[(obi.initial_state == st) & (obi.outcome == key)]
            vals.append(float(r.unit_share_within_subgroup.iloc[0]) * 100 if len(r) else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottom, color=c, label=lab, width=0.70,
               edgecolor="white", linewidth=0.3)
        bottom += vals
    for st, xi in zip(order6, x):
        r = obi[(obi.initial_state == st) & (obi.outcome == "canceled")]
        if len(r):
            ax.text(xi, 102, "%.1f" % (float(r.unit_share_within_subgroup.iloc[0]) * 100),
                    ha="center", fontsize=7.2, color=STATE_COL["C"])
    ax.set_xticks(x, order6)
    ax.set_xlabel("State at first report")
    ax.set_ylabel("Share of units (%)")
    ax.set_ylim(0, 112)
    ax.legend(ncol=4, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handlelength=0.8, handletextpad=0.3, columnspacing=0.7, frameon=False)
    subcap(ax, "c", "Outcome at May 2026 by entry state\n(cancelled share, %, annotated)",
           y=-0.27)

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
            ax.text(xi, v + 1.5, "%.0f" % v, ha="center", va="center", fontsize=7.2,
                    color="#333333", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_xticks(np.arange(3), ["Solar", "Storage", "Wind"])
    ax.set_xlabel("Technology group")
    ax.set_ylabel("Net schedule slippage (months)")
    ax.set_ylim(0, 42)
    ax.legend(ncol=2, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handlelength=0.8, handletextpad=0.3, columnspacing=0.9, frameon=False)
    subcap(ax, "d", "Median net slippage (bars),\n75th percentile (whiskers)", y=-0.27)

    fig.savefig(OUT / "Figure3.png")
    plt.close(fig)
    print("Figure 3 written")


# ---------------------------------------------------------------- Figure 4
def figure4():
    """Paired differences from the full logistic regression; drawn by fig4_equivalence.draw."""
    from fig4_equivalence import draw
    draw(OUT)
    print("Figure 4 written")


# ---------------------------------------------------------------- Figure 5
def figure5():
    """Calibration and decision curves; treat-all and treat-none labelled consistently."""
    opc = pd.read_csv(TAB / "operation_12m_calibration_bins.csv")
    cac = pd.read_csv(TAB / "cancellation_24m_calibration_bins.csv")
    opd = pd.read_csv(TAB / "operation_12m_decision_curve.csv")
    cad = pd.read_csv(TAB / "cancellation_24m_decision_curve.csv")

    show_op = ["proposed_delivery_ensemble", "full_logistic", "tabm_gpu",
               "delivery_sparse_moe_gpu", "catboost_gpu"]
    show_ca = ["delivery_competing_risk_net_gpu", "full_logistic", "xgboost_gpu",
               "tabm_gpu", "proposed_delivery_ensemble"]
    pal = [CB["red"], CB["black"], CB["blue"], CB["green"], CB["orange"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))
    fig.subplots_adjust(hspace=0.62, wspace=0.34)
    leg = dict(frameon=False, fontsize=7.4, handlelength=1.3, handletextpad=0.4,
               labelspacing=0.25)

    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], color=CB["grey"], linestyle=":", linewidth=0.9, zorder=1)
    for m, c in zip(show_op, pal):
        d = opc[opc.model == m].sort_values("mean_predicted")
        ax.plot(d.mean_predicted, d.observed_rate, "-o", ms=2.8, linewidth=0.9,
                color=c, label=NICE[m], mfc="white", mew=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", **leg)
    subcap(ax, "a", "Calibration: operation within 12 months", y=-0.19)

    ax = axes[0, 1]
    ref = opd[opd.model == show_op[0]].sort_values("threshold")
    ref = ref[ref.treat_all_net_benefit >= -0.02]
    ax.plot(ref.threshold, ref.treat_all_net_benefit, color=CB["grey"], linestyle="--",
            linewidth=1.0, label="Treat all")
    ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0, label="Treat none")
    for m, c in zip(show_op, pal):
        d = opd[opd.model == m].sort_values("threshold")
        ax.plot(d.threshold, d.net_benefit, linewidth=1.0, color=c, label=NICE[m])
    ax.set_xlabel("Decision threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.01, 0.5)
    ax.set_ylim(-0.17, 0.45)
    ax.legend(loc="lower left", ncol=2, columnspacing=0.8, **leg)
    subcap(ax, "b", "Decision curve: operation within 12 months", y=-0.19)

    ax = axes[1, 0]
    lim = 0.31
    ax.plot([0, lim], [0, lim], color=CB["grey"], linestyle=":", linewidth=0.9)
    for m, c in zip(show_ca, pal):
        d = cac[cac.model == m].sort_values("mean_predicted")
        ax.plot(d.mean_predicted, d.observed_rate, "-o", ms=2.8, linewidth=0.9,
                color=c, label=NICE[m], mfc="white", mew=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.legend(loc="upper left", **leg)
    subcap(ax, "c", "Calibration: cancellation within 24 months", y=-0.19)

    ax = axes[1, 1]
    ref = cad[cad.model == show_ca[0]].sort_values("threshold")
    ax.plot(ref.threshold, ref.treat_all_net_benefit, color=CB["grey"], linestyle="--",
            linewidth=1.0, label="Treat all")
    ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0, label="Treat none")
    for m, c in zip(show_ca, pal):
        d = cad[cad.model == m].sort_values("threshold")
        ax.plot(d.threshold, d.net_benefit, linewidth=1.0, color=c, label=NICE[m])
    prevalence = 58 / 1095
    ax.axvline(prevalence, color="#9A9A9A", linewidth=0.7, linestyle="-", zorder=0)
    ax.text(prevalence + 0.004, 0.0445, "prevalence\n5.3%", fontsize=7.0, color="#4A4A4A",
            va="top", ha="left")
    ax.set_xlabel("Decision threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.01, 0.30)
    ax.set_ylim(-0.020, 0.046)
    ax.legend(loc="upper right", **leg)
    subcap(ax, "d", "Decision curve: cancellation within 24 months", y=-0.19)

    fig.savefig(OUT / "Figure5.png")
    plt.close(fig)
    print("Figure 5 written")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
