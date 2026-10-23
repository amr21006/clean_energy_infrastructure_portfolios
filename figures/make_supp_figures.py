# -*- coding: utf-8 -*-
"""Supplementary Figures S1 to S3 from the stored analysis outputs.

S1  spell durations and next-state entropy
S2  cumulative incidence by entry state (final layout in supp_figure_s2.py)
S3  feature importance across the estimands (final layout in supp_figure_s3.py)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "delivery_twin" / "tables"
FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F", "lgrey": "#CFCFCF"}
OPC, CAC = "#0B6B3A", "#8B1A1A"
NL = "\n"

FEAT = {
    "months_to_reported_operation": "Months to reported operation date",
    "initial_reported_lead_months": "Initial reported lead time",
    "months_since_first_report": "Months since first report",
    "months_in_current_state": "Months in current state",
    "log1p_capacity_mw": "Capacity (log MW)",
    "schedule_revision_count": "Schedule revision count",
    "cumulative_schedule_slip_months": "Cumulative schedule slip",
    "absolute_schedule_movement_months": "Absolute schedule movement",
    "state_reversal_count": "State reversal count",
    "concurrent_entity_pipeline_units": "Concurrent entity pipeline units",
    "prior_entity_completions": "Prior entity completions",
    "current_state": "Current delivery state", "state_code": "US state",
    "technology_group": "Technology group", "sector_group": "Sector group",
    "energy_source_code": "Energy source code", "prime_mover_code": "Prime mover code",
}


def subcap(ax, letter, text, y=-0.28):
    ax.text(0.5, y, "(" + letter + ") " + text, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, linespacing=1.25)


def figS1():
    """Spell durations and next-state entropy."""
    sp = pd.read_csv(TAB / "state_spell_summary.csv")
    tm = pd.read_csv(TAB / "transition_metrics.csv")
    states = ["P", "L", "T", "U", "V", "TS"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    keep = ["solar", "storage", "wind"]
    sp2 = sp[sp.technology_group.isin(keep) & sp.state.isin(states)]
    w = 0.26
    tcol = {"solar": CB["orange"], "storage": CB["blue"], "wind": CB["green"]}
    for k, tech in enumerate(keep):
        d = sp2[sp2.technology_group == tech].set_index("state").reindex(states)
        xs = np.arange(len(states)) + (k - 1) * w
        med = d["median_months"].values
        lo = np.clip(med - d["p25_months"].values, 0, None)
        hi = np.clip(d["p75_months"].values - med, 0, None)
        ax.bar(xs, med, width=w, color=tcol[tech], label=tech.capitalize(),
               edgecolor="white", linewidth=0.3)
        ax.errorbar(xs, med, yerr=[lo, hi], fmt="none", ecolor="#3A3A3A",
                    elinewidth=0.6, capsize=1.4)
    ax.set_xticks(np.arange(len(states)), states)
    ax.set_xlabel("Reported delivery state")
    ax.set_ylabel("Spell duration (months)")
    ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=0.9,
              handletextpad=0.4, labelspacing=0.22)
    subcap(ax, "a", "Median spell length (bars) with" + NL + "interquartile range (whiskers)")

    ax = axes[1]
    t2 = tm[tm.current_state.isin(states)].copy()
    t2["current_state"] = pd.Categorical(t2.current_state, states, ordered=True)
    t2 = t2.sort_values("current_state")
    ax.bar(np.arange(len(t2)), t2.next_state_entropy.values, width=0.66,
           color=CB["purple"], edgecolor="white", linewidth=0.3)
    for xi, v in zip(np.arange(len(t2)), t2.next_state_entropy.values):
        ax.text(xi, v + 0.02, "%.2f" % v, ha="center", fontsize=6.8, color="#333333")
    ax.set_xticks(np.arange(len(t2)), t2.current_state.astype(str))
    ax.set_xlabel("Reported delivery state")
    ax.set_ylabel("Transition entropy (nats)")
    ax.set_ylim(0, 1.02)
    subcap(ax, "b", "Entropy of the next-state distribution," + NL + "Equation (2)")

    fig.savefig(FIG / "FigureS1.png")
    plt.close(fig)
    print("Figure S1 written (spell durations, entropy)")


def figS2():
    """Cumulative incidence of operation and cancellation."""
    h = pd.read_csv(TAB / "horizon_outcomes.csv")
    techs = ["solar", "storage", "wind"]
    states = ["P", "L", "T", "U", "V"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.6), sharey="row")
    fig.subplots_adjust(hspace=0.95, wspace=0.18)
    for r, (outcome, col, lab) in enumerate(
            [("operated", OPC, "Entered operation"), ("canceled", CAC, "Cancelled")]):
        for c, tech in enumerate(techs):
            ax = axes[r, c]
            w = 0.26
            for k, hm in enumerate([6, 12, 24]):
                vals, los, his = [], [], []
                for st in states:
                    d = h[(h.horizon_months == hm) & (h.initial_state == st)
                          & (h.technology_group == tech) & (h.outcome == outcome)]
                    if len(d) and int(d.eligible_units.iloc[0]) >= 20:
                        pv = float(d.probability.iloc[0]) * 100
                        vals.append(pv)
                        los.append(pv - float(d.ci95_low.iloc[0]) * 100)
                        his.append(float(d.ci95_high.iloc[0]) * 100 - pv)
                    else:
                        vals.append(np.nan); los.append(0); his.append(0)
                xs = np.arange(len(states)) + (k - 1) * w
                ax.bar(xs, vals, width=w, color=col, alpha=0.42 + 0.29 * k,
                       label="%d m" % hm, edgecolor="white", linewidth=0.3)
                ax.errorbar(xs, vals,
                            yerr=[np.clip(np.asarray(los, float), 0, None),
                                  np.clip(np.asarray(his, float), 0, None)],
                            fmt="none", ecolor="#3A3A3A", elinewidth=0.5, capsize=1.0)
            ax.set_xticks(np.arange(len(states)), states)
            if c == 0:
                ax.set_ylabel(lab + " (%)")
            ax.set_xlabel("State at first report")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7, title="Horizon", title_fontsize=7,
                          loc="upper left", handlelength=0.9, handletextpad=0.4,
                          labelspacing=0.2)
            subcap(ax, "abcdef"[r * 3 + c], tech.capitalize() + ":" + NL + lab.lower(),
                   y=-0.26)
    axes[0, 0].set_ylim(0, 100)
    axes[1, 0].set_ylim(0, 26)
    fig.savefig(FIG / "FigureS2.png")
    plt.close(fig)
    print("Figure S2 written (cumulative incidence)")


def figS3():
    """Feature importance for each estimand."""
    ns = pd.read_csv(TAB / "next_state_boosting_feature_importance.csv")
    op = pd.read_csv(TAB / "operation_12m_feature_importance.csv")
    ca = pd.read_csv(TAB / "cancellation_24m_feature_importance.csv")

    def norm(d):
        out = []
        for _, g in d.groupby("algorithm"):
            g = g.copy()
            tot = g.importance.sum()
            g["rel"] = g.importance / tot if tot > 0 else 0.0
            out.append(g)
        z = pd.concat(out)
        return z.groupby("feature", as_index=False)["rel"].mean().sort_values("rel")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9))
    fig.subplots_adjust(wspace=1.10)
    for ax, d, t, c, lab in [
            (axes[0], norm(ns), "Next reported state", CB["blue"], "a"),
            (axes[1], norm(op), "Operation within 12 months", OPC, "b"),
            (axes[2], norm(ca), "Cancellation within 24 months", CAC, "c")]:
        d = d.tail(11)
        y = np.arange(len(d))
        ax.barh(y, d.rel.values * 100, color=c, height=0.68, edgecolor="white",
                linewidth=0.3)
        ax.set_yticks(y, [FEAT.get(f, f) for f in d.feature], fontsize=6.8)
        ax.set_xlabel("Mean relative importance (%)")
        subcap(ax, lab, t, y=-0.20)
    fig.savefig(FIG / "FigureS3.png")
    plt.close(fig)
    print("Figure S3 written (feature importance)")


if __name__ == "__main__":
    for f in ("FigureS4.png", "FigureS5.png"):
        q = FIG / f
        if q.exists():
            q.unlink()
            print("removed redundant", f)
    figS1(); figS2(); figS3()
    print("\nSupplementary figures S1-S3 written to", FIG)
