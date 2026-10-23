"""Figure 4: held-out difference of each specification from the
full logistic regression, with paired 90% cluster-bootstrap intervals and the equivalence
margins shaded (ROC-AUC 0.02 for operation within 12 months, PR-AUC 0.05 for cancellation
within 24 months). Complements Table 2, which reports the levels."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_figures import CB, subcap

ROOT = Path(__file__).resolve().parents[1]
EQ = ROOT / "outputs" / "supplementary" / "s_equivalence_vs_full_logistic.csv"

BASE = {"Status-schedule logistic"}
PURPOSE = {"Competing-risk network", "Delivery-PathMoE (transition)", "Delivery-PathMoE (rollout)",
           "Delivery-state pooled ensemble"}
ABLATION = {"Equal-weight pool (ablation)"}
RENAME = {"Status-schedule logistic": "Status-and-schedule logistic"}


def _color(name: str) -> str:
    if name in BASE:
        return CB["grey"]
    if name in PURPOSE:
        return CB["red"]
    if name in ABLATION:
        return CB["purple"]
    return CB["blue"]


def draw(out_dir: Path) -> None:
    eq = pd.read_csv(EQ)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    fig.subplots_adjust(wspace=1.05, top=0.86)
    panels = [("operation_12m", "roc_auc", 0.02, "Difference in ROC-AUC",
               "a", "Operation within 12 months\n(margin ±0.02)"),
              ("cancellation_24m", "average_precision", 0.05, "Difference in PR-AUC",
               "b", "Cancellation within 24 months\n(margin ±0.05)")]
    for ax, (ep, metric, margin, xlab, letter, cap) in zip(axes, panels):
        d = eq[(eq.endpoint == ep) & (eq.metric == metric)].sort_values(
            "difference_vs_full_logistic").reset_index(drop=True)
        y = np.arange(len(d))
        ax.axvspan(-margin, margin, color="#E6E6E6", zorder=0)
        ax.axvline(0, color="#4A4A4A", linewidth=0.8, zorder=1)
        ax.errorbar(d.difference_vs_full_logistic, y,
                    xerr=[d.difference_vs_full_logistic - d.ci90_low,
                          d.ci90_high - d.difference_vs_full_logistic],
                    fmt="none", elinewidth=0.9, capsize=1.6, ecolor="#5A5A5A", zorder=2)
        for xi, yi, name, inside in zip(d.difference_vs_full_logistic, y, d.model,
                                        d.within_margin_90):
            c = _color(name)
            ax.plot(xi, yi, "o", ms=4.2, color=c, mfc=c if inside else "white", mew=1.1,
                    zorder=3)
        ax.set_yticks(y, [RENAME.get(m, m) for m in d.model], fontsize=7.8)
        ax.set_xlabel(xlab)
        lim = max(margin * 1.6, float(np.abs(d[["ci90_low", "ci90_high"]].values).max()) * 1.08)
        ax.set_xlim(-lim, lim)
        subcap(ax, letter, cap, y=-0.20)

    fam = [(CB["grey"], "Transparent baseline"),
           (CB["blue"], "Tree ensemble or tabular deep learning"),
           (CB["red"], "Purpose-built specification"), (CB["purple"], "Ablation")]
    handles = [Line2D([], [], marker="o", linestyle="none", ms=4.5, color=c, label=l)
               for c, l in fam]
    handles += [Line2D([], [], marker="o", linestyle="none", ms=4.5, color="#4A4A4A",
                       mfc="white", label="90% interval beyond the margin")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=7.6, handletextpad=0.3, columnspacing=1.2)
    fig.savefig(out_dir / "Figure4.png")
    plt.close(fig)
