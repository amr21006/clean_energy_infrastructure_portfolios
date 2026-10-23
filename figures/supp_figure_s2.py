# -*- coding: utf-8 -*-
"""Supplementary Figure S2: cumulative incidence by entry state, with the horizon legend in
months. Values and layout are those of make_supp_figures.figS2.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_supp_figures as base

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    h = pd.read_csv(base.TAB / "horizon_outcomes.csv")
    techs = ["solar", "storage", "wind"]
    states = ["P", "L", "T", "U", "V"]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.6), sharey="row")
    fig.subplots_adjust(hspace=0.95, wspace=0.18)
    for r, (outcome, col, lab) in enumerate(
            [("operated", base.OPC, "Entered operation"), ("canceled", base.CAC, "Cancelled")]):
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
                       label="%d months" % hm, edgecolor="white", linewidth=0.3)
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
            base.subcap(ax, "abcdef"[r * 3 + c], tech.capitalize() + ":" + base.NL + lab.lower(),
                        y=-0.26)
    axes[0, 0].set_ylim(0, 100)
    axes[1, 0].set_ylim(0, 26)
    fig.savefig(OUT / "FigureS2.png")
    plt.close(fig)
    print("Figure S2 written", OUT / "FigureS2.png")


if __name__ == "__main__":
    main()
