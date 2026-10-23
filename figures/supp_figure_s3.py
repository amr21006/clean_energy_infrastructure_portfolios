# -*- coding: utf-8 -*-
"""Supplementary Figure S3: feature importance across the estimands, with long predictor labels
wrapped so that the labels of one panel stay clear of the bars of the panel to its left.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_supp_figures as base

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# predictor labels as named in Supplementary Tables S4 and S22
LABELS = {**base.FEAT,
          "concurrent_entity_pipeline_units": "Entity units under development",
          "state_code": "State or territory"}


def wrap(label: str) -> str:
    return "\n".join(textwrap.wrap(label, 20))


def main() -> None:
    ns = pd.read_csv(base.TAB / "next_state_boosting_feature_importance.csv")
    op = pd.read_csv(base.TAB / "operation_12m_feature_importance.csv")
    ca = pd.read_csv(base.TAB / "cancellation_24m_feature_importance.csv")

    def norm(d):
        out = []
        for _, g in d.groupby("algorithm"):
            g = g.copy()
            tot = g.importance.sum()
            g["rel"] = g.importance / tot if tot > 0 else 0.0
            out.append(g)
        z = pd.concat(out)
        return z.groupby("feature", as_index=False)["rel"].mean().sort_values("rel")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.3))
    fig.subplots_adjust(wspace=1.0)
    for ax, d, t, c, lab in [
            (axes[0], norm(ns), "Next reported state", base.CB["blue"], "a"),
            (axes[1], norm(op), "Operation within 12 months", base.OPC, "b"),
            (axes[2], norm(ca), "Cancellation within 24 months", base.CAC, "c")]:
        d = d.tail(11)
        y = np.arange(len(d))
        ax.barh(y, d.rel.values * 100, color=c, height=0.68, edgecolor="white",
                linewidth=0.3)
        ax.set_yticks(y, [wrap(LABELS.get(f, f)) for f in d.feature], fontsize=6.8,
                      linespacing=0.95)
        ax.set_xlabel("Mean relative importance (%)")
        base.subcap(ax, lab, t, y=-0.18)
    fig.savefig(OUT / "FigureS3.png")
    plt.close(fig)
    print("Figure S3 written", OUT / "FigureS3.png")


if __name__ == "__main__":
    main()
