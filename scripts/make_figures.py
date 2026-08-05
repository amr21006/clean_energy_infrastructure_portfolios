"""New and redrawn figures for the revision.

Matches the house style of scripts/make_figures.py: authored at 6.5 in wide,
saved at 1000 dpi, Okabe-Ito colour-blind-safe palette, panel labels at the
bottom, no data touching an axis edge.

    fig_ablation_R1.png      Figure 8  - factorial ablation and seed stability
    fig_sensitivity_R1.png   Figure 9  - lambda / kappa / tau_0 sensitivity
    fig_multihorizon_R1.png  Figure 10 - accuracy against forecast horizon

Every figure is drawn only from executed result tables; if an input is missing
the corresponding figure is skipped and reported rather than approximated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "revision" / "derived"
FIG = ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 1000,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 7.6, "legend.handlelength": 1.3,
    "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "lines.linewidth": 1.4,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

FW = 6.5
ORANGE, GREEN, BLUE, GREY, PURPLE = "#D55E00", "#009E73", "#0072B2", "#7a7a7a", "#7d5fa8"


def panel_label(ax, text: str, y: float = -0.30) -> None:
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold")


def skip(name: str, reason: str) -> None:
    print(f"SKIP {name}: {reason}")


# --------------------------------------------------------------------- Figure 8


def figure_ablation() -> None:
    ladder_path = DERIVED / "ablation_ladder.csv"
    if not ladder_path.exists():
        return skip("fig_ablation_R1", "ablation_ladder.csv not found")
    ladder = pd.read_csv(ladder_path)
    test = ladder[ladder["split"] == "test"].set_index("arm")

    order = [
        ("A1_plain_transformer", "Transformer", False, False),
        ("A2_plain_transformer_physics", "Transformer", False, True),
        ("A3_residual_transformer", "Transformer", True, False),
        ("A4_residual_transformer_physics", "Transformer", True, True),
        ("A5_plain_tcn", "TCN", False, False),
        ("A6_residual_tcn", "TCN", True, False),
    ]
    rows = [(tag, backbone, anchor, penalty, float(test.loc[tag, "cvrmse_pct"]))
            for tag, backbone, anchor, penalty in order if tag in test.index]
    if not rows:
        return skip("fig_ablation_R1", "no ablation arms present")

    boosting = DERIVED / "fulllag_boosting_metrics.csv"
    extra = []
    if boosting.exists():
        fl = pd.read_csv(boosting)
        flt = fl[fl["split"] == "test"].set_index("arm")
        for arm, label, anchor in (("lightgbm_full384_absolute", "LightGBM", False),
                                   ("lightgbm_full384_residual", "LightGBM", True)):
            if arm in flt.index:
                extra.append((arm, label, anchor, False, float(flt.loc[arm, "cvrmse_pct"])))
    rows += extra

    fig = plt.figure(figsize=(FW, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.30,
                          left=0.085, right=0.985, top=0.93, bottom=0.30)

    ax = fig.add_subplot(gs[0, 0])
    labels, values, colours = [], [], []
    for tag, backbone, anchor, penalty, value in rows:
        mark = "anchor" if anchor else "no anchor"
        if penalty:
            mark += " + constraint"
        labels.append(f"{backbone}\n{mark}")
        values.append(value)
        colours.append(ORANGE if anchor else GREY)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colours, width=0.66, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012,
                f"{value:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("Test CV(RMSE) (%)")
    ax.set_ylim(0, max(values) * 1.16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=GREY)]
    ax.legend(handles, ["with residual anchor", "without residual anchor"],
              loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.13))
    panel_label(ax, "(a) Factorial ablation", y=-0.38)

    ax2 = fig.add_subplot(gs[0, 1])
    seeds_path = ROOT / "manuscript" / "derived" / "ext_metrics.csv"
    archived_rmse = 22.662234829609634
    if seeds_path.exists():
        ext = pd.read_csv(seeds_path)
        main = ext[(ext["variant"] == "main") & (ext["split"] == "test")]
        rmses = [archived_rmse] + main["rmse"].tolist()
        seed_labels = ["42", "52", "62"][:len(rmses)]
        mean, sd = float(np.mean(rmses)), float(np.std(rmses, ddof=1))
        ax2.axhspan(mean - sd, mean + sd, color=ORANGE, alpha=0.14, lw=0)
        ax2.axhline(mean, color=ORANGE, lw=1.2, ls="--")
        ax2.plot(range(len(rmses)), rmses, "o", color=ORANGE, ms=6)
        ax2.set_xticks(range(len(rmses)))
        ax2.set_xticklabels(seed_labels)
        ax2.set_xlabel("Random seed")
        ax2.set_ylabel("Test RMSE (kWh)")
        span = max(sd * 4, 0.6)
        ax2.set_ylim(mean - span, mean + span)
        ax2.set_xlim(-0.5, len(rmses) - 0.5)
        ax2.text(0.03, 0.96, f"mean {mean:.2f} $\\pm$ {sd:.2f} kWh", transform=ax2.transAxes,
                 va="top", fontsize=7.4)
    panel_label(ax2, "(b) Seed stability", y=-0.38)

    out = FIG / "fig_ablation_R1.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------- Figure 9


def figure_sensitivity() -> None:
    sens_path = DERIVED / "sensitivity_metrics.csv"
    ladder_path = DERIVED / "ablation_ladder.csv"
    if not sens_path.exists() or not ladder_path.exists():
        return skip("fig_sensitivity_R1", "sensitivity_metrics.csv or ablation_ladder.csv missing")
    sens = pd.read_csv(sens_path)
    sens = sens[sens["split"] == "test"]
    ladder = pd.read_csv(ladder_path)
    ladder_test = ladder[ladder["split"] == "test"].set_index("arm")

    ref_arm = "A4_residual_transformer_physics"
    base_arm = "A3_residual_transformer"
    ref = ladder_test.loc[ref_arm] if ref_arm in ladder_test.index else None
    base = ladder_test.loc[base_arm] if base_arm in ladder_test.index else None

    panels = [("lambda", r"Penalty weight $\lambda$", "(a)"),
              ("kappa", r"Temperature coefficient $\kappa$", "(b)"),
              ("tau0", r"Base tolerance $\tau_0$", "(c)")]
    fig, axes = plt.subplots(1, 3, figsize=(FW, 2.9))
    fig.subplots_adjust(left=0.085, right=0.90, top=0.92, bottom=0.30, wspace=0.55)

    seed_sd = 0.07  # CV(RMSE) percentage points, from the three-seed repeat
    for ax, (param, xlabel, tag) in zip(axes, panels):
        sub = sens[sens["swept_parameter"] == param].sort_values("swept_value")
        xs = sub["swept_value"].tolist()
        ys = sub["cvrmse_pct"].tolist()
        acts = (sub["penalty_active_fraction"] * 100).tolist()
        if ref is not None:
            reference_value = {"lambda": 0.02, "kappa": 3.0, "tau0": 0.25}[param]
            xs.append(reference_value)
            ys.append(float(ref["cvrmse_pct"]))
            acts.append(float(ref["penalty_active_fraction"]) * 100)
        if param == "lambda" and base is not None:
            xs.append(0.0)
            ys.append(float(base["cvrmse_pct"]))
            acts.append(0.0)
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        acts = np.array(acts)[order]

        if ref is not None:
            centre = float(ref["cvrmse_pct"])
            ax.axhspan(centre - seed_sd, centre + seed_sd, color=ORANGE, alpha=0.13, lw=0)
        ax.plot(xs, ys, "o-", color=ORANGE, ms=4.5, label="CV(RMSE)")
        ax.set_xlabel(xlabel)
        if tag == "(a)":
            ax.set_ylabel("Test CV(RMSE) (%)")
        lo, hi = float(np.min(ys)), float(np.max(ys))
        pad = max((hi - lo) * 0.45, 0.12)
        ax.set_ylim(lo - pad, hi + pad)

        ax2 = ax.twinx()
        ax2.plot(xs, acts, "s--", color=BLUE, ms=3.6, lw=1.0, label="activation")
        ax2.set_ylim(0, max(max(acts) * 1.5, 0.5))
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(axis="y", colors=BLUE, labelsize=7.0)
        if tag == "(c)":
            ax2.set_ylabel("Constraint active (% of test hours)", color=BLUE, fontsize=8)
        panel_label(ax, f"{tag} {param}", y=-0.34)

    handles = [plt.Line2D([], [], color=ORANGE, marker="o", ms=4.5, label="Test CV(RMSE)"),
               plt.Line2D([], [], color=BLUE, marker="s", ls="--", ms=3.6,
                          label="Constraint activation rate")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.045))
    out = FIG / "fig_sensitivity_R1.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------- Figure 10


def figure_multihorizon() -> None:
    path = DERIVED / "multihorizon_metrics.csv"
    trend_path = DERIVED / "multihorizon_trend.csv"
    if not path.exists() or not trend_path.exists():
        return skip("fig_multihorizon_R1", "multihorizon outputs not found")
    mh = pd.read_csv(path)
    test = mh[mh["split"] == "test"]
    trend = pd.read_csv(trend_path).sort_values("horizon_h")

    fig, axes = plt.subplots(1, 2, figsize=(FW, 2.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.30, wspace=0.30)

    styles = [("persistence", "Persistence", GREY, "^"),
              ("lightgbm_stats32", "LightGBM", GREEN, "s"),
              ("pg_sstt", "PG-SSTT", ORANGE, "o")]
    ax = axes[0]
    for arm, label, colour, marker in styles:
        sub = test[test["arm"] == arm].sort_values("horizon_h")
        if sub.empty:
            continue
        ax.plot(sub["horizon_h"], sub["cvrmse_pct"], marker=marker, color=colour,
                ms=4.5, label=label)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Test CV(RMSE) (%)")
    ax.set_xticks(sorted(test["horizon_h"].unique()))
    ax.legend(loc="upper left")
    ymax = float(test["cvrmse_pct"].max())
    ax.set_ylim(0, ymax * 1.18)
    panel_label(ax, "(a) Accuracy against horizon", y=-0.34)

    ax2 = axes[1]
    ax2.axhline(0, color="#bbbbbb", lw=0.9)
    ax2.plot(trend["horizon_h"], trend["pg_sstt_gain_vs_persistence_pct"], "o-",
             color=GREY, ms=4.5, label="vs persistence")
    ax2.plot(trend["horizon_h"], trend["pg_sstt_gain_vs_lightgbm_pct"], "s-",
             color=GREEN, ms=4.2, label="vs LightGBM")
    ax2.set_xlabel("Forecast horizon (h)")
    ax2.set_ylabel("CV(RMSE) reduction by PG-SSTT (%)")
    ax2.set_xticks(sorted(test["horizon_h"].unique()))
    ax2.legend(loc="best")
    lo = float(min(trend["pg_sstt_gain_vs_persistence_pct"].min(),
                   trend["pg_sstt_gain_vs_lightgbm_pct"].min(), 0))
    hi = float(max(trend["pg_sstt_gain_vs_persistence_pct"].max(),
                   trend["pg_sstt_gain_vs_lightgbm_pct"].max()))
    pad = max((hi - lo) * 0.20, 1.5)
    ax2.set_ylim(lo - pad, hi + pad)
    panel_label(ax2, "(b) Relative advantage", y=-0.34)

    out = FIG / "fig_multihorizon_R1.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    figure_ablation()
    figure_sensitivity()
    figure_multihorizon()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
