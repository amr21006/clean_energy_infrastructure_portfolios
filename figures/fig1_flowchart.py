"""Figure 1: the research framework.

Operators' reports from the physical delivery system form a multistate digital shadow of the
portfolio. The shadow supports three prediction targets, each running in its own lane through
the prediction evidence to a governance response. Data stewardship adds disclosure, links the
records and repeats the test, feeding back into the shadow. Eighteen uniform boxes, one to
three keywords each, three per group; every arrow joins two specific boxes, or a box and the
digital shadow for the feedback.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import CB, STATE_COL

W_IN = 6.6
BW, BH = 1.30, 0.42                 # uniform box size (inches)
FH, GAP, TOP = 0.66, 0.26, 0.05     # frame height, gap between frames
FX0, FX1 = 0.06, 6.25               # frames span
LANE0, LANE_SPAN = 0.62, 5.50       # lane area inside the frames, right of the group label
X_LOOP = 6.42                       # feedback path, right of the frames
FS, LFS = 8.4, 7.8
INK, LINE = "#1A1A1A", "#3F3F3F"

GROUPS = [  # label, edge, tint
    ("Delivery\nsystem", "#33455F", "#EEF2F7"),
    ("Digital\nshadow", "#1F6F54", "#EBF5F1"),
    ("Prediction\ntargets", "#2F5E8C", "#EAF1F8"),
    ("Prediction\nevidence", "#8A5A20", "#FBF3E8"),
    ("Governance\nresponse", "#6B2740", "#F9EDF0"),
    ("Data\nstewardship", "#5A4A6A", "#F2EEF6"),
]
LANE_EDGE = [CB["blue"], STATE_COL["O"], STATE_COL["C"]]
LANE_FILL = ["#E3EDF6", "#E2F0E9", "#F7E6E6"]


def draw(out_dir: Path) -> int:
    n = len(GROUPS)
    h_in = 2 * TOP + n * FH + (n - 1) * GAP
    fig = plt.figure(figsize=(W_IN, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(h_in, 0)  # y runs downward, in inches
    ax.axis("off")
    count = [0]

    gap = (LANE_SPAN - 3 * BW) / 2
    lane_x = [LANE0 + k * (BW + gap) for k in range(3)]          # left edges
    lane_c = [x + BW / 2 for x in lane_x]                        # centers

    def ftop(k):
        return TOP + k * (FH + GAP)

    def btop(k):
        return ftop(k) + (FH - BH) / 2

    def frame(k):
        lab, ec, fc = GROUPS[k]
        ax.add_patch(FancyBboxPatch((FX0, ftop(k)), FX1 - FX0, FH,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=1))
        ax.text(FX0 + 0.25, ftop(k) + FH / 2, lab, ha="center", va="center", rotation=90,
                fontsize=LFS, fontweight="bold", color=ec, linespacing=1.05, zorder=2)

    def box(x, top, lab, ec, fc="white"):
        ax.add_patch(FancyBboxPatch((x, top), BW, BH, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3))
        ax.text(x + BW / 2, top + BH / 2, "\n".join(lab), ha="center", va="center",
                fontsize=FS, color=INK, linespacing=1.05, zorder=4)
        count[0] += 1

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                                     color=LINE, shrinkA=0, shrinkB=0, zorder=5,
                                     linestyle=(0, (3, 2)) if dashed else "-"))

    def line(pts, dashed=False):
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=LINE, linewidth=1.0,
                linestyle=(0, (3, 2)) if dashed else "-", solid_capstyle="butt", zorder=5)

    def chain(k, labels, ec=None, fc="white"):
        ec = ec or GROUPS[k][1]
        for x, lab in zip(lane_x, labels):
            box(x, btop(k), lab, ec, fc)
        y = btop(k) + BH / 2
        for a, b in zip(lane_x[:-1], lane_x[1:]):
            arrow((a + BW, y), (b, y))

    def lanes(k, labels):
        for x, lab, ec, fc in zip(lane_x, labels, LANE_EDGE, LANE_FILL):
            box(x, btop(k), lab, ec, fc)

    def down(k):
        """Lane by lane, from the boxes of group k to the boxes of group k + 1."""
        for c in lane_c:
            arrow((c, btop(k) + BH), (c, btop(k + 1)))

    def elbow(k):
        """From the last box of group k to the first box of group k + 1."""
        y = ftop(k) + FH + GAP / 2
        line([(lane_c[2], btop(k) + BH), (lane_c[2], y), (lane_c[0], y)])
        arrow((lane_c[0], y), (lane_c[0], btop(k + 1)))

    for k in range(n):
        frame(k)

    # Delivery system: units move through stages, and operators report them each month
    chain(0, [("Planned", "units"), ("Delivery", "stages"), ("Operator", "reports")])
    # Digital shadow: the monthly inventory becomes state histories
    chain(1, [("Monthly", "inventory"), ("Nine delivery", "states"), ("Transition", "histories")])
    elbow(0)
    # Prediction targets: one lane per governance question
    lanes(2, [("Next reported", "state"), ("12-month", "operation"), ("24-month", "cancellation")])
    y = ftop(1) + FH + GAP / 2
    line([(lane_c[2], btop(1) + BH), (lane_c[2], y)])
    line([(lane_c[0], y), (lane_c[2], y)])
    for c in lane_c:
        arrow((c, y), (c, btop(2)))
    # Prediction evidence and governance response, lane by lane
    lanes(3, [("Accurate", "transitions"), ("Accurate and", "calibrated"),
              ("Moderate", "discrimination")])
    down(2)
    lanes(4, [("Progress", "control"), ("Schedule", "assurance"), ("Screening and", "contingency")])
    down(3)
    # Data stewardship: added disclosure, linked to the inventory and tested again
    chain(5, [("Queue, offtake,", "permitting"), ("Record", "linkage"), ("Repeated", "testing")],
          fc="#ECE6F2")
    # every governance response feeds data stewardship: one merging connector into its first box
    y = ftop(4) + FH + GAP / 2
    for c in lane_c[1:]:
        line([(c, btop(4) + BH), (c, y)])
    line([(lane_c[0], y), (lane_c[2], y)])
    line([(lane_c[0], btop(4) + BH), (lane_c[0], y)])
    arrow((lane_c[0], y), (lane_c[0], btop(5)))
    # Feedback: stewardship extends the digital shadow
    yb = btop(5) + BH / 2
    ys = ftop(1) + FH / 2
    line([(lane_x[2] + BW, yb), (X_LOOP, yb), (X_LOOP, ys)], dashed=True)
    arrow((X_LOOP, ys), (FX1, ys), dashed=True)

    fig.savefig(out_dir / "Figure1.png", dpi=500)
    plt.close(fig)
    return count[0]
