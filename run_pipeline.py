from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eia_delivery_twin.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the EIA clean-infrastructure delivery digital twin."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/delivery_twin")
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument(
        "--rebuild-panel",
        action="store_true",
        help="Ignore processed caches and rebuild the raw longitudinal panel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        rebuild_panel=args.rebuild_panel,
    )
