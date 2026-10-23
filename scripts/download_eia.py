"""Download the EIA source files listed in the download manifest and verify their hashes.

Usage:
    python scripts/download_eia.py --manifest data/raw/download_manifest.csv --raw-dir data/raw
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
from pathlib import Path

SUBDIR = {"EIA-860M": "eia860m", "EIA-860": "eia860_annual", "EIA-reference": "eia860m"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/download_manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    failures = 0
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = args.raw_dir / SUBDIR.get(row["series"], "other") / row["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                request = urllib.request.Request(row["source_url"],
                                                 headers={"User-Agent": "research-reproduction"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    target.write_bytes(response.read())
            ok = sha256(target) == row["sha256"]
            failures += not ok
            print(("OK   " if ok else "DIFF ") + str(target))
    print(f"{failures} file(s) differ from the recorded hashes."
          " EIA occasionally reissues files; differing files may change results slightly.")


if __name__ == "__main__":
    main()
