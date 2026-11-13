#!/usr/bin/env python3
"""Generate the decoding-latency sweep at d = 27 (bottom row of Fig. resource-result).

Copies examples/logicalT-exp/mean/d=27_{naive,autocorrected}.ft and changes only
the line `tdec: "390*tSE",` to `tdec: "<t>*tSE",`. That line must occur exactly
once in each source file.

Output: examples/logicalT-exp/latency/d=27_{naive,autocorrected}_tdec<t>.ft.
Earlier generated files in that folder are removed first, so the folder always
holds exactly the requested list (the collector runs every file in it).

Usage: generate.py [--tdec 0,50,...,1000,390]
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "mean"
IMPLEMENTATIONS = ("naive", "autocorrected")
SOURCE_LINE = '  tdec: "390*tSE",\n'
DEFAULT_TDEC = [*range(0, 1001, 50), 390]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tdec",
        default=",".join(str(t) for t in DEFAULT_TDEC),
        help="comma-separated decoding latencies in SE cycles (default 0,50,...,1000,390)",
    )
    args = parser.parse_args()
    values = parse_values(args.tdec)

    sources = {}
    for impl in IMPLEMENTATIONS:
        path = SOURCE_DIR / f"d=27_{impl}.ft"
        text = path.read_text(encoding="utf-8")
        count = text.count(SOURCE_LINE)
        if count != 1:
            raise SystemExit(f"{path}: expected the line {SOURCE_LINE.strip()!r} exactly once, found {count}")
        sources[impl] = text

    for old in SCRIPT_DIR.glob("d=27_*_tdec*.ft"):
        old.unlink()
    for t in values:
        for impl, text in sources.items():
            new_line = f'  tdec: "{t}*tSE",\n'
            (SCRIPT_DIR / f"d=27_{impl}_tdec{t}.ft").write_text(text.replace(SOURCE_LINE, new_line), encoding="utf-8")
    print(f"wrote {2 * len(values)} programs to {SCRIPT_DIR} (tdec = {', '.join(map(str, values))})")
    return 0


def parse_values(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise SystemExit(f"--tdec values must be non-negative integers, got {item!r}")
        if int(item) not in values:
            values.append(int(item))
    if not values:
        raise SystemExit("--tdec is empty")
    return sorted(values)


if __name__ == "__main__":
    raise SystemExit(main())
