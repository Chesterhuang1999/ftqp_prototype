#!/usr/bin/env python3
"""Collect the decoding-latency sweep at d = 27 (bottom row of Fig. resource-result).

Runs every examples/logicalT-exp/latency/d=27_{naive,autocorrected}_tdec<t>.ft
(see latency/generate.py) and writes one pgfplots table:

  \\pgfplotstableread{
  tdec errA timeA spaceA errB timeB spaceB
  ...
  }\\plotlatsweep

err = error of q[0] just before "release q", in %; time = final "time (#SE
cycles)"; space = peak physical qubits minus the 729-qubit data block of q.
A = naive, B = auto-corrected. Running and parsing reuse collect_results.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LATENCY_DIR = SCRIPT_DIR / "latency"
FILE_RE = re.compile(r"d=27_(naive|autocorrected)_tdec(\d+)\.ft")
DATA_BLOCK = 27 * 27

sys.path.insert(0, str(SCRIPT_DIR))
import collect_results as CR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--semantics", choices=("v1", "v2"), default="v2")
    parser.add_argument("--output", type=Path, required=True, help="where to write the table")
    args = parser.parse_args()
    CR.SEMANTICS = args.semantics

    repo_root = CR.find_repo_root(SCRIPT_DIR)
    results: dict[tuple[int, str], tuple[float, float, int]] = {}
    for path in sorted(LATENCY_DIR.glob("*.ft")):
        match = FILE_RE.fullmatch(path.name)
        if not match:
            raise SystemExit(f"unrecognized file in {LATENCY_DIR}: {path.name}")
        impl, tdec = match.group(1), int(match.group(2))
        results[(tdec, impl)] = run(repo_root, path)
    if not results:
        raise SystemExit(f"no programs in {LATENCY_DIR}; run latency/generate.py first")

    tdecs = sorted({t for t, _ in results})
    lines = [
        f"% decoding-latency sweep at d = 27 (error semantics: {args.semantics})",
        r"\pgfplotstableread{",
        "tdec  errA     timeA    spaceA  errB     timeB    spaceB",
    ]
    for t in tdecs:
        try:
            a, b = results[(t, "naive")], results[(t, "autocorrected")]
        except KeyError as exc:
            raise SystemExit(f"tdec={t}: missing naive or autocorrected program") from exc
        lines.append(f"{t:<5d} {a[0]:7.4f}  {a[1]:8.2f}  {a[2]:5d}   {b[0]:7.4f}  {b[1]:8.2f}  {b[2]:5d}")
    lines.append(r"}\plotlatsweep")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def run(repo_root: Path, path: Path) -> tuple[float, float, int]:
    output = CR.run_ftqp(repo_root, path)
    err_percent = CR.parse_final_q_epsilon(output) * 100.0
    time_match = CR.FTQP_SE_CYCLES_RE.search(output)
    space_match = CR.FTQP_SPACE_RE.search(output)
    if not time_match or not space_match:
        raise SystemExit(f"{path.name}: could not parse time or space")
    return err_percent, float(time_match.group(1)), int(space_match.group(1)) - DATA_BLOCK


if __name__ == "__main__":
    raise SystemExit(main())
