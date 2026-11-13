#!/usr/bin/env python3
"""Collect logicalT-exp results from the 99.9 and mean folders."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "output.txt"
DATASETS = (
    ("99.9", "99.9th percentile", "percentile"),
    ("mean", "mean", "mean"),
)
NUMBER_RE = r"[0-9.eE+-]+"
FTQP_Q_EPSILON_RE = re.compile(rf"\bq\[0\]=({NUMBER_RE})(?:@[A-Za-z_][A-Za-z0-9_]*)?")
FTQP_Q_EPSILON_LEGACY_RE = re.compile(rf"\bq\[0\]\s+({NUMBER_RE})\s+surface_code_")
FTQP_SE_CYCLES_RE = re.compile(rf"^time \(#SE cycles\)\s+({NUMBER_RE})\s*$", re.MULTILINE)
FTQP_SPACE_RE = re.compile(r"^space\s+([0-9]+)\s+physical qubits\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Result:
    d: int
    variant: str
    err_percent: float
    time_cycles: int
    space_cost: int
    dec_cycles: int | None = None


SEMANTICS = "v2"
RAW_DIR = None


def main() -> int:
    global SEMANTICS, RAW_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantics",
        choices=("v1", "v2"),
        default="v2",
        help="error analysis passed to the estimator (v1 = numbers of the first paper version)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"where to write the pgfplots tables. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(name for name, _, _ in DATASETS),
        help="comma-separated dataset folders to collect (default: 99.9,mean)",
    )
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR, help="Directory containing mean/ and 99.9/ inputs")
    parser.add_argument("--raw-dir", type=Path, help="Save per-case estimator output")
    args = parser.parse_args()
    RAW_DIR = args.raw_dir
    SEMANTICS = args.semantics
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    known = [name for name, _, _ in DATASETS]
    if not selected or any(name not in known for name in selected):
        raise SystemExit(f"--datasets must be a comma-separated subset of {','.join(known)}")

    repo_root = find_repo_root(SCRIPT_DIR)
    sections: list[str] = [f"% paste into exp.tex (error semantics: {SEMANTICS})"]
    for folder_name, label, macro_suffix in DATASETS:
        if folder_name not in selected:
            continue
        folder = args.data_root.resolve() / folder_name
        if not folder.is_dir():
            raise SystemExit(f"missing dataset folder: {folder}")
        results = collect_dataset(repo_root, folder)
        sections.append(render_tables(results, label, macro_suffix))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "ftqp_prototype" / "__main__.py").exists():
            return path
    raise SystemExit("could not find repository root containing ftqp_prototype/__main__.py")


def collect_dataset(repo_root: Path, folder: Path) -> dict[tuple[int, str, str], Result]:
    files = sorted(folder.glob("*.ft"))
    if not files:
        raise SystemExit(f"no .ft files found in {folder}")

    results: dict[tuple[int, str, str], Result] = {}
    for path in files:
        kind, d, variant = classify(path)
        stdout = run_ftqp(repo_root, path)
        err_percent, time_cycles, total_space = parse_cli_output(stdout)
        dec_cycles = parse_dec_cycles(path) if kind == "odd" else None
        results[(d, variant, kind)] = Result(
            d=d,
            variant=variant,
            err_percent=err_percent,
            time_cycles=time_cycles,
            space_cost=total_space - space_baseline(d, kind),
            dec_cycles=dec_cycles,
        )
    return results


def classify(path: Path) -> tuple[str, int, str]:
    odd = re.fullmatch(r"d=(\d+)_(naive|autocorrected)\.ft", path.name)
    if odd:
        return "odd", int(odd.group(1)), variant_name(odd.group(2))

    hybrid = re.fullmatch(r"hybrid(\d{2})(\d{2})_(naive|autocorrected)\.ft", path.name)
    if hybrid:
        low = int(hybrid.group(1))
        high = int(hybrid.group(2))
        if high != low + 2:
            raise SystemExit(f"{path.name}: expected hybrid(d-1)(d+1) naming")
        return "even", low + 1, variant_name(hybrid.group(3))

    raise SystemExit(f"unrecognized logicalT-exp filename: {path.name}")


def variant_name(implementation: str) -> str:
    if implementation == "naive":
        return "A"
    if implementation == "autocorrected":
        return "B"
    raise SystemExit(f"unknown implementation suffix: {implementation}")


def run_ftqp(repo_root: Path, path: Path) -> str:
    rel_path = path.relative_to(repo_root)
    command = [
        sys.executable, "-m", "ftqp_prototype", str(rel_path),
        "--trace", "--no-circuit", "--semantics", SEMANTICS,
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if RAW_DIR is not None:
        raw = RAW_DIR / path.parent.name
        raw.mkdir(parents=True, exist_ok=True)
        (raw / (path.stem + ".stdout.txt")).write_text(completed.stdout)
        (raw / (path.stem + ".stderr.txt")).write_text(completed.stderr)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"{path} failed:\n{message}")
    return completed.stdout


def parse_cli_output(output: str) -> tuple[float, int, int]:
    err_percent = parse_final_q_epsilon(output) * 100.0

    time_match = FTQP_SE_CYCLES_RE.search(output)
    if not time_match:
        raise SystemExit("could not find final time (#SE cycles)")
    time_cycles = int(round(float(time_match.group(1))))

    space_match = FTQP_SPACE_RE.search(output)
    if not space_match:
        raise SystemExit("could not find final space cost")
    total_space = int(space_match.group(1))

    return err_percent, time_cycles, total_space


def parse_final_q_epsilon(output: str) -> float:
    release_q = output.rfind("release q")
    trace_before_release = output[:release_q] if release_q >= 0 else output
    eps_matches = FTQP_Q_EPSILON_RE.findall(trace_before_release)
    if eps_matches:
        return float(eps_matches[-1])

    legacy_matches = FTQP_Q_EPSILON_LEGACY_RE.findall(trace_before_release)
    if legacy_matches:
        return float(legacy_matches[-1])

    raise SystemExit("could not find q[0] epsilon before release q")


def parse_dec_cycles(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    q_scheme_match = re.search(r"program\s*\{.*?logical\s+q\s*:=\s*new\s+(\w+)", text, re.DOTALL)
    if not q_scheme_match:
        raise SystemExit(f"{path.name}: could not find q allocation scheme")
    scheme_name = q_scheme_match.group(1)

    scheme_match = re.search(
        rf"qecc\s+{re.escape(scheme_name)}\b.*?tdec\s*:\s*\"([^\"]+)\"",
        text,
        re.DOTALL,
    )
    if not scheme_match:
        raise SystemExit(f"{path.name}: could not find tdec for {scheme_name}")

    expr = scheme_match.group(1).replace(" ", "")
    coeff_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\*tSE", expr)
    if coeff_match:
        return int(round(float(coeff_match.group(1))))
    number_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", expr)
    if number_match:
        return int(round(float(number_match.group(1))))
    raise SystemExit(f"{path.name}: unsupported tdec expression {expr!r}")


def space_baseline(d: int, kind: str) -> int:
    if kind == "even":
        return (d + 1) * (d + 1)
    return d * d


def render_tables(results: dict[tuple[int, str, str], Result], label: str, macro_suffix: str) -> str:
    odd_ds = sorted({d for d, _, kind in results if kind == "odd"})
    even_ds = sorted({d for d, _, kind in results if kind == "even"})

    lines: list[str] = []
    lines.append(f"% data for {label} decoding time")
    lines.append(r"\pgfplotstableread{")
    lines.append("d   dec")
    for d in odd_ds:
        a = require(results, d, "A", "odd")
        dec = a.dec_cycles if a.dec_cycles is not None else 0
        lines.append(f"{d:<3d} {dec}")
    lines.append(rf"}}\plottdec{macro_suffix}")
    lines.append("")

    lines.append(r"\pgfplotstableread{")
    lines.append("d   errA   timeA  spaceA  errB   timeB  spaceB")
    for d in odd_ds:
        lines.append(format_row(d, require(results, d, "A", "odd"), require(results, d, "B", "odd")))
    lines.append(rf"}}\plotdataodd{macro_suffix}")
    lines.append("")

    lines.append(r"\pgfplotstableread{")
    lines.append("d   errA   timeA  spaceA  errB   timeB  spaceB")
    for d in even_ds:
        lines.append(format_row(d, require(results, d, "A", "even"), require(results, d, "B", "even")))
    lines.append(rf"}}\plotdataeven{macro_suffix}")
    return "\n".join(lines)


def require(results: dict[tuple[int, str, str], Result], d: int, variant: str, kind: str) -> Result:
    try:
        return results[(d, variant, kind)]
    except KeyError as exc:
        raise SystemExit(f"missing result for d={d}, variant={variant}, kind={kind}") from exc


def format_row(d: int, a: Result, b: Result, dec: int | None = None) -> str:
    row = (
        f"{d:<3d} "
        f"{a.err_percent:5.2f}  {a.time_cycles:<5d}  {a.space_cost:<6d} "
        f"{b.err_percent:5.2f}  {b.time_cycles:<5d}  {b.space_cost:<6d}"
    )
    if dec is not None:
        row += f" {dec}"
    return row.rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
