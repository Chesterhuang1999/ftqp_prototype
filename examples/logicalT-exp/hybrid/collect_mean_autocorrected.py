#!/usr/bin/env python3
"""Sweep mixed distances for mean decoding time and autocorrected logical-T."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import itertools
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = None
MEAN_DIR = EXPERIMENT_DIR / "mean"
DEFAULT_OUTPUT = SCRIPT_DIR / "mean_autocorrected_distance_sweep.csv"
DISTANCES = (23, 25, 27, 29, 31)


@dataclass(frozen=True)
class RunResult:
    d1: int
    d2: int
    d3: int
    err_percent: float
    time_cycles: float
    space_cost: int


def main() -> int:
    global MEAN_DIR, ARTIFACTS_DIR
    parser = argparse.ArgumentParser(
        description=(
            "Generate and run all mean-decoding-time autocorrected logical-T "
            "mixed-distance cases."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--semantics",
        choices=("v1", "v2"),
        default="v2",
        help="error analysis passed to the estimator (v1 = numbers of the first paper version)",
    )
    parser.add_argument(
        "--keep-ft",
        action="store_true",
        help="Keep generated .ft files under examples/logicalT-exp/hybrid/generated.",
    )
    parser.add_argument("--mean-dir", type=Path, default=MEAN_DIR)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    MEAN_DIR = args.mean_dir.resolve()
    ARTIFACTS_DIR = args.artifacts_dir

    repo_root = find_repo_root(SCRIPT_DIR)
    hardware = read_hardware(MEAN_DIR / "d=31_autocorrected.ft")
    schemes = {d: read_scheme(d) for d in DISTANCES}

    generated_dir = SCRIPT_DIR / "generated"
    temp_context = None
    if args.keep_ft:
        generated_dir.mkdir(parents=True, exist_ok=True)
        work_dir = generated_dir
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="ftqp-logicalT-hybrid-")
        work_dir = Path(temp_context.name)

    try:
        results: list[RunResult] = []
        for d1, d2, d3 in itertools.product(DISTANCES, repeat=3):
            path = write_program(work_dir, hardware, schemes, d1, d2, d3)
            err_percent, time_cycles, space_cost = run_ftqp(repo_root, path, args.semantics)
            results.append(
                RunResult(
                    d1=d1,
                    d2=d2,
                    d3=d3,
                    err_percent=err_percent,
                    time_cycles=time_cycles,
                    space_cost=space_cost,
                )
            )
        write_csv(args.output, results)
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print(f"wrote {args.output}")
    return 0


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "ftqp_prototype" / "__main__.py").exists():
            return path
    raise SystemExit("could not find repository root containing ftqp_prototype/__main__.py")


def read_hardware(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"hardware\s+superconducting\s*=\s*\{.*?\};", text, re.DOTALL)
    if not match:
        raise SystemExit(f"{path}: could not find hardware block")
    return match.group(0)


def read_scheme(distance: int) -> str:
    path = MEAN_DIR / f"d={distance}_autocorrected.ft"
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"qecc\s+surface_code_{distance}\s+using\s+superconducting\s*=\s*\{{.*?\n\}};",
        text,
        re.DOTALL,
    )
    if not match:
        raise SystemExit(f"{path}: could not find surface_code_{distance}")
    return match.group(0)


def write_program(
    output_dir: Path,
    hardware: str,
    schemes: dict[int, str],
    d1: int,
    d2: int,
    d3: int,
) -> Path:
    used_distances = sorted({d1, d2, d3})
    content = "\n\n".join(
        [
            hardware,
            *(schemes[d] for d in used_distances),
            auto_corrected_implementation(d2, d3),
            program(d1),
        ]
    )
    path = output_dir / f"mean_autocorrected_d{d1}_{d2}_{d3}.ft"
    path.write_text(content, encoding="utf-8")
    return path


def auto_corrected_implementation(d2: int, d3: int) -> str:
    return f"""def auto_corrected_implementation(q: Codeblock) {{
    logical anc0 := new surface_code_{d2};
    logical anc1 := new surface_code_{d3};
    // call distill15to1(anc0);
    if M(ZZ)[q, anc0]; {{
        if M(ZY)[anc0, anc1]; {{
            H[q];H[q];H[q];H[q];H[q];H[q];
            H[q];H[q];H[q];H[q];H[q];H[q];
            H[q];H[q];H[q];H[q];H[q];H[q];
            H[q];H[q];H[q];H[q];H[q];H[q];
            H[q];H[q];H[q];H[q];H[q];H[q];
            H[q];H[q];H[q];H[q];H[q];H[q];
        }}
        then {{
            if M(X)[anc0]; then Z[q];
        }}
    }} then H[anc1];
    if M(Z)[anc1]; then skip; // controlled X/Z measurement.
    release(anc1);
    release(anc0);
}}"""


def program(d1: int) -> str:
    return f"""program {{
  logical q := new surface_code_{d1};
  call auto_corrected_implementation(q);
  release(q);
}}"""


def run_ftqp(repo_root: Path, path: Path, semantics: str = "v2") -> tuple[float, float, int]:
    completed = subprocess.run(
        [
            sys.executable, "-m", "ftqp_prototype", str(path),
            "--trace", "--no-circuit", "--semantics", semantics,
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if ARTIFACTS_DIR is not None:
        case = ARTIFACTS_DIR / path.stem
        case.mkdir(parents=True, exist_ok=False)
        shutil.copy2(path, case / "input.ft")
        (case / "run.log").write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr)
        (case / "run.json").write_text(json.dumps({"exit_code": completed.returncode, "semantics": semantics, "command": completed.args}))
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"{path.name} failed:\n{message}")
    return parse_cli_output(completed.stdout)


def parse_cli_output(output: str) -> tuple[float, float, int]:
    # Error of q when it is released: the last `q[0]=<eps>` in the trace before
    # "release q" (works for both v1 and v2 output).
    release_q = output.rfind("release q")
    trace = output[:release_q] if release_q >= 0 else output
    eps_matches = re.findall(r"\bq\[0\]=([0-9.eE+-]+)(?:@[A-Za-z_][A-Za-z0-9_]*)?", trace)
    if not eps_matches:
        raise SystemExit("could not find q[0] epsilon before release q")
    err_percent = float(eps_matches[-1]) * 100.0

    time_match = re.search(r"^time \(#SE cycles\)\s+([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    if not time_match:
        raise SystemExit("could not find final time (#SE cycles)")
    time_cycles = float(time_match.group(1))

    space_match = re.search(r"^space\s+([0-9]+)\s+physical qubits\s*$", output, re.MULTILINE)
    if not space_match:
        raise SystemExit("could not find final space cost")
    space_cost = int(space_match.group(1))

    return err_percent, time_cycles, space_cost


def write_csv(path: Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "d1_q",
                "d2_anc0",
                "d3_anc1",
                "error_percent",
                "time_se_cycles",
                "space_physical_qubits",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.d1,
                    result.d2,
                    result.d3,
                    f"{result.err_percent:.6f}",
                    f"{result.time_cycles:.6f}",
                    result.space_cost,
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
