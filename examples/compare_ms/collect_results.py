#!/usr/bin/env python3
"""Collect Microsoft QRE and FTQP estimates for compare_ms benchmarks."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "output.txt"
BENCHMARKS = ("qft", "bv", "ripple")
SIZES = (100, 200, 300, 400)


@dataclass(frozen=True)
class Estimate:
    physical_qubits: str
    time: str
    extra_time: str = ""


@dataclass(frozen=True)
class Row:
    benchmark: str
    qubits: int
    microsoft: Estimate
    ftqp: Estimate


def main() -> int:
    progress("locating repository root...")
    repo_root = find_repo_root(SCRIPT_DIR)
    progress(f"repository root: {repo_root}")
    progress("checking Microsoft QDK estimator Python...")
    estimator_python = find_qdk_python()
    if estimator_python is None:
        progress("Microsoft QDK estimator Python: not found; Microsoft estimates will be marked ERROR")
    else:
        progress(f"Microsoft QDK estimator Python: {estimator_python}")
    rows = collect_rows(repo_root, estimator_python)
    progress(f"writing LaTeX output to {OUTPUT_PATH}...")
    OUTPUT_PATH.write_text(render_latex(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


def progress(message: str) -> None:
    print(message, flush=True)


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "ftqp_prototype" / "__main__.py").exists():
            return path
    raise SystemExit("could not find repository root containing ftqp_prototype/__main__.py")


def find_qdk_python() -> Path | None:
    candidates: list[Path] = []
    env_python = os.environ.get("QDK_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend(
        [
            Path("/tmp/ftqp-qdk-venv/bin/python"),
            Path(sys.executable),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        completed = subprocess.run(
            [str(candidate), "-c", "import qdk.qre"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            return candidate
    return None


def collect_rows(repo_root: Path, estimator_python: Path | None) -> list[Row]:
    rows: list[Row] = []
    total = len(BENCHMARKS) * len(SIZES)
    completed = 0
    progress(f"collecting {total} benchmark results...")
    for benchmark in BENCHMARKS:
        for size in SIZES:
            completed += 1
            label = f"{benchmark}{size}"
            qasm_path = SCRIPT_DIR / f"{benchmark}{size}.qasm"
            ft_path = SCRIPT_DIR / f"{benchmark}{size}.ft"
            if not qasm_path.exists():
                raise SystemExit(f"missing QASM file: {qasm_path}")
            if not ft_path.exists():
                raise SystemExit(f"missing FT file: {ft_path}")
            progress(f"[{completed}/{total}] {label}: running Microsoft estimator...")
            microsoft = run_microsoft_estimator(estimator_python, qasm_path)
            progress(f"[{completed}/{total}] {label}: Microsoft estimator {estimate_summary(microsoft)}")
            progress(f"[{completed}/{total}] {label}: running FTQP...")
            ftqp = run_ftqp(repo_root, ft_path)
            progress(f"[{completed}/{total}] {label}: FTQP {estimate_summary(ftqp)}")
            rows.append(
                Row(
                    benchmark=benchmark,
                    qubits=size,
                    microsoft=microsoft,
                    ftqp=ftqp,
                )
            )
    return rows


def estimate_summary(estimate: Estimate) -> str:
    if estimate.physical_qubits == "ERROR" or estimate.time == "ERROR":
        return f"ERROR: {estimate.extra_time}"
    details = f"ok: {estimate.physical_qubits} physical qubits, time {estimate.time} ns"
    if estimate.extra_time:
        details += f", {estimate.extra_time} SE cycles"
    return details


def run_microsoft_estimator(estimator_python: Path | None, qasm_path: Path) -> Estimate:
    if estimator_python is None:
        return Estimate("ERROR", "ERROR", "qdk module not found")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "ftqp-matplotlib"))
    completed = subprocess.run(
        [str(estimator_python), str(SCRIPT_DIR / "estimator.py"), str(qasm_path)],
        cwd=SCRIPT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return Estimate("ERROR", "ERROR", first_error_line(completed))
    try:
        return parse_microsoft_output(completed.stdout)
    except ValueError as exc:
        return Estimate("ERROR", "ERROR", str(exc))


def parse_microsoft_output(output: str) -> Estimate:
    qubits_match = re.search(r"^\s*qubits\s*=\s*([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    runtime_match = re.search(r"^\s*runtime_ns\s*=\s*([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    if qubits_match is None:
        raise ValueError("missing Microsoft qubits field")
    if runtime_match is None:
        raise ValueError("missing Microsoft runtime_ns field")
    return Estimate(qubits_match.group(1), runtime_match.group(1))


def run_ftqp(repo_root: Path, ft_path: Path) -> Estimate:
    rel_path = ft_path.relative_to(repo_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftqp_prototype",
            str(rel_path),
            "--trace",
            "--no-circuit",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return Estimate("ERROR", "ERROR", first_error_line(completed))
    try:
        return parse_ftqp_output(completed.stdout)
    except ValueError as exc:
        return Estimate("ERROR", "ERROR", str(exc))


def parse_ftqp_output(output: str) -> Estimate:
    time_match = re.search(r"^time \(absolute\)\s+([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    cycles_match = re.search(r"^time \(#SE cycles\)\s+([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    space_match = re.search(r"^space\s+([0-9]+)\s+physical qubits\s*$", output, re.MULTILINE)
    if time_match is None:
        raise ValueError("missing FTQP time (absolute) field")
    if space_match is None:
        raise ValueError("missing FTQP space field")
    extra = cycles_match.group(1) if cycles_match is not None else ""
    return Estimate(space_match.group(1), time_match.group(1), extra)


def first_error_line(completed: subprocess.CompletedProcess[str]) -> str:
    message = completed.stderr.strip() or completed.stdout.strip()
    if not message:
        return f"exit status {completed.returncode}"
    return message.splitlines()[0]


def render_latex(rows: list[Row]) -> str:
    colors = {
        100: "qubitsHundred",
        200: "qubitsTwoHundred",
        300: "qubitsThreeHundred",
        400: "qubitsFourHundred",
    }
    lines = [
        r"\begin{groupplot}[",
        r"  group style={",
        r"    group name=resourceplots,",
        r"    group size=3 by 1,",
        r"    horizontal sep=2.0em,",
        r"  },",
        r"  width=0.25*396pt,",
        r"  height=0.25*396pt,",
        r"  xlabel={Time cost (ms)},",
        r"  ylabel={Space cost (qubits)},",
        r"  xlabel style={yshift=0.5em},",
        r"  ylabel style={yshift=-0.5em},",
        r"  grid=both,",
        r"  % grid style={dashed,gray!30},",
        r"  scaled x ticks=false,",
        r"  scaled y ticks=true,",
        r"  tick label style={font=\tiny},",
        r"  label style={font=\tiny},",
        r"  title style={font=\scriptsize},",
        r"]",
    ]

    by_benchmark = {benchmark: [row for row in rows if row.benchmark == benchmark] for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        lines.extend(render_axis(benchmark, by_benchmark[benchmark], colors))

    lines.extend(["", r"\end{groupplot}"])
    return "\n".join(lines)


def render_axis(
    benchmark: str,
    rows: list[Row],
    colors: dict[int, str],
) -> list[str]:
    titles = {
        "qft": "QFT",
        "bv": "BV",
        "ripple": "Ripple-Carry Adder",
    }
    title = titles[benchmark]
    lines = [
        r"\nextgroupplot[",
        rf"  title={{{title}}},",
    ]
    lines.append(r"]")

    for row in sorted(rows, key=lambda item: item.qubits):
        color = colors[row.qubits]
        if (
            row.microsoft.physical_qubits == "ERROR"
            or row.microsoft.time == "ERROR"
            or row.ftqp.physical_qubits == "ERROR"
            or row.ftqp.time == "ERROR"
        ):
            lines.append(f"% {row.benchmark}{row.qubits} skipped")
            continue
        lines.extend(
            [
                rf"\addplot+[compareLine, color={color}, forget plot] coordinates {{",
                f"  ({milliseconds(row.microsoft.time)},{row.microsoft.physical_qubits})",
                f"  ({milliseconds(row.ftqp.time)},{row.ftqp.physical_qubits})",
                r"};",
                rf"\addplot+[msPoint, color={color}, forget plot] coordinates {{",
                f"  ({milliseconds(row.microsoft.time)},{row.microsoft.physical_qubits})",
                r"};",
                rf"\addplot+[ourPoint, color={color}, forget plot] coordinates {{",
                f"  ({milliseconds(row.ftqp.time)},{row.ftqp.physical_qubits})",
                r"};",
            ]
        )

    return lines


def milliseconds(value: str) -> str:
    return f"{float(value) / 1_000_000:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
