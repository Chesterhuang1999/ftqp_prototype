"""Command line interface for the FTQP prototype."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TextIO

from ftqp_prototype.compiler import Compiler
from ftqp_prototype.parser import parse_file
from ftqp_prototype.parser.ft_parser import ParseError
from ftqp_prototype.parser.lexer import LexError
from ftqp_prototype.resource import ResourceEstimator
from ftqp_prototype.resource.estimator import (
    ErrorContext,
    ResourceEstimate,
    ResourceEstimationError,
    TraceEntry,
)
from ftqp_prototype.resource.distill import format_axis, format_qubit
from ftqp_prototype.resource.estimator_distill import DistillationEstimate, DistillationEstimator
from ftqp_prototype.resource.estimator_v2 import ResourceEstimateV2, ResourceEstimatorV2
from ftqp_prototype.typechecker import TypeChecker, ValidationError
from ftqp_prototype.visualization import render_ascii, render_dot, render_tree_ascii, render_tree_dot

_CALL_RENAMED_REGISTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)__call([0-9]+)_([0-9]+)(?:_([0-9]+))?\b"
)
_TRACE_EVENT_WIDTH = 32
_TRACE_CONTEXT_WIDTH = 34


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse, validate, compile, and estimate .ft programs.")
    parser.add_argument("file", help="path to a .ft program")
    parser.add_argument("--dot", action="store_true", help="render Graphviz DOT instead of ASCII")
    parser.add_argument("--no-circuit", action="store_true", help="suppress compiled circuit visualization")
    parser.add_argument("--no-estimate", action="store_true", help="suppress resource estimate")
    parser.add_argument("--trace", action="store_true", help="print the error-context trace")
    parser.add_argument("--tree", action="store_true", help="show the structural compiler tree instead of the circuit timeline")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="parse and type-check without compiling or estimating",
    )
    parser.add_argument(
        "--semantics",
        choices=("v1", "v2", "distill"),
        default="v2",
        help="error analysis: v2 = source-aware rules (default), v1 = scalar rules of the "
        "first paper, distill = distillation estimator of §5.5 (acceptance and accepted error)",
    )
    parser.add_argument(
        "--outputs",
        help="v2: comma-separated program-level registers whose joint error is reported "
        "(default: registers released after all operations, except the reserved name 'anc')",
    )
    parser.add_argument(
        "--no-distill",
        action="store_true",
        help="v2: do not detect distillation protocols (no postselection model)",
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="distillation: leave out storage errors of the outputs and checks "
        "(plain §5.5 recursion)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="v2: give every fault event its own id (slow reference mode for testing)",
    )
    args = parser.parse_args(argv)

    try:
        program = parse_file(Path(args.file))
        symbols = TypeChecker().check(program)
    except (LexError, ParseError, ValidationError, ResourceEstimationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check_only:
        print(f"{args.file}: validation successful")
        return 0

    circuit = None
    estimate = None
    try:
        if not args.no_circuit:
            circuit = Compiler(symbols).compile_program(program)
        if not args.no_estimate:
            outputs = (
                [name.strip() for name in args.outputs.split(",") if name.strip()]
                if args.outputs
                else None
            )
            if args.semantics == "v1":
                estimate = ResourceEstimator(symbols).estimate_program(program)
            elif args.semantics == "distill":
                estimate = DistillationEstimator(
                    symbols, outputs=outputs, storage=not args.no_storage
                ).estimate_program(program)
            else:
                estimator = ResourceEstimatorV2(
                    symbols,
                    compress=not args.no_compress,
                    outputs=outputs,
                    detect_distillation=not args.no_distill,
                )
                estimator.distill_storage = not args.no_storage  # read by the distillation plug-in
                estimate = estimator.estimate_program(program)
    except ResourceEstimationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.tree:
        circuit_output = (
            render_tree_dot(circuit) if args.dot else render_tree_ascii(circuit)
        ) if circuit is not None else ""
    else:
        circuit_output = (
            render_dot(circuit) if args.dot else render_ascii(circuit)
        ) if circuit is not None else ""

    if args.dot and not args.no_circuit:
        print(circuit_output)
        if estimate is not None:
            print(file=sys.stderr)
            _print_estimate(estimate, args.trace, sys.stderr)
        return 0

    if not args.no_circuit:
        print("Compiled circuit")
        print("================")
        print(circuit_output)
        print()

    if estimate is not None:
        _print_estimate(estimate, args.trace, sys.stdout)
    return 0


def _print_estimate(estimate: ResourceEstimate, trace: bool, stream: TextIO) -> None:
    show_cycles = estimate.time_cycles is not None
    print("Resource estimate", file=stream)
    print("=================", file=stream)
    print(f"{'time (absolute)':<20} {estimate.time:>12.6g}", file=stream)
    if show_cycles:
        print(f"{'time (#SE cycles)':<20} {estimate.time_cycles:>12.6g}", file=stream)
    print(f"{'space':<20} {estimate.space:>12} physical qubits", file=stream)
    if estimate.mixed_time_units and estimate.time_unit_scheme is not None:
        print(
            f"{'time reference':<20} {estimate.time_unit_scheme} syndrome-extraction cycle",
            file=stream,
        )
    print(f"{'error semantics':<20} {getattr(estimate, 'semantics', 'v1'):>12}", file=stream)
    print(file=stream)
    print("Final error context", file=stream)
    print("-------------------", file=stream)
    print(_format_context_table(estimate.context), file=stream)
    if isinstance(estimate, DistillationEstimate):
        _print_distill_sections(estimate, stream)
    elif isinstance(estimate, ResourceEstimateV2):
        _print_v2_sections(estimate, stream)
    if trace:
        print(file=stream)
        print("Trace", file=stream)
        print("-----", file=stream)
        print(f"{len(estimate.trace)} entries; context is compacted as grouped live blocks.", file=stream)
        print(_format_trace_table(estimate.trace, show_cycles), file=stream)


def _print_v2_sections(estimate: ResourceEstimateV2, stream: TextIO) -> None:
    print(file=stream)
    print("Program outputs (error at release)", file=stream)
    print("----------------------------------", file=stream)
    if not estimate.outputs:
        print("(none)", file=stream)
    else:
        print(_format_context_table(ErrorContext(dict(estimate.outputs))), file=stream)
        print(
            f"{'joint output error':<26} {estimate.output_risk:.6g}  "
            "(shared fault sources counted once)",
            file=stream,
        )
        print(
            f"{'naive combination':<26} {estimate.naive_output_risk:.6g}  "
            "(blocks treated as independent)",
            file=stream,
        )
    for report in estimate.distillations:
        print(file=stream)
        print("Distillation detected", file=stream)
        print("---------------------", file=stream)
        print(f"{'scope':<26} {_format_label(report.label)}", file=stream)
        print(
            f"{'rotations / checks / out':<26} {report.rotations} / {len(report.checks)} / "
            f"{len(report.outputs)}",
            file=stream,
        )
        print(f"{'max rotation error q_j':<26} {report.rotation_error:.6g}", file=stream)
        print(f"{'acceptance A':<26} {report.acceptance:.6g}", file=stream)
        print(f"{'accepted error eps_acc':<26} {report.accepted_error:.6g}", file=stream)
        if len(report.outputs) > 1:
            print(
                f"{'error of one output':<26} {max(report.output_errors):.6g}  "
                "(largest; eps_acc = at least one output wrong)",
                file=stream,
            )
        print(f"{'storage errors':<26} {_storage_note(report)}", file=stream)


def _print_distill_sections(estimate: DistillationEstimate, stream: TextIO) -> None:
    for report in estimate.protocols:
        protocol = report.protocol
        print(file=stream)
        print("Distillation (§5.5)", file=stream)
        print("-------------------", file=stream)
        print(f"{'scope':<28} {_format_label(protocol.scope)}", file=stream)
        print(f"{'rotations n':<28} {report.rotations}", file=stream)
        checks = ", ".join(f"{basis}({format_qubit(q)})" for q, basis in report.checks)
        print(f"{'checks m':<28} {len(report.checks)}  ({_format_label(checks)})", file=stream)
        outputs = ", ".join(format_qubit(q) for q in report.outputs)
        print(f"{'outputs k':<28} {len(report.outputs)}  ({_format_label(outputs)})", file=stream)
        print(f"{'acceptance A':<28} {report.acceptance:.6g}", file=stream)
        print(f"{'expected attempts 1/A':<28} {1.0 / report.acceptance:.6g}", file=stream)
        print(f"{'accepted output error':<28} {report.accepted_error:.6g}", file=stream)
        if len(report.outputs) > 1:
            print(
                f"{'  = at least one output':<28} {report.accepted_error:.6g}",
                file=stream,
            )
            each = ", ".join(
                f"{_format_label(format_qubit(q))}={e:.6g}"
                for q, e in zip(report.outputs, report.output_errors)
            )
            print(
                f"{'error of one output':<28} {max(report.output_errors):.6g}  (largest; {each})",
                file=stream,
            )
        print(f"{'storage errors':<28} {_storage_note(report)}", file=stream)
        print(file=stream)
        headers = ("j", "axis P_j", "q_F", "q_m", "q_j", "+anc storage", "l_j", "h_j")
        rows = [
            (
                str(index),
                _format_label(format_axis(rot.axis)),
                f"{rot.q_f:.4g}",
                f"{rot.q_m:.4g}",
                f"{rot.q:.4g}",
                f"{rot.q_total:.4g}" if report.storage else "-",
                format(rot.check_flips, f"0{len(report.checks)}b")[::-1],
                format(rot.output_flips, f"0{len(report.outputs)}b")[::-1],
            )
            for index, rot in enumerate(protocol.rotations, start=1)
        ]
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)), file=stream)
        print("  ".join("-" * w for w in widths), file=stream)
        for row in rows:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)), file=stream)
    if estimate.disturbance_risk is not None:
        print(file=stream)
        print(
            f"{'without postselection':<28} {estimate.disturbance_risk:.6g}  "
            "(§5.4 disturbance risk of the outputs, for comparison)",
            file=stream,
        )


def _storage_note(report) -> str:
    if not report.storage:
        return "not included (--no-storage)"
    return (
        f"included; without them A={report.acceptance_no_storage:.6g}, "
        f"eps_acc={report.accepted_error_no_storage:.6g}"
    )


def _format_context_table(context: ErrorContext) -> str:
    rows = _context_rows(context)
    headers = ("context", "epsilon", "scheme")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(row[2]) for row in rows)),
    ]
    lines = [
        f"{headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:<{widths[2]}}",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}",
    ]
    for block, epsilon, scheme in rows:
        lines.append(f"{block:<{widths[0]}}  {epsilon:>{widths[1]}}  {scheme:<{widths[2]}}")
    return "\n".join(lines)


def _format_trace_table(trace: list[TraceEntry], show_cycles: bool = False) -> str:
    rows: list[tuple[str, ...]] = []
    for idx, entry in enumerate(trace, start=1):
        live = len(entry.context.blocks)
        max_epsilon = _max_epsilon(entry.context)
        context = _context_summary(entry.context)
        label = _format_label(entry.label)
        if show_cycles:
            rows.append(
                (
                    f"{idx:03d}",
                    _shorten(label, _TRACE_EVENT_WIDTH),
                    f"{entry.time:.6g}",
                    f"{entry.time_cycles:.6g}" if entry.time_cycles is not None else "",
                    str(entry.space),
                    str(live),
                    max_epsilon,
                    _shorten(context, _TRACE_CONTEXT_WIDTH),
                )
            )
        else:
            rows.append(
                (
                    f"{idx:03d}",
                    _shorten(label, _TRACE_EVENT_WIDTH),
                    f"{entry.time:.6g}",
                    str(entry.space),
                    str(live),
                    max_epsilon,
                    _shorten(context, _TRACE_CONTEXT_WIDTH),
                )
            )
    if not rows:
        rows = [
            ("", "(no trace entries)", "", "", "", "", "-", "(empty)")
            if show_cycles
            else ("", "(no trace entries)", "", "", "", "-", "(empty)")
        ]

    headers = (
        ("#", "event", "time", "#SE", "space", "live", "max eps", "context")
        if show_cycles
        else ("#", "event", "time", "space", "live", "max eps", "context")
    )
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    right_aligned = {0, 2, 3, 4, 5, 6} if show_cycles else {0, 2, 3, 4, 5}
    lines = [
        "  ".join(
            f"{headers[col]:>{widths[col]}}"
            if col in right_aligned
            else f"{headers[col]:<{widths[col]}}"
            for col in range(len(headers))
        ),
        "  ".join("-" * widths[col] for col in range(len(headers))),
    ]
    for row in rows:
        if show_cycles:
            number, event, time, cycles, space, live, max_eps, context = row
            lines.append(
                f"{number:>{widths[0]}}  "
                f"{event:<{widths[1]}}  "
                f"{time:>{widths[2]}}  "
                f"{cycles:>{widths[3]}}  "
                f"{space:>{widths[4]}}  "
                f"{live:>{widths[5]}}  "
                f"{max_eps:>{widths[6]}}  "
                f"{context:<{widths[7]}}"
            )
        else:
            number, event, time, space, live, max_eps, context = row
            lines.append(
                f"{number:>{widths[0]}}  "
                f"{event:<{widths[1]}}  "
                f"{time:>{widths[2]}}  "
                f"{space:>{widths[3]}}  "
                f"{live:>{widths[4]}}  "
                f"{max_eps:>{widths[5]}}  "
                f"{context:<{widths[6]}}"
            )
    return "\n".join(lines)


def _context_rows(context: ErrorContext) -> list[tuple[str, str, str]]:
    if not context.blocks:
        return [("(empty)", "-", "-")]
    rows = []
    for (reg, idx), state in sorted(context.blocks.items()):
        rows.append((_format_block(reg, idx), f"{state.epsilon:.6g}", state.scheme.name))
    return rows


def _context_summary(context: ErrorContext) -> str:
    if not context.blocks:
        return "(empty)"

    scheme_names = {state.scheme.name for state in context.blocks.values()}
    show_scheme = len(scheme_names) > 1
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for (reg, idx), state in sorted(context.blocks.items()):
        key = (_format_register(reg), f"{state.epsilon:.6g}", state.scheme.name)
        grouped.setdefault(key, []).append(idx)

    parts = []
    for (reg, epsilon, scheme), indices in grouped.items():
        scheme_suffix = f"@{scheme}" if show_scheme else ""
        parts.append(f"{reg}[{_format_index_ranges(indices)}]={epsilon}{scheme_suffix}")
    return "; ".join(parts)


def _format_index_ranges(indices: list[int]) -> str:
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(_format_index_range(start, prev))
        start = prev = idx
    ranges.append(_format_index_range(start, prev))
    return ",".join(ranges)


def _format_index_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _max_epsilon(context: ErrorContext) -> str:
    if not context.blocks:
        return "-"
    return f"{max(state.epsilon for state in context.blocks.values()):.6g}"


def _format_block(reg: str, idx: int) -> str:
    return f"{_format_register(reg)}[{idx}]"


def _format_label(label: str) -> str:
    return _CALL_RENAMED_REGISTER_RE.sub(_format_register_match, label)


def _format_register(reg: str) -> str:
    match = _CALL_RENAMED_REGISTER_RE.fullmatch(reg)
    if match is None:
        return reg
    return _format_register_match(match)


def _format_register_match(match: re.Match[str]) -> str:
    suffix = f"{match.group(2)}:{match.group(3)}"
    if match.group(4) is not None:
        suffix += f".{match.group(4)}"
    return f"{match.group(1)}@call{suffix}"


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
