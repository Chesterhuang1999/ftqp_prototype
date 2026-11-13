#!/usr/bin/env python3
"""Generate the RQ3 attribution-ladder programs (Sec. 6.3) from the QASM benchmarks.

Each <bench><size>.qasm (gates h, x, z, t, cx) becomes one .ft program per
configuration, written to <out>/<config>/<bench><size>.ft:

  c0  QREv3's assumptions. Cliffords, including CX, are zero-time primitives;
      T is a primitive taking 15 cycles with a 9,700-qubit factory; tdec = 0.
      k logical qubits.
  c1  c0 + CNOT by lattice surgery: each cx becomes the ZZ/XX/Z measurement
      block on one extra logical ancilla q[k], so the code block has k+1
      logical qubits. tdec = 0.
  c2  c1 + decoding latency: tdec = --tdec-us, and T takes 15 cycles + tdec
      (the wait for the Clifford correction after injection).

Layout, as in QREv3's surface-code model at distance d:
  patches(k) = 2k + ceil(sqrt(8k)) + 1
  data qubits   n      = patches(k) * d^2
  SE ancillas   #anc   = patches(k) * (d^2 - 1)
Unlike the hand-written files, n is computed for the same k as the SE ancillas.

Usage:
  python generate.py                       # c0 and c1
  python generate.py --tdec-us 3.2         # c0, c1 and c2
  python generate.py --config c2 --tdec-us 3.2 --out /tmp/rq3
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS = ("qft", "bv", "ripple")
SIZES = (100, 200, 300, 400)
CONFIGS = ("c0", "c1", "c2")

# QREv3 setting of Sec. 6.3: GateBased(error_rate=1e-4, gate_time=100,
# measurement_time=500), SurfaceCode(distance=21), RoundBasedFactory.
HARDWARE = "t1: 100, t2: 100, tm: 500, p1: 1e-4, p2: 1e-4, pm: 1e-4"  # times in ns
T_CYCLES = 15           # T-state injection time, in SE cycles
FACTORY_QUBITS = 9700   # physical qubits of one T factory
QRE_T_ERROR = 2.165e-9  # error_rate of the factory QREv3 selects in this setting

SINGLE = {"h": "H", "x": "X", "z": "Z", "t": "T"}
GATE = re.compile(r"^(\w+)\s+(.+);$")
QUBIT = re.compile(r"^q\[(\d+)\]$")


def num(value: float) -> str:
    """Plain decimal literal: the spec expression parser rejects '2.165e-09'."""
    from decimal import Decimal

    text = format(Decimal(repr(float(value))), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def patches(k: int) -> int:
    return 2 * k + math.ceil(math.sqrt(8 * k)) + 1


def parse_qasm(path: Path) -> tuple[int, list[tuple[str, tuple[int, ...]]]]:
    """Return (number of qubits, [(gate, qubit indices)])."""
    width = None
    gates: list[tuple[str, tuple[int, ...]]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith(("OPENQASM", "include")):
            continue
        decl = re.match(r"^qubit\[(\d+)\]\s+q;$", line)
        if decl:
            width = int(decl[1])
            continue
        m = GATE.match(line)
        if m is None:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse '{raw.strip()}'")
        name = m[1]
        args = []
        for arg in m[2].split(","):
            q = QUBIT.match(arg.strip())
            if q is None:
                raise SystemExit(f"{path.name}:{lineno}: unsupported operand '{arg.strip()}'")
            args.append(int(q[1]))
        arity = 2 if name == "cx" else 1
        if name not in SINGLE and name != "cx":
            raise SystemExit(f"{path.name}:{lineno}: unsupported gate '{name}'")
        if len(args) != arity:
            raise SystemExit(f"{path.name}:{lineno}: '{name}' takes {arity} qubit(s)")
        gates.append((name, tuple(args)))
    if width is None:
        raise SystemExit(f"{path.name}: missing 'qubit[N] q;'")
    return width, gates


def spec(config: str, width: int, d: int, tdec_ns: float, p_t: float) -> str:
    surgery = config != "c0"
    k = width + 1 if surgery else width
    tiles = patches(k)
    free = '{ arity: 1, time: "0", "#SE": "0", "#ancilla": "0" }'
    ops = [f"    X: {free},", f"    Z: {free},", f"    H: {free},"]
    if not surgery:
        ops.append('    CX: { arity: 2, time: "0", "#SE": "0", "#ancilla": "0" },')
    t_time = f"{T_CYCLES}*tSE" if config != "c2" else f"{T_CYCLES}*tSE + {num(tdec_ns)}"
    return "\n".join(
        [
            f"hardware qubit_gate_ns_e4 = {{\n  {HARDWARE}\n}};",
            "",
            f"qecc ms_like using qubit_gate_ns_e4 = {{",
            f"  [[n,k,d]]: [{tiles * d * d}, {k}, {d}],   // n = patches(k) * d^2, patches(k) = {tiles}",
            '  SE: { "#ancilla": "(2*k + ceil(sqrt(8*k)) + 1) * (d * d - 1)", tSE: "4t1+2tm" },',
            '  PREP: { time: "d*tSE", "#SE": "0", "#ancilla": "0" },',
            "  OPL: {",
            *ops,
            "  },",
            "  OPF: {",
            f'    T: {{ arity: 1, time: "{t_time}", pF: "{num(p_t)}", '
            f'"#ancilla": "{FACTORY_QUBITS}+(2*k + ceil(sqrt(8*k)) + 1) * (d * d - 1)" }}',
            "  },",
            f'  tdec: "{num(tdec_ns)}",',
            '  pL: "0.3*(max(p1,p2,pm)/0.01)^((d+1)/2)",',
            '  naux: "0"',
            "};",
        ]
    )


def body(config: str, width: int, gates: list[tuple[str, tuple[int, ...]]]) -> list[str]:
    anc = width  # q[width] is the reusable lattice-surgery ancilla (c1, c2)
    lines: list[str] = []
    for name, args in gates:
        if name != "cx":
            lines.append(f"    {SINGLE[name]}[q[{args[0]}]];")
        elif config == "c0":
            lines.append(f"    CX[q[{args[0]}], q[{args[1]}]];")
        else:
            c, t = args
            lines += [
                f"    // cx q[{c}], q[{t}]",
                f"    H[q[{anc}]];",
                f"    if M(ZZ)[q[{c}], q[{anc}]]; {{",
                f"        if M(XX)[q[{anc}], q[{t}]]; then Z[q[{c}]]",
                f"    }} then X[q[{t}]];",
                f"    if M(Z)[q[{anc}]]; then {{ X[q[{t}]]; X[q[{anc}]] }};",
            ]
    return lines


def render(config: str, qasm: Path, d: int, tdec_us: float, p_t: float) -> str:
    width, gates = parse_qasm(qasm)
    tdec_ns = tdec_us * 1000.0 if config == "c2" else 0.0
    counts = {g: sum(1 for name, _ in gates if name == g) for g in ("t", "cx")}
    header = [
        f"// Generated by generate.py from {qasm.name}; do not edit.",
        f"// Configuration {config}: {width} qubits, {counts['t']} T, {counts['cx']} CX, d = {d}, "
        f"tdec = {num(tdec_ns)} ns.",
    ]
    return "\n".join(
        [
            *header,
            spec(config, width, d, tdec_ns, p_t),
            "",
            "program {",
            "    logical q := new ms_like;",
            *body(config, width, gates),
            "    release(q);",
            "}",
            "",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", choices=CONFIGS, help="default: c0 c1, plus c2 if --tdec-us is given")
    ap.add_argument("--tdec-us", type=float, help="decoding latency in microseconds (needed for c2)")
    ap.add_argument("--distance", type=int, default=21)
    ap.add_argument("--t-error", type=float, default=QRE_T_ERROR, help="error of one T state (default: QREv3's factory)")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "generated")
    args = ap.parse_args()

    configs = args.config or (CONFIGS if args.tdec_us is not None else CONFIGS[:2])
    if "c2" in configs and args.tdec_us is None:
        ap.error("c2 needs --tdec-us")
    for config in configs:
        target = args.out / config
        target.mkdir(parents=True, exist_ok=True)
        for bench in BENCHMARKS:
            for size in SIZES:
                qasm = SCRIPT_DIR / f"{bench}{size}.qasm"
                text = render(config, qasm, args.distance, args.tdec_us or 0.0, args.t_error)
                (target / f"{bench}{size}.ft").write_text(text, encoding="utf-8")
        print(f"{config}: wrote {len(BENCHMARKS) * len(SIZES)} programs to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
