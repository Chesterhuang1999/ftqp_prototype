#!/usr/bin/env python3
"""Generate a hierarchical FTQP translation of the upstream EEA circuit.

The generated program preserves every upstream instruction definition and
recursively lowers it until only X, CX, and CCX remain as QEC primitives.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import TextIO

from qiskit.circuit import Instruction, QuantumCircuit


PRIMITIVES = {"x": "X", "cx": "CX", "ccx": "CCX"}
UPSTREAM_REVISION = "b836f59a1a1bc8d122a2c1e6b794103b7415499b"


@dataclass(frozen=True)
class Description:
    fingerprint: str
    name: str | None
    counts: Counter


class FTQPGenerator:
    def __init__(self, output: TextIO) -> None:
        self.output = output
        self.known: dict[str, Description] = {}
        self.names: set[str] = set()
        self.definition_count = 0

    def describe(self, inst: Instruction, object_cache: dict[int, Description]) -> Description:
        primitive = PRIMITIVES.get(inst.name)
        if primitive is not None:
            return Description(f"P:{primitive}", None, Counter({primitive: 1}))

        cached = object_cache.get(id(inst))
        if cached is not None:
            return cached

        definition = inst.definition
        if definition is None:
            raise ValueError(f"instruction {inst.name!r} has no definition")

        children: list[tuple[Description, tuple[int, ...]]] = []
        digest = hashlib.sha256()
        digest.update(f"N:{inst.num_qubits}:{inst.num_clbits}|".encode())
        total = Counter()

        for item in definition.data:
            child = self.describe(item.operation, object_cache)
            local_qubits = tuple(definition.find_bit(qubit).index for qubit in item.qubits)
            children.append((child, local_qubits))
            digest.update(child.fingerprint.encode())
            digest.update(repr(local_qubits).encode())
            digest.update(b";")
            total += child.counts

        fingerprint = digest.hexdigest()
        known = self.known.get(fingerprint)
        if known is not None:
            object_cache[id(inst)] = known
            return known

        name = self._unique_name(inst.name, fingerprint)
        description = Description(fingerprint, name, total)
        self.known[fingerprint] = description
        object_cache[id(inst)] = description
        self._write_definition(name, inst.num_qubits, children)
        self.definition_count += 1
        return description

    def _unique_name(self, source_name: str, fingerprint: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", source_name).strip("_") or "gate"
        if base[0].isdigit():
            base = f"g_{base}"
        candidate = f"eea_{base}_{fingerprint[:12]}"
        suffix = 1
        while candidate in self.names:
            suffix += 1
            candidate = f"eea_{base}_{fingerprint[:12]}_{suffix}"
        self.names.add(candidate)
        return candidate

    def _write_definition(
        self,
        name: str,
        arity: int,
        children: list[tuple[Description, tuple[int, ...]]],
    ) -> None:
        params = ", ".join(f"q{i}" for i in range(arity))
        self.output.write(f"def {name}({params}) {{\n")
        for child, local_qubits in children:
            args = ", ".join(f"q{index}" for index in local_qubits)
            if child.name is None:
                primitive = child.fingerprint.removeprefix("P:")
                self.output.write(f"  {primitive}[{args}];\n")
            else:
                self.output.write(f"  call {child.name}({args});\n")
        self.output.write("};\n\n")


def load_upstream(path: Path):
    spec = importlib.util.spec_from_file_location("upstream_eea_circuit", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import upstream module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_step_circuit(upstream, n: int, len_width: int, step: int) -> QuantumCircuit:
    regs = upstream.make_global_registers(n=n, len_width=len_width)
    qc = QuantumCircuit(*regs, name=f"EEA_STEP_{step}")
    (
        phase1,
        phase2,
        iteration,
        zero,
        sign,
        work1,
        work2,
        l_t,
        l_q,
        l_s,
        l_rp,
        pre_s,
        post_s,
        phase_s,
        step7_zero_lt,
        step7_carry_lt,
        step7_zero_rp,
        step7_carry_rp,
        pre,
        pre2,
        pre3,
    ) = regs
    upstream.append_one_step_T(
        qc,
        T=step,
        n=n,
        len_width=len_width,
        Phase1=phase1,
        Phase2=phase2,
        Iter=iteration,
        Zero=zero,
        Sign=sign,
        Work1=work1,
        Work2=work2,
        l_t=l_t,
        l_q=l_q,
        l_s=l_s,
        l_rp=l_rp,
        PreS=pre_s,
        PostS=post_s,
        PhaseS=phase_s,
        Step7Zero_lt=step7_zero_lt,
        Step7Carry_lt=step7_carry_lt,
        Step7Zero_rp=step7_zero_rp,
        Step7Carry_rp=step7_carry_rp,
        Pre=pre,
        Pre2=pre2,
        Pre3=pre3,
    )
    return qc


def write_header(output: TextIO, n: int, logical_qubits: int, steps: int) -> None:
    data_qubits = logical_qubits * 31 * 31
    output.write(
        f"""// Generated hierarchical FTQP translation of the n={n} modular-inversion EEA circuit.
// Upstream revision: {UPSTREAM_REVISION}
//
// Every upstream composite instruction is defined below as an FTQP subcircuit.
// The only specified QEC primitives are the one-, two-, and three-qubit gates
// X, CX, and CCX. The program contains {steps} EEA steps over {logical_qubits}
// logical qubits. Primitive costs are intentionally explicit and can be
// replaced without changing the translated algorithm.
//
// CCX is non-Clifford. Its cost uses a conservative exact decomposition into
// 7 T/T-dagger gates, 6 CX gates, and 2 H gates. A T injection is charged 15
// syndrome-extraction cycles, while each CX/H is charged d cycles. The seven
// T states are consumed sequentially from one reusable 9,700-physical-qubit
// magic-state factory. Therefore:
//
//   CCX cycles  = 7*15 + (6+2)*d
//   CCX ancilla = magic-state factory + two logical workspace patches

hardware superconducting_eea = {{
  t1: 1e-8,
  t2: 1e-7,
  tm: 1e-6,
  p1: 1e-4,
  p2: 1e-4,
  pm: 1e-4
}};

qecc surface_code_eea{n} using superconducting_eea = {{
  [[n,k,d]]: [{data_qubits}, {logical_qubits}, 31],

  SE: {{
    "#ancilla": "k*(d*d-1)",
    tSE: "2*t1+4*t2+tm"
  }},

  PREP: {{
    time: "d*tSE",
    "#SE": "d",
    "#ancilla": "0"
  }},

  OPL: {{
    X:   {{ arity: 1, time: "d*tSE", "#SE": "d", "#ancilla": "0" }},
    CX:  {{ arity: 2, time: "d*tSE", "#SE": "d", "#ancilla": "d*d" }},
    CCX: {{
      arity: 3,
      time: "(7*15+8*d)*tSE",
      "#SE": "7*15+8*d",
      "#ancilla": "9700+2*d*d"
    }}
  }},

  OPF: {{}},
  tdec: "680*tSE",
  pL: "min(1,k*0.1*(max(p1,p2,pm)/0.0057)^((d+1)/2))",
  naux: "d*d"
}};

"""
    )


def generate(upstream_path: Path, output_path: Path, n: int, steps: int | None) -> None:
    upstream = load_upstream(upstream_path)
    config = upstream.get_n_config(n)
    len_width = config["len_width"]
    total_steps = config["T_max"] if steps is None else steps
    if not 1 <= total_steps <= config["T_max"]:
        raise SystemExit(f"--steps must be between 1 and {config['T_max']}")

    probe_regs = upstream.make_global_registers(n=n, len_width=len_width)
    logical_qubits = sum(len(register) for register in probe_regs)
    top_level_calls: list[tuple[int, Description, tuple[int, ...], str]] = []
    grand_total = Counter()

    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
        write_header(output, n, logical_qubits, total_steps)
        generator = FTQPGenerator(output)

        for step in range(1, total_steps + 1):
            circuit = make_step_circuit(upstream, n, len_width, step)
            object_cache: dict[int, Description] = {}
            for item in circuit.data:
                description = generator.describe(item.operation, object_cache)
                global_qubits = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)
                top_level_calls.append(
                    (step, description, global_qubits, item.operation.name)
                )
                grand_total += description.counts

        output.write(
            "// Recursive expansion totals:\n"
            f"//   X   = {grand_total['X']}\n"
            f"//   CX  = {grand_total['CX']}\n"
            f"//   CCX = {grand_total['CCX']}\n"
            f"//   all primitives = {sum(grand_total.values())}\n\n"
        )
        output.write("program {\n")
        output.write(f"  logical state := new surface_code_eea{n};\n")
        output.write("  prepare(state);\n")
        previous_step = 0
        for step, description, qubits, source_name in top_level_calls:
            if step != previous_step:
                output.write(f"\n  // EEA step {step}\n")
                previous_step = step
            args = ", ".join(f"state[{index}]" for index in qubits)
            if description.name is None:
                primitive = description.fingerprint.removeprefix("P:")
                output.write(f"  {primitive}[{args}]; // {source_name}\n")
            else:
                output.write(
                    f"  call {description.name}({args}); // {source_name}\n"
                )
        output.write("  release(state);\n")
        output.write("}\n")

    print(f"wrote {output_path}")
    print(f"logical qubits: {logical_qubits}")
    print(f"subcircuit definitions: {generator.definition_count}")
    print(
        "primitive counts: "
        f"X={grand_total['X']} CX={grand_total['CX']} CCX={grand_total['CCX']} "
        f"total={sum(grand_total.values())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="path to the upstream eea_circuit.py file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("eaa.ft"),
        help="generated FTQP path",
    )
    parser.add_argument("--n", type=int, default=64, help="upstream EEA size")
    parser.add_argument(
        "--steps",
        type=int,
        help="generate only the first N steps; defaults to the complete configured circuit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.upstream, args.output, args.n, args.steps)
