"""Import a simple OpenQASM 3 program into Microsoft's QRE estimator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from qdk.qre import estimate, instruction_name, plot_estimates, property_name
from qdk.qre.application import OpenQASMApplication
from qdk.qre.models import (
    GateBased,
    RoundBasedFactory,
    SurfaceCode,
)
from qdk.qre.property_keys import DISTANCE


def format_property_key(key: int) -> str:
    name = property_name(key)
    if name is None:
        return f"property_{key}"
    return f"{name} ({key})"


def format_value(value: Any) -> str:
    return str(value)


def code_distances(entry: Any) -> list[int]:
    if DISTANCE in entry.properties:
        return [entry.properties[DISTANCE]]

    distances = set()
    for node in entry.source.nodes:
        transform = getattr(node, "transform", None)
        distance = getattr(transform, "distance", None)
        if distance is not None:
            distances.add(distance)

    return sorted(distances)


def print_properties(entry: Any) -> None:
    if not entry.properties:
        print("  properties: none")
        return

    print("  properties:")
    for key, value in sorted(entry.properties.items()):
        print(f"    {format_property_key(key)} = {format_value(value)}")


def print_factories(entry: Any) -> None:
    if not entry.factories:
        print("  factories: none")
        return

    print("  factories:")
    for instruction_id, factory in sorted(entry.factories.items()):
        name = instruction_name(instruction_id) or f"instruction_{instruction_id}"
        print(f"    {name} ({instruction_id}):")
        print(f"      copies = {factory.copies}")
        print(f"      runs = {factory.runs}")
        print(f"      states = {factory.states}")
        print(f"      error_rate = {factory.error_rate}")


def print_source(entry: Any) -> None:
    print("  instruction source:")
    print(str(entry.source))


def print_entry(index: int, entry: Any) -> None:
    distances = code_distances(entry)
    distance_text = ", ".join(str(distance) for distance in distances) or "n/a"

    print(f"estimate[{index}]")
    print(f"  qubits = {entry.qubits}")
    print(f"  runtime_ns = {entry.runtime}")
    print(f"  error = {entry.error}")
    print(f"  code_distance = {distance_text}")
    print_properties(entry)
    print_factories(entry)
    print_source(entry)










def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import an OpenQASM 3 program into Microsoft's QRE estimator."
    )
    parser.add_argument("qasm_path", type=Path, help="path to the OpenQASM 3 program")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qasm_path = args.qasm_path
    source = qasm_path.read_text(encoding="utf-8")

    app = OpenQASMApplication(source)
    arch = GateBased(error_rate=1e-4, gate_time=100, measurement_time=500)
    isa_query = SurfaceCode.q(distance=21) * RoundBasedFactory.q()

    estimates = estimate(app, arch, isa_query, max_error=1e-4, name=qasm_path.name)
    
    for index, entry in enumerate(estimates, start=1):
        print_entry(index, entry)
        # print(f"qubits={entry.qubits} runtime={entry.runtime} error={entry.error}")
    
    
    # out_dir = Path("./")
    # out_dir.mkdir(parents=True, exist_ok=True)

    # fig = plot_estimates(estimates, figsize=(6, 4))

    # if fig is not None:
    #     fig.savefig(out_dir / "estimator.svg", format="svg", bbox_inches="tight")
    # else:
    #     plt.gcf().savefig(out_dir / "estimator.svg", format="svg", bbox_inches="tight")

if __name__ == "__main__":
    main()
