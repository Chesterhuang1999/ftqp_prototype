"""Readable renderers for compiled circuits.

The public renderers show a circuit schedule: code blocks are rows, and compiled
operations are columns. Tree renderers are kept for debugging the structural
compiler output, but they are not the default visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

from ftqp_prototype.compiler.circuit import CircuitNode


@dataclass(frozen=True)
class CircuitStep:
    index: int
    time: int
    kind: str
    label: str
    blocks: tuple[str, ...]
    measured_blocks: tuple[str, ...] = ()
    controlled_blocks: tuple[str, ...] = ()
    control_sources: tuple[int, ...] = ()


@dataclass
class _RawStep:
    index: int
    time: int
    kind: str
    label: str
    blocks: tuple[str, ...]
    measured_blocks: tuple[str, ...] = ()
    controlled_blocks: tuple[str, ...] = ()
    control_sources: tuple[int, ...] = ()


class _ScheduleBuilder:
    def __init__(self) -> None:
        self.steps: list[_RawStep] = []
        self._next_index = 1

    def add(
        self,
        time: int,
        kind: str,
        label: str,
        blocks: tuple[str, ...],
        *,
        measured_blocks: tuple[str, ...] = (),
        controlled_blocks: tuple[str, ...] = (),
        control_sources: tuple[int, ...] = (),
    ) -> int:
        index = self._next_index
        self._next_index += 1
        self.steps.append(
            _RawStep(
                index=index,
                time=time,
                kind=kind,
                label=label,
                blocks=blocks,
                measured_blocks=measured_blocks,
                controlled_blocks=controlled_blocks,
                control_sources=control_sources,
            )
        )
        return index


def render_ascii(root: CircuitNode) -> str:
    """Render a compiled circuit as code-block wires and operation columns."""

    steps = schedule_steps(root)
    blocks = _all_blocks(steps)
    if not steps:
        return "(empty circuit)"
    if not blocks:
        return "\n".join(f"{step.index:02d}. {step.label}" for step in steps)

    times = _all_times(steps)
    lifetimes = _block_lifetimes(steps)
    cell_width = _ascii_cell_width(steps, times)
    label_width = max(len(block) for block in blocks)
    occupancy = _occupancy(steps)
    lines = ["time  : " + " ".join(f"{time:>{cell_width}}" for time in times)]
    for block in blocks:
        cells = []
        for time in times:
            if _is_live(block, time, lifetimes):
                step = occupancy.get((block, time))
                token = _ascii_token(step, block, cell_width)
            else:
                token = " " * cell_width
            cells.append(f"{token:>{cell_width}}")
        lines.append(f"{block:<{label_width}}: " + " ".join(cells))
    lines.append("")
    lines.append("Legend: M=measured block, C=decode+controlled operation,")
    lines.append("        P=prepare, O=operation, A=allocate, R=release, U=repeat-until")
    lines.append("")
    lines.append("Steps:")
    show_times = any(step.time != step.index for step in steps)
    for step in steps:
        time = f" t={step.time:02d}" if show_times else ""
        target = f" ({', '.join(step.blocks)})" if step.blocks else ""
        detail = _step_detail(step)
        lines.append(f"{step.index:02d}.{time} {step.label}{target}{detail}")
    controls = _control_summary(steps)
    if controls:
        lines.append("")
        lines.append("Control dependencies:")
        lines.extend(controls)
    return "\n".join(lines)


def render_tree_ascii(root: CircuitNode) -> str:
    """Render the structural compiled-circuit tree for debugging."""

    lines: list[str] = []
    _render(root, "", True, lines)
    return "\n".join(lines)


def _render(node: CircuitNode, prefix: str, last: bool, lines: list[str]) -> None:
    connector = "`- " if last else "+- "
    block_suffix = f" [{', '.join(node.blocks)}]" if node.blocks else ""
    lines.append(f"{prefix}{connector}{node.kind}: {node.label}{block_suffix}")
    child_prefix = prefix + ("   " if last else "|  ")
    for idx, child in enumerate(node.children):
        _render(child, child_prefix, idx == len(node.children) - 1, lines)


def render_dot(root: CircuitNode) -> str:
    """Render a compiled circuit timeline as Graphviz DOT."""

    steps = schedule_steps(root)
    blocks = _all_blocks(steps)
    lines = [
        "digraph FTQP {",
        "  graph [rankdir=LR, splines=ortho, nodesep=0.16, ranksep=0.06];",
        '  node [fontname=Courier, fontsize=10, margin="0.025,0.02"];',
        "  edge [fontname=Courier, fontsize=9, arrowsize=0.55];",
    ]
    if not steps:
        lines.append('  empty [label="empty circuit", shape=box];')
        lines.append("}")
        return "\n".join(lines)

    lifetimes = _block_lifetimes(steps)
    times = _all_times(steps)
    occupancy = _occupancy(steps)
    cell_ids: dict[tuple[str, int], str] = {}
    for row, block in enumerate(blocks):
        safe_block = _safe_id(block)
        label_id = f"row_{row}_label"
        lines.append(f'  {label_id} [label="{_escape(block)}", shape=plaintext];')
        previous = label_id
        for time in times:
            if not _is_live(block, time, lifetimes):
                continue
            cell_id = f"cell_{row}_{time}_{safe_block}"
            cell_ids[(block, time)] = cell_id
            step = occupancy.get((block, time))
            if step is not None:
                lines.append(
                    f'  {cell_id} [label="{_escape(_dot_label(step))}", '
                    f'shape=box, style="rounded,filled", fillcolor="{_color(step.kind)}"];'
                )
                edge_style = 'dir=none, color="#555555", penwidth=1.4'
            else:
                lines.append(
                    f'  {cell_id} [label="", shape=point, '
                    'width=0.015, color="#888888"];'
                )
                edge_style = 'dir=none, color="#888888"'
            lines.append(f"  {previous} -> {cell_id} [{edge_style}];")
            previous = cell_id

    for time in times:
        ranked = [
            cell_ids[(block, time)]
            for block in blocks
            if (block, time) in cell_ids
        ]
        if ranked:
            lines.append("  { rank=same; " + "; ".join(ranked) + "; }")

    for step in steps:
        if not _is_joint_pauli_measurement(step):
            continue
        participants = [
            cell_ids[(block, step.time)]
            for block in blocks
            if block in step.blocks and (block, step.time) in cell_ids
        ]
        for source_cell, target_cell in zip(participants, participants[1:]):
            lines.append(
                f"  {source_cell} -> {target_cell} "
                '[dir=none, style=dashed, color="#64748b", '
                "penwidth=1.2, constraint=false];"
            )

    by_index = {step.index: step for step in steps}
    for step in steps:
        if not step.control_sources:
            continue
        for source_index in step.control_sources:
            source = by_index[source_index]
            source_blocks = source.measured_blocks or source.blocks
            if not source_blocks:
                continue
            source_block = source_blocks[0]
            source_cell = cell_ids.get((source_block, source.time))
            if source_cell is None:
                continue
            for target_block in step.blocks:
                target_cell = cell_ids.get((target_block, step.time))
                if target_cell is None:
                    continue
                lines.append(
                    f"  {source_cell} -> {target_cell} "
                    f'[style=dashed, color="#dc2626", penwidth=1.8, '
                    f'fontcolor="#dc2626", constraint=false];'
                )

    lines.append("}")
    return "\n".join(lines)


def render_tree_dot(root: CircuitNode) -> str:
    """Render the structural compiled-circuit tree as Graphviz DOT."""

    lines = ["digraph FTQP {", "  node [shape=box, fontname=Courier];"]
    counter = 0

    def visit(node: CircuitNode, parent: int | None = None) -> int:
        nonlocal counter
        node_id = counter
        counter += 1
        label = node.label.replace('"', '\\"')
        if node.blocks:
            label += "\\n" + ", ".join(node.blocks)
        lines.append(f'  n{node_id} [label="{node.kind}: {label}"];')
        if parent is not None:
            lines.append(f"  n{parent} -> n{node_id};")
        for child in node.children:
            visit(child, node_id)
        return node_id

    visit(root)
    lines.append("}")
    return "\n".join(lines)


def schedule_steps(root: CircuitNode) -> list[CircuitStep]:
    builder = _ScheduleBuilder()
    _schedule(root, builder, 1, ())
    return [
        CircuitStep(
            index=step.index,
            time=step.time,
            kind=step.kind,
            label=step.label,
            blocks=step.blocks,
            measured_blocks=step.measured_blocks,
            controlled_blocks=step.controlled_blocks,
            control_sources=step.control_sources,
        )
        for step in sorted(builder.steps, key=lambda step: (step.time, step.index))
    ]


def _schedule(
    node: CircuitNode,
    builder: _ScheduleBuilder,
    time: int,
    control_sources: tuple[int, ...],
) -> int:
    if node.kind in {"program", "sequence", "call", "during", "then"}:
        current = time
        for child in node.children:
            current = _schedule(child, builder, current, control_sources)
        return current

    if node.kind == "allocation":
        chain, body = _allocation_chain(node)
        for allocation in chain:
            builder.add(
                time,
                "allocation",
                _allocation_label(allocation.label),
                tuple(allocation.blocks),
                control_sources=control_sources,
            )
        current = time + 1
        if body is not None:
            current = _schedule(body, builder, current, control_sources)
        for allocation in reversed(chain):
            builder.add(
                current,
                "release",
                _release_label(allocation.label),
                tuple(allocation.blocks),
                control_sources=control_sources,
            )
        return current + 1

    if node.kind == "prepare":
        blocks = tuple(node.blocks)
        prepare_label = node.label.replace(" with scheme PREP + syndrome extraction", "")
        builder.add(time, "prepare", prepare_label, blocks, control_sources=control_sources)
        return time + 1

    if node.kind == "primitive":
        blocks = tuple(node.blocks)
        op_label = node.label.replace(" + syndrome extraction", "")
        builder.add(time, "primitive", op_label, blocks, control_sources=control_sources)
        return time + 1

    if node.kind == "conditional":
        measured = tuple(node.blocks)
        then_node = next((child for child in node.children if child.kind == "then"), None)
        controlled = tuple(_collect_blocks(then_node)) if then_node else ()
        measurement_label = _measurement_label(node.label)
        measurement_id = builder.add(
            time,
            "measurement",
            measurement_label,
            measured,
            measured_blocks=measured,
            controlled_blocks=controlled,
            control_sources=control_sources,
        )
        current = time + 1
        for child in node.children:
            if child.kind == "during":
                current = _schedule(child, builder, current, control_sources)
        for child in node.children:
            if child.kind == "then":
                current = _schedule(
                    child,
                    builder,
                    current,
                    control_sources + (measurement_id,),
                )
        return current

    if node.kind == "parallel":
        next_times: list[int] = []
        if node.children:
            next_times.append(
                _schedule(node.children[0], builder, time, control_sources)
            )
        if len(node.children) > 1:
            next_times.append(
                _schedule(node.children[1], builder, time, control_sources)
            )
        return max(next_times, default=time)

    if node.blocks:
        builder.add(
            time,
            node.kind,
            node.label,
            tuple(node.blocks),
            control_sources=control_sources,
        )
        time += 1
    for child in node.children:
        time = _schedule(child, builder, time, control_sources)
    return time


def _allocation_chain(node: CircuitNode) -> tuple[list[CircuitNode], CircuitNode | None]:
    chain = [node]
    current = node
    while len(current.children) == 1 and current.children[0].kind == "allocation":
        current = current.children[0]
        chain.append(current)
    body = current.children[0] if len(current.children) == 1 else None
    return chain, body


def _all_blocks(steps: list[CircuitStep]) -> list[str]:
    blocks: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for block in step.blocks:
            if block not in seen:
                seen.add(block)
                blocks.append(block)
    return blocks


def _all_times(steps: list[CircuitStep]) -> list[int]:
    return sorted({step.time for step in steps})


def _occupancy(steps: list[CircuitStep]) -> dict[tuple[str, int], CircuitStep]:
    cells: dict[tuple[str, int], CircuitStep] = {}
    for step in steps:
        for block in step.blocks:
            cells[(block, step.time)] = step
    return cells


def _ascii_cell_width(steps: list[CircuitStep], times: list[int]) -> int:
    max_time = max(times, default=0)
    max_index = max((step.index for step in steps), default=0)
    return max(3, len(str(max_time)), len(str(max_index)) + 1)


def _block_lifetimes(steps: list[CircuitStep]) -> dict[str, tuple[int, int]]:
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for step in steps:
        for block in step.blocks:
            first.setdefault(block, step.time)
            if step.kind == "release":
                last[block] = step.time
            else:
                last.setdefault(block, step.time)
                last[block] = max(last[block], step.time)
    return {block: (first[block], last[block]) for block in first}


def _is_live(block: str, time: int, lifetimes: dict[str, tuple[int, int]]) -> bool:
    start, end = lifetimes.get(block, (time, time))
    return start <= time <= end


def _collect_blocks(node: CircuitNode) -> list[str]:
    if node is None:
        return []
    blocks = set(node.blocks)
    for child in node.children:
        blocks.update(_collect_blocks(child))
    return sorted(blocks, key=_block_sort_key)


def _block_sort_key(block: str) -> tuple[str, int, str]:
    if "[" in block and block.endswith("]"):
        name, idx = block[:-1].split("[", 1)
        try:
            return (name, int(idx), block)
        except ValueError:
            pass
    return (block, -1, block)


def _allocation_label(label: str) -> str:
    head = label.split(";")[0].strip()
    if ":=" in head:
        return head.split(":=", 1)[1].strip()
    if "=" in head:
        return head.split("=", 1)[1].strip()
    return head


def _release_label(label: str) -> str:
    if "release(" in label:
        return label[label.index("release(") :].rstrip()
    return "release"


def _measurement_label(label: str) -> str:
    if label.startswith("if "):
        label = label[3:]
    return label


def _is_joint_pauli_measurement(step: CircuitStep) -> bool:
    return (
        step.kind == "measurement"
        and step.label.startswith("joint Pauli ")
        and len(step.blocks) > 1
    )


def _short_label(step: CircuitStep) -> str:
    label = step.label
    replacements = [
        ("logical primitive measurement ", ""),
        ("faulty primitive measurement ", ""),
        ("logical primitive ", ""),
        ("faulty primitive ", ""),
        (" with scheme PREP + syndrome extraction", ""),
        (" + syndrome extraction", ""),
        ("joint Pauli ", ""),
        ("primitive ", ""),
        ("decode+then: ", "decode+"),
        ("decode+then/", "decode+/"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return label


def _ascii_token(step: CircuitStep | None, block: str, width: int) -> str:
    if step is None or block not in step.blocks:
        return "-" * width
    if step.kind == "se":
        return f"SE{step.index:02d}"
    if step.kind == "measurement" and block in step.measured_blocks:
        return f"M{step.index:02d}"
    if step.control_sources and block in step.blocks:
        return f"C{step.index:02d}"
    prefix = {
        "allocation": "A",
        "release": "R",
        "prepare": "P",
        "primitive": "O",
        "parallel": "B",
        "repeat": "U",
    }.get(step.kind, "X")
    return f"{prefix}{step.index:02d}"


def _step_detail(step: CircuitStep) -> str:
    details: list[str] = []
    if step.measured_blocks:
        details.append("measured: " + ", ".join(step.measured_blocks))
    if step.controlled_blocks:
        details.append("controls: " + ", ".join(step.controlled_blocks))
    if step.control_sources:
        details.append("controlled by " + ", ".join(f"M{source:02d}" for source in step.control_sources))
    return " [" + "; ".join(details) + "]" if details else ""


def _control_summary(steps: list[CircuitStep]) -> list[str]:
    lines: list[str] = []
    measurements = {step.index: step for step in steps if step.kind == "measurement"}
    for measurement in measurements.values():
        controlled_steps = [
            step.index for step in steps if measurement.index in step.control_sources
        ]
        if not controlled_steps:
            continue
        measured = ", ".join(measurement.measured_blocks or measurement.blocks)
        controlled = ", ".join(measurement.controlled_blocks) or "(no block-local then branch)"
        targets = ", ".join(f"{index:02d}" for index in controlled_steps)
        lines.append(
            f"M{measurement.index:02d}: measured {measured} -> controls {controlled}; "
            f"controlled stages: {targets}"
        )
    return lines


def _dot_label(step: CircuitStep) -> str:
    return _short_label(step)


def _color(kind: str) -> str:
    return {
        "allocation": "#e0f2fe",
        "release": "#f1f5f9",
        "prepare": "#dcfce7",
        "primitive": "#dbeafe",
        "se": "#fef08a",
        "measurement": "#fef3c7",
        "parallel": "#ede9fe",
        "repeat": "#fce7f3",
    }.get(kind, "#ffffff")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)
