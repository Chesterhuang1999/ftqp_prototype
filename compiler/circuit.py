"""Compiled dynamic circuit representation.

This is intentionally symbolic: it preserves the structural compilation rules
from the paper instead of expanding into physical gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CircuitNode:
    kind: str
    label: str
    blocks: list[str] = field(default_factory=list)
    children: list["CircuitNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def node(kind: str, label: str, blocks: list[str] | None = None, *children: CircuitNode) -> CircuitNode:
    return CircuitNode(kind=kind, label=label, blocks=blocks or [], children=list(children))
