"""Abstract syntax tree for the FTQP prototype language.

The AST mirrors the formal syntax in the theory documents, with a few concrete
syntax conveniences such as brace-delimited allocation scopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


SubcircuitParamKind = Literal["pauli", "codeblock"]


@dataclass(frozen=True)
class SourceSpan:
    """A lightweight source location used in diagnostics."""

    filename: str
    line: int
    column: int

    def format(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


@dataclass
class Node:
    span: SourceSpan | None = None

    def loc(self) -> str:
        return self.span.format() if self.span else "<unknown>"


@dataclass(frozen=True)
class QubitRef:
    """Reference to a logical qubit or code block.

    `q` is represented as no indices, `q[i]` as a block index, and
    `q[i][j]` as a specific logical qubit in a block. In logical-qubit
    contexts, semantic analysis interprets `q[i]` as logical qubit `i` when
    `q` is known to contain exactly one code block.
    """

    name: str
    block: int | None = None
    logical: int | None = None
    span: SourceSpan | None = None

    def display(self) -> str:
        out = self.name
        if self.block is not None:
            out += f"[{self.block}]"
        if self.logical is not None:
            out += f"[{self.logical}]"
        return out


@dataclass(frozen=True)
class PauliLiteral:
    value: str
    span: SourceSpan | None = None


@dataclass(frozen=True)
class IdentifierArg:
    """Bare identifier in a call argument before type resolution."""

    name: str
    span: SourceSpan | None = None


CallArg = QubitRef | PauliLiteral | IdentifierArg


@dataclass
class HardwareDecl(Node):
    name: str = ""
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class OperationDecl(Node):
    name: str = ""
    arity: int = 0
    time: str = "0"
    ancilla: str = "0"
    n_se: str | None = None
    p_error: str | None = None
    kind: Literal["logical", "faulty"] = "logical"


@dataclass
class QECDecl(Node):
    name: str = ""
    hardware: str | None = None
    n: int = 0
    k: int = 0
    d: int = 0
    se_ancilla: str = "0"
    t_se: str = "0"
    prep_time: str = "0"
    prep_n_se: str = "0"
    prep_ancilla: str = "0"
    logical_ops: dict[str, OperationDecl] = field(default_factory=dict)
    faulty_ops: dict[str, OperationDecl] = field(default_factory=dict)
    t_dec: str = "0"
    p_l: str = "0"
    n_aux: str = "0"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubcircuitDecl(Node):
    name: str = ""
    params: list[str] = field(default_factory=list)
    param_types: dict[str, SubcircuitParamKind] = field(default_factory=dict)
    body: Statement | None = None


@dataclass
class ProgramDecl(Node):
    body: Statement | None = None


TopLevelDecl = HardwareDecl | QECDecl | SubcircuitDecl | ProgramDecl


@dataclass
class Program(Node):
    declarations: list[TopLevelDecl] = field(default_factory=list)

    @property
    def body(self) -> Statement | None:
        for decl in self.declarations:
            if isinstance(decl, ProgramDecl):
                return decl.body
        return None

    @property
    def hardware_decls(self) -> list[HardwareDecl]:
        return [d for d in self.declarations if isinstance(d, HardwareDecl)]

    @property
    def qec_decls(self) -> list[QECDecl]:
        return [d for d in self.declarations if isinstance(d, QECDecl)]

    @property
    def subcircuits(self) -> list[SubcircuitDecl]:
        return [d for d in self.declarations if isinstance(d, SubcircuitDecl)]


@dataclass
class Statement(Node):
    pass


@dataclass
class Skip(Statement):
    pass


@dataclass
class Sequence(Statement):
    statements: list[Statement] = field(default_factory=list)


@dataclass
class Allocation(Statement):
    register: str = ""
    count: int = 1
    scheme: str = ""
    body: Statement = field(default_factory=Skip)


@dataclass
class Reset(Statement):
    target: QubitRef = field(default_factory=lambda: QubitRef(""))
    keyword: str = "prepare"


@dataclass
class PrimitiveOp(Statement):
    name: str = ""
    targets: list[QubitRef] = field(default_factory=list)


@dataclass
class MapOp(Statement):
    """Sequentially apply one unary primitive to each target."""

    name: str = ""
    targets: list[QubitRef] = field(default_factory=list)


@dataclass
class XorConst(Statement):
    """Apply X to the set bits of a compact classical mask."""

    target: QubitRef = field(default_factory=lambda: QubitRef(""))
    width: int = 0
    value: int = 0


@dataclass
class Parallel(Statement):
    left: Statement = field(default_factory=Skip)
    right: Statement = field(default_factory=Skip)


@dataclass
class Measurement(Node):
    """Measurement expression M(name)[targets].

    The checker resolves `name` to either a joint Pauli string or a primitive
    measurement operation after subcircuit substitution.
    """

    name: str = ""
    was_string: bool = False

    def display(self) -> str:
        return f"M({self.name})"


@dataclass
class Conditional(Statement):
    measurement: Measurement = field(default_factory=Measurement)
    targets: list[QubitRef] = field(default_factory=list)
    during: Statement = field(default_factory=Skip)
    then_branch: Statement = field(default_factory=Skip)


@dataclass
class RepeatUntil(Statement):
    body: Statement = field(default_factory=Skip)
    threshold: float = 0.0


@dataclass
class Call(Statement):
    name: str = ""
    args: list[CallArg] = field(default_factory=list)


def as_sequence(statements: Iterable[Statement], span: SourceSpan | None = None) -> Statement:
    """Collapse an iterable of statements into skip, a single item, or Sequence."""

    items = [stmt for stmt in statements if not isinstance(stmt, Skip)]
    if not items:
        return Skip(span=span)
    if len(items) == 1:
        return items[0]
    return Sequence(span=span or items[0].span, statements=items)
