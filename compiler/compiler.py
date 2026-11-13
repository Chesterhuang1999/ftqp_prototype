"""Compiler from validated FTQP AST to a symbolic dynamic-circuit tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from ftqp_prototype.compiler.circuit import CircuitNode, node
from ftqp_prototype.parser.ast import (
    Allocation,
    Call,
    Conditional,
    IdentifierArg,
    MapOp,
    Parallel,
    PauliLiteral,
    PrimitiveOp,
    Program,
    QubitRef,
    Reset,
    RepeatUntil,
    Sequence,
    Skip,
    Statement,
    SubcircuitDecl,
    XorConst,
)
from ftqp_prototype.resource.models import QecScheme
from ftqp_prototype.typechecker.checker import (
    CheckResult,
    RegisterBinding,
    _call_rename_tag,
    call_arg_formal_pairs,
    pauli_operation_as_parallel,
    substitute_statement,
)


@dataclass
class CompileEnv:
    registers: dict[str, RegisterBinding] = field(default_factory=dict)

    def copy(self) -> "CompileEnv":
        return CompileEnv(registers=dict(self.registers))


class Compiler:
    def __init__(self, symbols: CheckResult) -> None:
        self.schemes = symbols.schemes
        self.subcircuits = symbols.subcircuits

    def compile_program(self, program: Program) -> CircuitNode:
        if program.body is None:
            return node("empty", "empty program")
        return node("program", "program", [], self.compile_statement(program.body, CompileEnv()))

    def compile_statement(self, stmt: Statement, env: CompileEnv) -> CircuitNode:
        if isinstance(stmt, Skip):
            return node("skip", "skip")
        if isinstance(stmt, Sequence):
            return node(
                "sequence",
                "sequential composition",
                [],
                *(self.compile_statement(child, env) for child in stmt.statements),
            )
        if isinstance(stmt, Allocation):
            scheme = self.schemes[stmt.scheme]
            child_env = env.copy()
            child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
            blocks = [f"{stmt.register}[{i}]" for i in range(stmt.count)]
            body = self.compile_statement(stmt.body, child_env)
            return node(
                "allocation",
                f"logical {stmt.register}[{stmt.count}] := new {stmt.scheme}; release({stmt.register})",
                blocks,
                body,
            )
        if isinstance(stmt, Reset):
            block = self._block_label(stmt.target, env)
            return node("prepare", f"{stmt.keyword}({block}) with scheme PREP + syndrome extraction", [block])
        if isinstance(stmt, PrimitiveOp):
            expanded = pauli_operation_as_parallel(stmt)
            if expanded is not None:
                return self.compile_statement(expanded, env)
            block = self._block_label(stmt.targets[0], env) if stmt.targets else "<?>"
            scheme = env.registers[stmt.targets[0].name].scheme
            op = scheme.find_operation(stmt.name)
            op_kind = op.kind if op else "primitive"
            targets = ", ".join(ref.display() for ref in stmt.targets)
            label = f"{op_kind} primitive {stmt.name}[{targets}]"
            if op_kind == "logical":
                label += " + syndrome extraction"
            return node("primitive", label, [block])
        if isinstance(stmt, MapOp):
            blocks = sorted({self._block_label(ref, env) for ref in stmt.targets})
            return node(
                "map",
                f"map unary primitive {stmt.name} sequentially over {len(stmt.targets)} targets",
                blocks,
            )
        if isinstance(stmt, XorConst):
            block = self._block_label(stmt.target, env)
            return node(
                "map",
                f"xor {stmt.width}-bit classical mask using "
                f"{stmt.value.bit_count()} sequential X primitives",
                [block],
            )
        if isinstance(stmt, Parallel):
            return node(
                "parallel",
                "parallel composition",
                [],
                self.compile_statement(stmt.left, env.copy()),
                self.compile_statement(stmt.right, env.copy()),
            )
        if isinstance(stmt, Conditional):
            blocks = sorted({self._block_label(ref, env) for ref in stmt.targets})
            if _is_pauli(stmt.measurement.name):
                meas_kind = "joint Pauli"
            else:
                op_kind = "primitive"
                if stmt.targets:
                    binding = env.registers.get(stmt.targets[0].name)
                    op = binding.scheme.find_operation(stmt.measurement.name) if binding else None
                    op_kind = op.kind if op else op_kind
                meas_kind = f"{op_kind} primitive measurement"
            label = f"if {meas_kind} {stmt.measurement.display()}[{_refs(stmt.targets)}]"
            return node(
                "conditional",
                label,
                blocks,
                node("during", "execute while measurement/decoding is pending", [], self.compile_statement(stmt.during, env.copy())),
                node("then", "conditional branch after outcome is available", [], self.compile_statement(stmt.then_branch, env.copy())),
            )
        if isinstance(stmt, RepeatUntil):
            return node(
                "repeat",
                f"repeat until {stmt.threshold:g}",
                sorted(self._footprint_labels(stmt.body, env)),
                self.compile_statement(stmt.body, env.copy()),
            )
        if isinstance(stmt, Call):
            decl = self.subcircuits[stmt.name]
            instantiated = self._instantiate_call(stmt, decl, env)
            return node(
                "call",
                f"call {stmt.name}(...) expanded by substitution",
                [],
                self.compile_statement(instantiated, env.copy()),
            )
        return node("unknown", type(stmt).__name__)

    def _instantiate_call(self, call: Call, decl: SubcircuitDecl, env: CompileEnv) -> Statement:
        qsubs: dict[str, QubitRef] = {}
        psubs: dict[str, str] = {}
        pairs = call_arg_formal_pairs(decl, call.args)
        if pairs is None:
            pairs = list(zip(decl.params, call.args, strict=True))
        for formal, arg in pairs:
            kind = decl.param_types.get(formal)
            if isinstance(arg, QubitRef):
                qsubs[formal] = arg
            elif isinstance(arg, PauliLiteral):
                psubs[formal] = arg.value
            elif isinstance(arg, IdentifierArg):
                if kind == "pauli" or (kind is None and _is_pauli(arg.name)):
                    psubs[formal] = arg.name
                else:
                    qsubs[formal] = QubitRef(arg.name, span=arg.span)
        return (
            substitute_statement(
                decl.body,
                qsubs,
                psubs,
                reserved=set(env.registers),
                rename_tag=_call_rename_tag(call),
            )
            if decl.body
            else Skip()
        )

    def _block_label(self, ref: QubitRef, env: CompileEnv) -> str:
        binding = env.registers.get(ref.name)
        if binding is None:
            return ref.display()
        if ref.logical is None and ref.block is not None and binding.count == 1:
            return f"{ref.name}[0]"
        block = ref.block if ref.block is not None else 0
        return f"{ref.name}[{block}]"

    def _footprint_labels(self, stmt: Statement, env: CompileEnv) -> set[str]:
        if isinstance(stmt, Skip):
            return set()
        if isinstance(stmt, Sequence):
            out: set[str] = set()
            for child in stmt.statements:
                out |= self._footprint_labels(child, env)
            return out
        if isinstance(stmt, Allocation):
            scheme = self.schemes[stmt.scheme]
            child_env = env.copy()
            child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
            local = {f"{stmt.register}[{i}]" for i in range(stmt.count)}
            return self._footprint_labels(stmt.body, child_env) - local
        if isinstance(stmt, Reset):
            return {self._block_label(stmt.target, env)}
        if isinstance(stmt, PrimitiveOp):
            return {self._block_label(ref, env) for ref in stmt.targets}
        if isinstance(stmt, MapOp):
            return {self._block_label(ref, env) for ref in stmt.targets}
        if isinstance(stmt, XorConst):
            return {self._block_label(stmt.target, env)}
        if isinstance(stmt, Parallel):
            return self._footprint_labels(stmt.left, env) | self._footprint_labels(stmt.right, env)
        if isinstance(stmt, Conditional):
            return (
                {self._block_label(ref, env) for ref in stmt.targets}
                | self._footprint_labels(stmt.during, env)
                | self._footprint_labels(stmt.then_branch, env)
            )
        if isinstance(stmt, RepeatUntil):
            return self._footprint_labels(stmt.body, env)
        if isinstance(stmt, Call):
            decl = self.subcircuits.get(stmt.name)
            if decl is None:
                return set()
            return self._footprint_labels(self._instantiate_call(stmt, decl, env), env.copy())
        return set()


def _refs(refs: list[QubitRef]) -> str:
    return ", ".join(ref.display() for ref in refs)


def _is_pauli(value: str) -> bool:
    return bool(value) and all(ch in "IXYZ" for ch in value)
