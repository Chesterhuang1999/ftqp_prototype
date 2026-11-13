"""Static semantic checks for FTQP programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ftqp_prototype.parser.ast import (
    Allocation,
    Call,
    CallArg,
    Conditional,
    HardwareDecl,
    IdentifierArg,
    MapOp,
    Measurement,
    Parallel,
    PauliLiteral,
    PrimitiveOp,
    Program,
    ProgramDecl,
    QECDecl,
    QubitRef,
    Reset,
    RepeatUntil,
    Sequence,
    Skip,
    SourceSpan,
    Statement,
    SubcircuitDecl,
    XorConst,
)
from ftqp_prototype.resource.expressions import ExpressionError
from ftqp_prototype.resource.models import HardwareConfig, QecScheme, RuntimeOperation
from ftqp_prototype.typechecker.errors import Diagnostic, ValidationError

BlockId = tuple[str, int]
LogicalId = tuple[str, int, int]
CallExpected = Literal["pauli", "qubit", "codeblock"]


@dataclass
class RegisterBinding:
    name: str
    count: int
    scheme: QecScheme
    span: SourceSpan | None = None


@dataclass
class CheckEnv:
    registers: dict[str, RegisterBinding] = field(default_factory=dict)

    def copy(self) -> "CheckEnv":
        return CheckEnv(registers=dict(self.registers))


@dataclass
class CheckResult:
    hardware: dict[str, HardwareConfig]
    schemes: dict[str, QecScheme]
    subcircuits: dict[str, SubcircuitDecl]


@dataclass(frozen=True)
class FormalUsage:
    pauli: set[str]
    qubit: set[str]

    def expected_for(self, name: str) -> Literal["pauli", "qubit"] | None:
        in_pauli = name in self.pauli
        in_qubit = name in self.qubit
        if in_pauli and not in_qubit:
            return "pauli"
        if in_qubit and not in_pauli:
            return "qubit"
        return None


class TypeChecker:
    """Validate a parsed FTQP program and build runtime declarations."""

    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []
        self.hardware: dict[str, HardwareConfig] = {}
        self.schemes: dict[str, QecScheme] = {}
        self.subcircuits: dict[str, SubcircuitDecl] = {}
        self.formal_usage: dict[str, FormalUsage] = {}
        self.expansion_stack: list[str] = []
        self.checked_call_instances: set[tuple[object, ...]] = set()

    def check(self, program: Program) -> CheckResult:
        self.diagnostics = []
        self.hardware = {}
        self.schemes = {}
        self.subcircuits = {}
        self.expansion_stack = []
        self.checked_call_instances = set()
        program_seen = False

        for decl in program.declarations:
            if isinstance(decl, HardwareDecl):
                self._declare_hardware(decl)
            elif isinstance(decl, QECDecl):
                self._declare_qec(decl)
            elif isinstance(decl, SubcircuitDecl):
                self._declare_subcircuit(decl)
            elif isinstance(decl, ProgramDecl):
                if program_seen:
                    self.error(decl, "only one program block is allowed")
                program_seen = True
                self._check_statement(decl.body or Skip(), CheckEnv())

        if not program_seen:
            self.error(program, "missing top-level program { ... } block")

        self._check_subcircuit_graph()
        self._check_all_declared_subcircuit_calls()

        if self.diagnostics:
            raise ValidationError(self.diagnostics)
        return CheckResult(self.hardware, self.schemes, self.subcircuits)

    def error(self, node_or_span: object, message: str) -> None:
        span = getattr(node_or_span, "span", None)
        location = span.format() if span else "<unknown>"
        self.diagnostics.append(Diagnostic(message, location))

    def _declare_hardware(self, decl: HardwareDecl) -> None:
        if decl.name in self.hardware:
            self.error(decl, f"duplicate hardware configuration {decl.name!r}")
            return
        required = {"p1", "p2", "pm", "t1", "t2", "tm"}
        missing = sorted(required - set(decl.params))
        if missing:
            self.error(decl, f"hardware {decl.name!r} is missing parameters: {', '.join(missing)}")
            return
        for key in ["p1", "p2", "pm"]:
            value = decl.params[key]
            if not (0 <= value <= 1):
                self.error(decl, f"hardware {decl.name!r} has invalid error rate {key}={value}")
        for key in ["t1", "t2", "tm"]:
            value = decl.params[key]
            if value < 0:
                self.error(decl, f"hardware {decl.name!r} has negative duration {key}={value}")
        self.hardware[decl.name] = HardwareConfig.from_decl(decl)

    def _declare_qec(self, decl: QECDecl) -> None:
        if decl.name in self.schemes:
            self.error(decl, f"duplicate QEC scheme {decl.name!r}")
            return
        if decl.hardware is None:
            self.error(
                decl,
                f"QEC scheme {decl.name!r} must explicitly bind hardware with 'using <hardware>'",
            )
            return
        if decl.hardware not in self.hardware:
            self.error(
                decl,
                f"QEC scheme {decl.name!r} references undeclared hardware {decl.hardware!r}",
            )
            return
        if decl.n <= 0 or decl.k <= 0 or decl.d <= 0 or decl.k > decl.n:
            self.error(decl, f"QEC scheme {decl.name!r} has invalid [[n,k,d]] parameters")
            return
        scheme = QecScheme.from_decl(decl, self.hardware[decl.hardware])
        self._validate_scheme_expressions(decl, scheme)
        self.schemes[decl.name] = scheme

    def _validate_scheme_expressions(self, decl: QECDecl, scheme: QecScheme) -> None:
        fields = {
            "SE.#ancilla": decl.se_ancilla,
            "SE.tSE": decl.t_se,
            "PREP.time": decl.prep_time,
            "PREP.#SE": decl.prep_n_se,
            "PREP.#ancilla": decl.prep_ancilla,
            "tdec": decl.t_dec,
            "pL": decl.p_l,
            "naux": decl.n_aux,
        }
        for op in list(decl.logical_ops.values()) + list(decl.faulty_ops.values()):
            if op.arity <= 0:
                self.error(op, f"operation {op.name!r} must have positive arity")
            fields[f"{op.name}.time"] = op.time
            fields[f"{op.name}.#ancilla"] = op.ancilla
            if op.kind == "logical":
                fields[f"{op.name}.#SE"] = op.n_se or "0"
            else:
                fields[f"{op.name}.pF"] = op.p_error or "0"
        for field, expr in fields.items():
            try:
                value = scheme.eval(expr)
            except ExpressionError as exc:
                self.error(decl, f"cannot evaluate {decl.name}.{field}: {exc}")
                continue
            if field.endswith(("pL", "pF")) and not (0 <= value <= 1):
                self.error(decl, f"{decl.name}.{field} evaluates to invalid probability {value}")
            if "#SE" in field and value < 0:
                self.error(decl, f"{decl.name}.{field} evaluates to negative cycle count {value}")
            if field == "naux" and value < 0:
                self.error(decl, f"{decl.name}.{field} evaluates to negative auxiliary qubit count {value}")

    def _declare_subcircuit(self, decl: SubcircuitDecl) -> None:
        if decl.name in self.subcircuits:
            self.error(decl, f"duplicate subcircuit {decl.name!r}")
            return
        if len(set(decl.params)) != len(decl.params):
            self.error(decl, f"subcircuit {decl.name!r} has duplicate formal parameters")
        unknown_typed = sorted(set(decl.param_types) - set(decl.params))
        if unknown_typed:
            self.error(
                decl,
                f"subcircuit {decl.name!r} has types for undeclared parameters: "
                + ", ".join(unknown_typed),
            )
        self.subcircuits[decl.name] = decl
        self.formal_usage[decl.name] = _collect_formal_usage(decl)
        self._check_closed_body(decl)

    def _check_closed_body(self, decl: SubcircuitDecl) -> None:
        """Subcircuit bodies may only use their parameters and their own allocations.

        Without this, a call's footprint (its argument blocks) could miss a
        block the body touches, and `call f(b) || X[g]` could race on `g`.
        """
        params = set(decl.params)
        reported: set[str] = set()
        for name, node in _free_register_refs(decl.body or Skip(), params, params):
            if name in reported:
                continue
            reported.add(name)
            self.error(
                node,
                f"subcircuit {decl.name!r} uses register {name!r}, which is neither a "
                "parameter nor allocated inside the subcircuit; pass it as an argument",
            )

    def _check_subcircuit_graph(self) -> None:
        graph: dict[str, set[str]] = {name: set() for name in self.subcircuits}
        for name, decl in self.subcircuits.items():
            for call in _iter_calls(decl.body or Skip()):
                if call.name not in self.subcircuits:
                    self.error(call, f"call to undeclared subcircuit {call.name!r}")
                else:
                    graph[name].add(call.name)

        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle = stack[stack.index(name) :] + [name]
                self.error(self.subcircuits[name], "recursive subcircuit cycle: " + " -> ".join(cycle))
                return
            visiting.add(name)
            stack.append(name)
            for nxt in graph.get(name, set()):
                dfs(nxt)
            stack.pop()
            visiting.remove(name)
            visited.add(name)

        for name in graph:
            dfs(name)

    def _check_all_declared_subcircuit_calls(self) -> None:
        for decl in self.subcircuits.values():
            for call in _iter_calls(decl.body or Skip()):
                target = self.subcircuits.get(call.name)
                if target and len(call.args) != len(target.params):
                    self.error(
                        call,
                        f"call to {call.name!r} passes {len(call.args)} arguments, "
                        f"but declaration expects {len(target.params)}",
                    )

    def _check_statement(self, stmt: Statement, env: CheckEnv) -> None:
        if isinstance(stmt, Skip):
            return
        if isinstance(stmt, Sequence):
            for child in stmt.statements:
                self._check_statement(child, env)
            return
        if isinstance(stmt, Allocation):
            self._check_allocation(stmt, env)
            return
        if isinstance(stmt, Reset):
            self._resolve_block(stmt.target, env, "prepare/reset target")
            if stmt.target.logical is not None:
                self.error(stmt, f"{stmt.keyword} expects a code block, got {stmt.target.display()}")
            return
        if isinstance(stmt, PrimitiveOp):
            self._check_primitive(stmt, env)
            return
        if isinstance(stmt, MapOp):
            self._check_map_op(stmt, env)
            return
        if isinstance(stmt, XorConst):
            self._check_xor_const(stmt, env)
            return
        if isinstance(stmt, Parallel):
            self._check_parallel(stmt, env)
            return
        if isinstance(stmt, Conditional):
            self._check_conditional(stmt, env)
            return
        if isinstance(stmt, RepeatUntil):
            self._check_repeat_until(stmt, env)
            return
        if isinstance(stmt, Call):
            self._check_call(stmt, env)
            return
        self.error(stmt, f"unsupported statement type {type(stmt).__name__}")

    def _check_allocation(self, stmt: Allocation, env: CheckEnv) -> None:
        if stmt.count <= 0:
            self.error(stmt, f"allocation of register {stmt.register!r} must have positive block count")
        if stmt.register in env.registers:
            self.error(stmt, f"register {stmt.register!r} is already live in this scope")
            return
        scheme = self.schemes.get(stmt.scheme)
        if scheme is None:
            self.error(stmt, f"allocation uses undeclared QEC scheme {stmt.scheme!r}")
            return
        env.registers[stmt.register] = RegisterBinding(
            stmt.register, stmt.count, scheme, stmt.span
        )
        self._check_statement(stmt.body, env)
        env.registers.pop(stmt.register, None)

    def _check_primitive(self, stmt: PrimitiveOp, env: CheckEnv) -> None:
        if _is_pauli(stmt.name):
            if len(stmt.name) != len(stmt.targets):
                self.error(
                    stmt,
                    f"Pauli operation {stmt.name!r} has arity {len(stmt.name)}, "
                    f"got {len(stmt.targets)} targets",
                )
                return
            expanded = pauli_operation_as_parallel(stmt)
            if expanded is not None:
                self._check_statement(expanded, env)
                return

        logicals = [self._resolve_logical(ref, env, f"target of {stmt.name}") for ref in stmt.targets]
        if any(item is None for item in logicals):
            return
        if len(set(logicals)) != len(logicals):
            self.error(stmt, f"operation {stmt.name!r} has duplicate target logical qubits")
            return
        blocks = {(reg, block) for reg, block, _ in logicals if reg is not None}
        if len(blocks) != 1:
            self.error(stmt, f"operation {stmt.name!r} must act within one code block")
            return
        block = next(iter(blocks))
        scheme = env.registers[block[0]].scheme
        op = scheme.find_operation(stmt.name)
        if op is None:
            self.error(
                stmt,
                f"scheme {scheme.name!r} does not support primitive operation {stmt.name!r}",
            )
            return
        if op.arity != len(stmt.targets):
            self.error(
                stmt,
                f"operation {stmt.name!r} expects arity {op.arity}, got {len(stmt.targets)}",
            )

    def _check_map_op(self, stmt: MapOp, env: CheckEnv) -> None:
        if not stmt.targets:
            self.error(stmt, f"map {stmt.name!r} requires at least one target")
            return
        logicals = [
            self._resolve_logical(ref, env, f"target of map {stmt.name}")
            for ref in stmt.targets
        ]
        if any(item is None for item in logicals):
            return
        if len(set(logicals)) != len(logicals):
            self.error(stmt, f"map {stmt.name!r} has duplicate target logical qubits")
            return
        for ref in stmt.targets:
            self._check_primitive(
                PrimitiveOp(span=stmt.span, name=stmt.name, targets=[ref]),
                env,
            )

    def _check_xor_const(self, stmt: XorConst, env: CheckEnv) -> None:
        if stmt.width <= 0:
            self.error(stmt, "xor range width must be positive")
            return
        if stmt.value < 0 or stmt.value.bit_length() > stmt.width:
            self.error(stmt, "xor mask does not fit in the declared range width")
            return
        start = self._resolve_logical(stmt.target, env, "xor target range")
        if start is None:
            return
        binding = env.registers[start[0]]
        if start[2] + stmt.width > binding.scheme.k:
            self.error(
                stmt,
                f"xor range [{start[2]}:{stmt.width}] exceeds scheme "
                f"{binding.scheme.name!r} with k={binding.scheme.k}",
            )
            return
        op = binding.scheme.find_operation("X")
        if op is None:
            self.error(
                stmt,
                f"scheme {binding.scheme.name!r} does not support primitive operation 'X'",
            )
        elif op.arity != 1:
            self.error(stmt, "xor source sugar requires unary primitive X")

    def _check_parallel(self, stmt: Parallel, env: CheckEnv) -> None:
        left_env = env.copy()
        right_env = env.copy()
        before = len(self.diagnostics)
        self._check_statement(stmt.left, left_env)
        self._check_statement(stmt.right, right_env)
        if len(self.diagnostics) > before:
            return
        left = self.footprint(stmt.left, env)
        right = self.footprint(stmt.right, env)
        overlap = left & right
        if overlap:
            pretty = ", ".join(_fmt_block(block) for block in sorted(overlap))
            self.error(stmt, f"parallel branches touch overlapping code blocks: {pretty}")

    def _check_conditional(self, stmt: Conditional, env: CheckEnv) -> None:
        self._check_measurement(stmt.measurement, stmt.targets, stmt, env)
        during_env = env.copy()
        then_env = env.copy()
        self._check_statement(stmt.during, during_env)
        self._check_statement(stmt.then_branch, then_env)

    def _check_repeat_until(self, stmt: RepeatUntil, env: CheckEnv) -> None:
        if stmt.threshold <= 0.0:
            self.error(stmt, "repeat-until threshold must be positive")
            return
        before = len(self.diagnostics)
        self._check_statement(stmt.body, env.copy())
        if len(self.diagnostics) == before and not self.footprint(stmt.body, env):
            self.error(stmt, "repeat-until body must touch at least one live code block")

    def _check_measurement(
        self, measurement: Measurement, targets: list[QubitRef], owner: Statement, env: CheckEnv
    ) -> None:
        logicals = [self._resolve_logical(ref, env, f"target of {measurement.display()}") for ref in targets]
        if any(item is None for item in logicals):
            return
        if len(set(logicals)) != len(logicals):
            self.error(owner, f"{measurement.display()} has duplicate target logical qubits")
            return
        if _is_pauli(measurement.name):
            if len(measurement.name) != len(targets):
                self.error(
                    owner,
                    f"joint Pauli {measurement.display()} has arity {len(measurement.name)}, "
                    f"got {len(targets)} targets",
                )
            return

        blocks = {(reg, block) for reg, block, _ in logicals if reg is not None}
        if len(blocks) != 1:
            self.error(owner, f"primitive measurement {measurement.display()} must stay in one code block")
            return
        block = next(iter(blocks))
        scheme = env.registers[block[0]].scheme
        op = scheme.find_operation(measurement.name)
        if op is None:
            self.error(
                owner,
                f"scheme {scheme.name!r} does not support primitive measurement {measurement.name!r}",
            )
            return
        if op.arity != len(targets):
            self.error(
                owner,
                f"measurement {measurement.name!r} expects arity {op.arity}, got {len(targets)}",
            )

    def _check_call(self, stmt: Call, env: CheckEnv) -> None:
        decl = self.subcircuits.get(stmt.name)
        if decl is None:
            self.error(stmt, f"call to undeclared subcircuit {stmt.name!r}")
            return
        if len(stmt.args) != len(decl.params):
            self.error(
                stmt,
                f"call to {stmt.name!r} passes {len(stmt.args)} arguments, "
                f"but declaration expects {len(decl.params)}",
            )
            return
        if stmt.name in self.expansion_stack:
            cycle = " -> ".join(self.expansion_stack + [stmt.name])
            self.error(stmt, f"recursive subcircuit expansion is not allowed: {cycle}")
            return
        cache_key = (
            stmt.name,
            *(_call_argument_cache_key(arg, env) for arg in stmt.args),
        )
        if cache_key in self.checked_call_instances:
            return
        before = len(self.diagnostics)
        body = self._instantiate_call(stmt, decl, env)
        if body is not None:
            self.expansion_stack.append(stmt.name)
            self._check_statement(body, env.copy())
            self.expansion_stack.pop()
        if len(self.diagnostics) == before:
            self.checked_call_instances.add(cache_key)

    def _instantiate_call(
        self, call: Call, decl: SubcircuitDecl, env: CheckEnv
    ) -> Statement | None:
        usage = self.formal_usage.get(decl.name) or FormalUsage(set(), set())
        qsubs: dict[str, QubitRef] = {}
        psubs: dict[str, str] = {}
        pairs = call_arg_formal_pairs(decl, call.args)
        if pairs is None:
            self.error(
                call,
                f"call to {decl.name!r} cannot match arguments to typed parameters",
            )
            return None
        for formal, arg in pairs:
            expected = _expected_formal_kind(decl, usage, formal)
            resolved = self._resolve_call_arg(arg, expected, env)
            if resolved is None:
                return None
            if isinstance(resolved, QubitRef):
                qsubs[formal] = resolved
            else:
                psubs[formal] = resolved.value
        return substitute_statement(
            decl.body or Skip(),
            qsubs,
            psubs,
            reserved=set(env.registers),
            rename_tag=_call_rename_tag(call),
        )

    def _resolve_call_arg(
        self,
        arg: CallArg,
        expected: CallExpected | None,
        env: CheckEnv,
    ) -> QubitRef | PauliLiteral | None:
        if isinstance(arg, QubitRef):
            if expected == "pauli":
                self.error_span(arg.span, f"expected Pauli argument, got qubit {arg.display()}")
                return None
            if expected == "codeblock":
                if arg.logical is not None:
                    self.error_span(
                        arg.span, f"expected code block argument, got logical qubit {arg.display()}"
                    )
                    return None
                if self._resolve_block(arg, env, "subcircuit argument") is None:
                    return None
            else:
                if self._resolve_logical(arg, env, "subcircuit argument") is None:
                    return None
            return arg
        if isinstance(arg, PauliLiteral):
            if not _is_pauli(arg.value):
                self.error_span(arg.span, f"invalid Pauli string {arg.value!r}")
                return None
            if expected in {"qubit", "codeblock"}:
                noun = "code block" if expected == "codeblock" else "qubit"
                self.error_span(arg.span, f"expected {noun} argument, got Pauli {arg.value!r}")
                return None
            return arg
        if isinstance(arg, IdentifierArg):
            if expected == "pauli":
                if not _is_pauli(arg.name):
                    self.error_span(arg.span, f"expected a Pauli string such as XZX, got {arg.name!r}")
                    return None
                return PauliLiteral(arg.name, arg.span)
            qref = QubitRef(arg.name, span=arg.span)
            if expected == "codeblock":
                if self._resolve_block(qref, env, "subcircuit argument") is None:
                    return None
            else:
                if self._resolve_logical(qref, env, "subcircuit argument") is None:
                    return None
            return qref
        return None

    def error_span(self, span: SourceSpan | None, message: str) -> None:
        self.diagnostics.append(Diagnostic(message, span.format() if span else "<unknown>"))

    def _resolve_block(self, ref: QubitRef, env: CheckEnv, context: str) -> BlockId | None:
        binding = env.registers.get(ref.name)
        if binding is None:
            self.error_span(ref.span, f"unknown register {ref.name!r} in {context}")
            return None
        if ref.block is None:
            if binding.count != 1:
                self.error_span(
                    ref.span,
                    f"{ref.display()} is ambiguous: register has {binding.count} blocks; use {ref.name}[i]",
                )
                return None
            return (ref.name, 0)
        if ref.block >= binding.count:
            self.error_span(
                ref.span,
                f"block index {ref.block} out of range for register {ref.name!r} "
                f"with {binding.count} blocks",
            )
            return None
        return (ref.name, ref.block)

    def _resolve_logical(
        self, ref: QubitRef, env: CheckEnv, context: str
    ) -> LogicalId | None:
        binding = env.registers.get(ref.name)
        if binding is None:
            self.error_span(ref.span, f"unknown register {ref.name!r} in {context}")
            return None
        if ref.logical is None and ref.block is not None and binding.count == 1:
            if ref.block >= binding.scheme.k:
                self.error_span(
                    ref.span,
                    f"logical index {ref.block} out of range for scheme {binding.scheme.name!r} "
                    f"with k={binding.scheme.k}",
                )
                return None
            return (ref.name, 0, ref.block)
        block = self._resolve_block(ref, env, context)
        if block is None:
            return None
        binding = env.registers[block[0]]
        if ref.logical is None:
            if binding.scheme.k != 1:
                self.error_span(
                    ref.span,
                    f"{ref.display()} names a code block with k={binding.scheme.k}; "
                    "use q[i][j] to select a logical qubit",
                )
                return None
            logical = 0
        else:
            if ref.logical >= binding.scheme.k:
                self.error_span(
                    ref.span,
                    f"logical index {ref.logical} out of range for scheme {binding.scheme.name!r} "
                    f"with k={binding.scheme.k}",
                )
                return None
            logical = ref.logical
        return (block[0], block[1], logical)

    def footprint(self, stmt: Statement, env: CheckEnv) -> set[BlockId]:
        if isinstance(stmt, Skip):
            return set()
        if isinstance(stmt, Sequence):
            out: set[BlockId] = set()
            for child in stmt.statements:
                out |= self.footprint(child, env)
            return out
        if isinstance(stmt, Allocation):
            scheme = self.schemes.get(stmt.scheme)
            if not scheme:
                return set()
            child_env = env.copy()
            child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
            local = {(stmt.register, i) for i in range(stmt.count)}
            return self.footprint(stmt.body, child_env) - local
        if isinstance(stmt, Reset):
            block = self._resolve_block(stmt.target, env, "footprint")
            return {block} if block else set()
        if isinstance(stmt, PrimitiveOp):
            return {item[:2] for item in (self._resolve_logical(r, env, "footprint") for r in stmt.targets) if item}
        if isinstance(stmt, MapOp):
            return {item[:2] for item in (self._resolve_logical(r, env, "footprint") for r in stmt.targets) if item}
        if isinstance(stmt, XorConst):
            logical = self._resolve_logical(stmt.target, env, "footprint")
            return {logical[:2]} if logical else set()
        if isinstance(stmt, Parallel):
            return self.footprint(stmt.left, env) | self.footprint(stmt.right, env)
        if isinstance(stmt, Conditional):
            measured = {
                item[:2]
                for item in (self._resolve_logical(r, env, "footprint") for r in stmt.targets)
                if item
            }
            return measured | self.footprint(stmt.during, env) | self.footprint(stmt.then_branch, env)
        if isinstance(stmt, RepeatUntil):
            return self.footprint(stmt.body, env)
        if isinstance(stmt, Call):
            out: set[BlockId] = set()
            decl = self.subcircuits.get(stmt.name)
            usage = self.formal_usage.get(stmt.name) or FormalUsage(set(), set())
            pairs = call_arg_formal_pairs(decl, stmt.args) if decl else None
            if pairs is None:
                pairs = list(zip(decl.params, stmt.args)) if decl else [("", arg) for arg in stmt.args]
            for formal, arg in pairs:
                expected = _expected_formal_kind(decl, usage, formal) if decl else None
                if expected == "pauli":
                    continue
                if isinstance(arg, QubitRef):
                    if expected == "codeblock":
                        block = self._resolve_block(arg, env, "call footprint")
                    else:
                        logical = self._resolve_logical(arg, env, "call footprint")
                        block = logical[:2] if logical else None
                    if block is not None:
                        out.add(block)
                elif isinstance(arg, IdentifierArg):
                    qref = QubitRef(arg.name, span=arg.span)
                    if expected == "codeblock":
                        block = self._resolve_block(qref, env, "call footprint")
                    else:
                        logical = self._resolve_logical(qref, env, "call footprint")
                        block = logical[:2] if logical else None
                    if block is not None:
                        out.add(block)
            return out
        return set()

def substitute_statement(
    stmt: Statement,
    qsubs: dict[str, QubitRef],
    psubs: dict[str, str],
    rensubs: dict[str, str] | None = None,
    reserved: set[str] | None = None,
    rename_tag: str = "local",
) -> Statement:
    """Instantiate a subcircuit body by replacing formal Pauli/qubit names."""

    rensubs = rensubs or {}
    reserved = reserved or set()

    if isinstance(stmt, Skip):
        return Skip(span=stmt.span)
    if isinstance(stmt, Sequence):
        return Sequence(
            span=stmt.span,
            statements=[
                substitute_statement(child, qsubs, psubs, rensubs, reserved, rename_tag)
                for child in stmt.statements
            ],
        )
    if isinstance(stmt, Allocation):
        child_qsubs = dict(qsubs)
        child_qsubs.pop(stmt.register, None)
        child_rensubs = dict(rensubs)
        child_rensubs.pop(stmt.register, None)
        register = stmt.register
        if register in reserved:
            register = _fresh_register_name(register, reserved, rename_tag)
            child_rensubs[stmt.register] = register
        child_reserved = set(reserved)
        child_reserved.add(register)
        return Allocation(
            span=stmt.span,
            register=register,
            count=stmt.count,
            scheme=stmt.scheme,
            body=substitute_statement(
                stmt.body, child_qsubs, psubs, child_rensubs, child_reserved, rename_tag
            ),
        )
    if isinstance(stmt, Reset):
        return Reset(
            span=stmt.span, target=_subst_qref(stmt.target, qsubs, rensubs), keyword=stmt.keyword
        )
    if isinstance(stmt, PrimitiveOp):
        op_name = _subst_pauli_expr(stmt.name, psubs)
        return PrimitiveOp(
            span=stmt.span,
            name=op_name if op_name is not None else stmt.name,
            targets=[_subst_qref(ref, qsubs, rensubs) for ref in stmt.targets],
        )
    if isinstance(stmt, MapOp):
        return MapOp(
            span=stmt.span,
            name=stmt.name,
            targets=[_subst_qref(ref, qsubs, rensubs) for ref in stmt.targets],
        )
    if isinstance(stmt, XorConst):
        return XorConst(
            span=stmt.span,
            target=_subst_qref(stmt.target, qsubs, rensubs),
            width=stmt.width,
            value=stmt.value,
        )
    if isinstance(stmt, Parallel):
        return Parallel(
            span=stmt.span,
            left=substitute_statement(stmt.left, qsubs, psubs, rensubs, reserved, rename_tag),
            right=substitute_statement(stmt.right, qsubs, psubs, rensubs, reserved, rename_tag),
        )
    if isinstance(stmt, Conditional):
        measurement_name = _subst_pauli_expr(stmt.measurement.name, psubs)
        measurement = Measurement(
            span=stmt.measurement.span,
            name=measurement_name if measurement_name is not None else stmt.measurement.name,
            was_string=stmt.measurement.was_string,
        )
        return Conditional(
            span=stmt.span,
            measurement=measurement,
            targets=[_subst_qref(ref, qsubs, rensubs) for ref in stmt.targets],
            during=substitute_statement(stmt.during, qsubs, psubs, rensubs, reserved, rename_tag),
            then_branch=substitute_statement(
                stmt.then_branch, qsubs, psubs, rensubs, reserved, rename_tag
            ),
        )
    if isinstance(stmt, RepeatUntil):
        return RepeatUntil(
            span=stmt.span,
            body=substitute_statement(stmt.body, qsubs, psubs, rensubs, reserved, rename_tag),
            threshold=stmt.threshold,
        )
    if isinstance(stmt, Call):
        return Call(
            span=stmt.span,
            name=stmt.name,
            args=[_subst_call_arg(arg, qsubs, psubs, rensubs) for arg in stmt.args],
        )
    return stmt


def _subst_qref(
    ref: QubitRef, qsubs: dict[str, QubitRef], rensubs: dict[str, str] | None = None
) -> QubitRef:
    rensubs = rensubs or {}
    actual = qsubs.get(ref.name)
    if actual is not None:
        if ref.block is None and ref.logical is None:
            return QubitRef(actual.name, actual.block, actual.logical, ref.span)
        if actual.logical is None and ref.block is not None and ref.logical is None:
            return QubitRef(actual.name, actual.block, ref.block, ref.span)
        return ref
    if ref.name in rensubs:
        return QubitRef(rensubs[ref.name], ref.block, ref.logical, ref.span)
    return ref


def _subst_call_arg(
    arg: CallArg,
    qsubs: dict[str, QubitRef],
    psubs: dict[str, str],
    rensubs: dict[str, str] | None = None,
) -> CallArg:
    rensubs = rensubs or {}
    if isinstance(arg, QubitRef):
        return _subst_qref(arg, qsubs, rensubs)
    if isinstance(arg, IdentifierArg):
        if arg.name in qsubs:
            return qsubs[arg.name]
        if arg.name in rensubs:
            return IdentifierArg(rensubs[arg.name], arg.span)
        pauli = _subst_pauli_expr(arg.name, psubs)
        if pauli is not None:
            return PauliLiteral(pauli, arg.span)
        return arg
    if isinstance(arg, PauliLiteral):
        pauli = _subst_pauli_expr(arg.value, psubs)
        return PauliLiteral(pauli if pauli is not None else arg.value, arg.span)
    return arg


def _fresh_register_name(base: str, reserved: set[str], tag: str) -> str:
    stem = f"{base}__{tag}"
    candidate = stem
    index = 1
    while candidate in reserved:
        index += 1
        candidate = f"{stem}_{index}"
    return candidate


def _call_rename_tag(call: Call) -> str:
    if call.span is None:
        return "call"
    return f"call{call.span.line}_{call.span.column}"


def _collect_formal_usage(decl: SubcircuitDecl) -> FormalUsage:
    params = set(decl.params)
    pauli: set[str] = set()
    qubit: set[str] = set()
    for ref in _iter_qrefs(decl.body or Skip()):
        if ref.name in params:
            qubit.add(ref.name)
    for op in _iter_primitive_ops(decl.body or Skip()):
        if op.name in params:
            pauli.add(op.name)
        pauli |= _pauli_expr_formals(op.name, params)
    for meas in _iter_measurements(decl.body or Skip()):
        if meas.name in params:
            pauli.add(meas.name)
        pauli |= _pauli_expr_formals(meas.name, params)
    return FormalUsage(pauli=pauli, qubit=qubit)


def pauli_operation_as_parallel(stmt: PrimitiveOp) -> Statement | None:
    if not _is_pauli(stmt.name) or len(stmt.name) <= 1 or len(stmt.name) != len(stmt.targets):
        return None
    components = [
        PrimitiveOp(span=stmt.span, name=component, targets=[target])
        for component, target in zip(stmt.name, stmt.targets, strict=True)
    ]
    return _as_parallel_tree(components, stmt.span)


def _as_parallel_tree(statements: list[Statement], span: SourceSpan | None) -> Statement:
    if len(statements) == 1:
        return statements[0]
    return Parallel(
        span=span,
        left=statements[0],
        right=_as_parallel_tree(statements[1:], span),
    )


def _expected_formal_kind(
    decl: SubcircuitDecl | None, usage: FormalUsage, formal: str
) -> CallExpected | None:
    if decl is not None and formal in decl.param_types:
        return decl.param_types[formal]
    return usage.expected_for(formal)


def call_arg_formal_pairs(
    decl: SubcircuitDecl, args: list[CallArg]
) -> list[tuple[str, CallArg]] | None:
    if not _uses_type_directed_call_binding(decl):
        return list(zip(decl.params, args, strict=True))

    unused = list(range(len(args)))
    pairs: list[tuple[str, CallArg]] = []
    for formal in decl.params:
        expected = decl.param_types[formal]
        match_index = next(
            (index for index in unused if _arg_syntax_matches_kind(args[index], expected)),
            None,
        )
        if match_index is None:
            return None
        unused.remove(match_index)
        pairs.append((formal, args[match_index]))
    return pairs


def _uses_type_directed_call_binding(decl: SubcircuitDecl) -> bool:
    return bool(decl.params) and all(formal in decl.param_types for formal in decl.params)


def _arg_syntax_matches_kind(arg: CallArg, expected: CallExpected) -> bool:
    if expected == "pauli":
        if isinstance(arg, PauliLiteral):
            return _is_pauli(arg.value)
        if isinstance(arg, IdentifierArg):
            return _is_pauli(arg.name)
        return False
    if expected in {"codeblock", "qubit"}:
        return not isinstance(arg, PauliLiteral) and not (
            isinstance(arg, IdentifierArg) and _is_pauli(arg.name)
        )
    return True


def _subst_pauli_expr(value: str, psubs: dict[str, str]) -> str | None:
    parsed = _parse_pauli_expr(value, set(psubs), psubs)
    return parsed[0] if parsed else None


def _pauli_expr_formals(value: str, formals: set[str]) -> set[str]:
    parsed = _parse_pauli_expr(value, formals, {formal: formal for formal in formals})
    return parsed[1] if parsed else set()


def _parse_pauli_expr(
    value: str, formals: set[str], replacements: dict[str, str]
) -> tuple[str, set[str]] | None:
    if not value:
        return None
    ordered_formals = sorted(formals, key=len, reverse=True)
    memo: dict[int, tuple[str, set[str]] | None] = {}

    def parse_at(index: int) -> tuple[str, set[str]] | None:
        if index == len(value):
            return "", set()
        if index in memo:
            cached = memo[index]
            return (cached[0], set(cached[1])) if cached else None

        for formal in ordered_formals:
            if formal and value.startswith(formal, index):
                rest = parse_at(index + len(formal))
                if rest is not None:
                    expanded, used = rest
                    result = replacements[formal] + expanded, used | {formal}
                    memo[index] = result
                    return result

        ch = value[index]
        if ch in "IXYZ":
            rest = parse_at(index + 1)
            if rest is not None:
                expanded, used = rest
                result = ch + expanded, used
                memo[index] = result
                return result

        memo[index] = None
        return None

    return parse_at(0)


def _iter_calls(stmt: Statement):
    if isinstance(stmt, Call):
        yield stmt
    for child in _children(stmt):
        yield from _iter_calls(child)


def _call_argument_cache_key(arg: CallArg, env: CheckEnv) -> tuple[object, ...]:
    if isinstance(arg, QubitRef):
        binding = env.registers.get(arg.name)
        return (
            "qubit",
            arg.name,
            arg.block,
            arg.logical,
            binding.scheme.name if binding else None,
            binding.count if binding else None,
        )
    if isinstance(arg, PauliLiteral):
        return ("pauli", arg.value)
    if isinstance(arg, IdentifierArg):
        binding = env.registers.get(arg.name)
        return (
            "identifier",
            arg.name,
            binding.scheme.name if binding else None,
            binding.count if binding else None,
        )
    return ("unknown", repr(arg))


def _iter_qrefs(stmt: Statement):
    if isinstance(stmt, Reset):
        yield stmt.target
    elif isinstance(stmt, PrimitiveOp):
        yield from stmt.targets
    elif isinstance(stmt, MapOp):
        yield from stmt.targets
    elif isinstance(stmt, XorConst):
        yield stmt.target
    elif isinstance(stmt, Conditional):
        yield from stmt.targets
    elif isinstance(stmt, Call):
        for arg in stmt.args:
            if isinstance(arg, QubitRef):
                yield arg
    for child in _children(stmt):
        yield from _iter_qrefs(child)


def _free_register_refs(stmt: Statement, bound: set[str], params: set[str]):
    """Yield (name, node) for register references not bound by a parameter or allocation."""
    if isinstance(stmt, Allocation):
        yield from _free_register_refs(stmt.body, bound | {stmt.register}, params)
        return
    refs: list[QubitRef] = []
    if isinstance(stmt, Reset):
        refs = [stmt.target]
    elif isinstance(stmt, (PrimitiveOp, MapOp)):
        refs = list(stmt.targets)
    elif isinstance(stmt, XorConst):
        refs = [stmt.target]
    elif isinstance(stmt, Conditional):
        refs = list(stmt.targets)
    elif isinstance(stmt, Call):
        for arg in stmt.args:
            if isinstance(arg, QubitRef):
                refs.append(arg)
            elif isinstance(arg, IdentifierArg):
                # Pauli literals and Pauli expressions over formals are not registers.
                if _is_pauli(arg.name) or _pauli_expr_formals(arg.name, params):
                    continue
                if arg.name not in bound:
                    yield arg.name, arg
    for ref in refs:
        if ref.name not in bound:
            yield ref.name, ref
    for child in _children(stmt):
        yield from _free_register_refs(child, bound, params)


def _iter_measurements(stmt: Statement):
    if isinstance(stmt, Conditional):
        yield stmt.measurement
    for child in _children(stmt):
        yield from _iter_measurements(child)


def _iter_primitive_ops(stmt: Statement):
    if isinstance(stmt, PrimitiveOp):
        yield stmt
    for child in _children(stmt):
        yield from _iter_primitive_ops(child)


def _children(stmt: Statement) -> list[Statement]:
    if isinstance(stmt, Sequence):
        return list(stmt.statements)
    if isinstance(stmt, Allocation):
        return [stmt.body]
    if isinstance(stmt, Parallel):
        return [stmt.left, stmt.right]
    if isinstance(stmt, Conditional):
        return [stmt.during, stmt.then_branch]
    if isinstance(stmt, RepeatUntil):
        return [stmt.body]
    return []


def _is_pauli(value: str) -> bool:
    return bool(value) and all(ch in "IXYZ" for ch in value)


def _fmt_block(block: BlockId) -> str:
    return f"{block[0]}[{block[1]}]"
