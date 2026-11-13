"""Big-step resource and error-context semantics for FTQP statements."""

from __future__ import annotations

from collections.abc import Iterable
import math
from dataclasses import dataclass, field

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
from ftqp_prototype.resource.models import QecScheme, RuntimeOperation
from ftqp_prototype.typechecker.checker import (
    CheckResult,
    RegisterBinding,
    _call_rename_tag,
    call_arg_formal_pairs,
    pauli_operation_as_parallel,
    substitute_statement,
)

BlockId = tuple[str, int]


class ResourceEstimationError(Exception):
    """Raised when the resource semantics cannot produce a finite estimate."""


@dataclass
class ErrorState:
    scheme: QecScheme
    epsilon: float = 0.0


@dataclass
class ErrorContext:
    blocks: dict[BlockId, ErrorState] = field(default_factory=dict)

    def copy(self) -> "ErrorContext":
        return ErrorContext(
            {block: ErrorState(state.scheme, state.epsilon) for block, state in self.blocks.items()}
        )

    def restrict(self, domain: set[BlockId]) -> "ErrorContext":
        return ErrorContext({block: self.blocks[block] for block in domain if block in self.blocks})

    def merge(self, other: "ErrorContext") -> "ErrorContext":
        merged = self.copy()
        merged.blocks.update(other.copy().blocks)
        return merged

    def format(self) -> str:
        if not self.blocks:
            return "(empty)"
        parts = []
        for (reg, idx), state in sorted(self.blocks.items()):
            parts.append(f"{reg}[{idx}]: eps={state.epsilon:.6g}, scheme={state.scheme.name}")
        return "; ".join(parts)


@dataclass
class TraceEntry:
    label: str
    time: float
    space: int
    context: ErrorContext
    time_cycles: float | None = None


@dataclass
class ResourceEstimate:
    context: ErrorContext
    time: float
    space: int
    trace: list[TraceEntry] = field(default_factory=list)
    time_cycles: float | None = None
    time_unit: float | None = None
    time_unit_scheme: str | None = None
    mixed_time_units: bool = False


@dataclass
class EstimateEnv:
    registers: dict[str, RegisterBinding] = field(default_factory=dict)

    def copy(self) -> "EstimateEnv":
        return EstimateEnv(registers=dict(self.registers))


class ResourceEstimator:
    def __init__(self, symbols: CheckResult) -> None:
        self.schemes = symbols.schemes
        self.subcircuits = symbols.subcircuits
        self.call_profile_cache: dict[
            tuple[object, ...], tuple[float, int, float]
        ] = {}
        self.static_body_cache: dict[str, bool] = {}

    def estimate_program(self, program: Program) -> ResourceEstimate:
        if program.body is None:
            return ResourceEstimate(ErrorContext(), 0.0, 0, [], time_cycles=0.0, time_unit=0.0)
        result = self.estimate(program.body, ErrorContext(), EstimateEnv())
        return self._annotate_se_cycles(result)

    def _annotate_se_cycles(self, result: ResourceEstimate) -> ResourceEstimate:
        """Attach SE-cycle counts to public time results without changing time."""

        used_schemes: list[QecScheme] = []
        seen: set[str] = set()
        for entry in result.trace:
            for state in entry.context.blocks.values():
                if state.scheme.name not in seen:
                    used_schemes.append(state.scheme)
                    seen.add(state.scheme.name)

        positive_schemes = [scheme for scheme in used_schemes if scheme.t_se_value() > 0]
        if not positive_schemes:
            if result.time == 0 and all(entry.time == 0 for entry in result.trace):
                trace = [
                    TraceEntry(entry.label, entry.time, entry.space, entry.context, time_cycles=0.0)
                    for entry in result.trace
                ]
                return ResourceEstimate(
                    result.context,
                    result.time,
                    result.space,
                    trace,
                    time_cycles=0.0,
                    time_unit=0.0,
                )
            return result

        reference = positive_schemes[0]
        time_unit = reference.t_se_value()
        mixed = any(
            not math.isclose(scheme.t_se_value(), time_unit)
            for scheme in positive_schemes[1:]
        )
        trace = [
            TraceEntry(entry.label, entry.time, entry.space, entry.context, entry.time / time_unit)
            for entry in result.trace
        ]
        return ResourceEstimate(
            result.context,
            result.time,
            result.space,
            trace,
            time_cycles=result.time / time_unit,
            time_unit=time_unit,
            time_unit_scheme=reference.name,
            mixed_time_units=mixed,
        )

    def estimate(
        self, stmt: Statement, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        if isinstance(stmt, Skip):
            return self._result(context.copy(), 0.0, 0, "skip")
        if isinstance(stmt, Sequence):
            return self._estimate_sequence(stmt, context, env)
        if isinstance(stmt, Allocation):
            return self._estimate_allocation(stmt, context, env)
        if isinstance(stmt, Reset):
            return self._estimate_reset(stmt, context, env)
        if isinstance(stmt, PrimitiveOp):
            expanded = pauli_operation_as_parallel(stmt)
            if expanded is not None:
                return self.estimate(expanded, context, env)
            return self._estimate_primitive(stmt, context, env)
        if isinstance(stmt, MapOp):
            return self._estimate_map_op(stmt, context, env)
        if isinstance(stmt, XorConst):
            return self._estimate_xor_const(stmt, context, env)
        if isinstance(stmt, Parallel):
            return self._estimate_parallel(stmt, context, env)
        if isinstance(stmt, Conditional):
            return self._estimate_conditional(stmt, context, env)
        if isinstance(stmt, RepeatUntil):
            return self._estimate_repeat_until(stmt, context, env)
        if isinstance(stmt, Call):
            decl = self.subcircuits[stmt.name]
            cache_key = (
                stmt.name,
                tuple(
                    sorted(
                        (block, state.scheme.name)
                        for block, state in context.blocks.items()
                    )
                ),
                tuple(
                    arg.value
                    for arg in stmt.args
                    if isinstance(arg, PauliLiteral)
                ),
            )
            cached = (
                self.call_profile_cache.get(cache_key)
                if len(context.blocks) == 1
                else None
            )
            if cached is not None:
                time, space, added_failure = cached
                block = next(iter(context.blocks))
                epsilon = _oplus(
                    context.blocks[block].epsilon,
                    added_failure,
                )
                updated = self.step(context, {block}, epsilon, time)
                return self._result(
                    updated,
                    time,
                    space,
                    f"call {stmt.name} (cached gate-level profile)",
                )
            body = self._instantiate_call(stmt, decl, env)
            static_body = _is_static_single_block_body(
                body,
                self.subcircuits,
                self.static_body_cache,
            )
            cacheable = (
                len(context.blocks) == 1
                and static_body
            )
            initial_epsilon = (
                context.blocks[next(iter(context.blocks))].epsilon
                if cacheable
                else 0.0
            )
            result = self.estimate(body, context, env.copy())
            result.trace.insert(0, TraceEntry(f"call {stmt.name}", 0.0, 0, context.copy()))
            if cacheable:
                block = next(iter(context.blocks))
                final_epsilon = result.context.blocks[block].epsilon
                if initial_epsilon >= 1.0:
                    added_failure = 1.0
                else:
                    added_failure = _prob(
                        (final_epsilon - initial_epsilon)
                        / (1.0 - initial_epsilon)
                    )
                self.call_profile_cache[cache_key] = (
                    result.time,
                    result.space,
                    added_failure,
                )
            return result
        return self._result(context.copy(), 0.0, 0, f"unsupported {type(stmt).__name__}")

    def _estimate_sequence(
        self, stmt: Sequence, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        total_time = 0.0
        max_space = 0
        trace: list[TraceEntry] = []
        current = context.copy()
        for child in stmt.statements:
            child_result = self.estimate(child, current, env)
            current = child_result.context
            total_time += child_result.time
            max_space = max(max_space, child_result.space)
            trace.extend(child_result.trace)
        return ResourceEstimate(current, total_time, max_space, trace)

    def _estimate_allocation(
        self, stmt: Allocation, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        scheme = self.schemes[stmt.scheme]
        extended = context.copy()
        local = {(stmt.register, i) for i in range(stmt.count)}
        for block in local:
            extended.blocks[block] = ErrorState(scheme, 0.0)

        child_env = env.copy()
        child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
        body = self.estimate(stmt.body, extended, child_env)
        external_domain = set(context.blocks)
        final_context = body.context.restrict(external_domain)
        data_qubits = scheme.n * stmt.count
        trace = [
            TraceEntry(
                f"allocate {stmt.register}[{stmt.count}] with {stmt.scheme}",
                0.0,
                data_qubits,
                extended.copy(),
            ),
            *body.trace,
            TraceEntry(f"release {stmt.register}", body.time, data_qubits + body.space, final_context.copy()),
        ]
        return ResourceEstimate(final_context, body.time, data_qubits + body.space, trace)

    def _estimate_reset(
        self, stmt: Reset, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        block = self._resolve_block(stmt.target, env)
        scheme = context.blocks[block].scheme
        n_se = max(0.0, scheme.eval(scheme.prep_n_se))
        p = _prob(n_se * scheme.p_l_value())
        time = max(0.0, scheme.eval(scheme.prep_time))
        prep_ancilla = max(0, int(math.ceil(scheme.eval(scheme.prep_ancilla))))
        space = prep_ancilla + self.anc_se(context)
        updated = self.step(context, {block}, p, time)
        return self._result(updated, time, space, f"{stmt.keyword}({_fmt_block(block)})")

    def _estimate_primitive(
        self, stmt: PrimitiveOp, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.name)
        assert op is not None
        return self._apply_operation(
            context,
            block,
            scheme,
            op,
            f"{op.kind} {stmt.name}[{', '.join(ref.display() for ref in stmt.targets)}]",
        )

    def _estimate_map_op(
        self, stmt: MapOp, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.name)
        assert op is not None
        count = len(stmt.targets)
        per_gate_time = max(0.0, scheme.eval(op.time))
        if op.kind == "logical":
            per_gate_failure = _prob(
                max(0.0, scheme.eval(op.n_se or "0")) * scheme.p_l_value()
            )
            space = (
                max(0, int(math.ceil(scheme.eval(op.ancilla))))
                + self.anc_se(context)
            )
        else:
            per_gate_failure = _prob(scheme.eval(op.p_error or "0"))
            space = (
                max(0, int(math.ceil(scheme.eval(op.ancilla))))
                + self.anc_se(context, set(context.blocks) - {block})
            )
        combined_failure = _prob(1.0 - (1.0 - per_gate_failure) ** count)
        total_time = count * per_gate_time
        epsilon = _oplus(context.blocks[block].epsilon, combined_failure)
        updated = self.step(context, {block}, epsilon, total_time)
        return self._result(
            updated,
            total_time,
            space,
            f"map {stmt.name} sequentially over {count} targets",
        )

    def _estimate_xor_const(
        self, stmt: XorConst, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        count = stmt.value.bit_count()
        if count == 0:
            return self._result(context.copy(), 0.0, 0, "xor zero mask")
        start = self._resolve_logical(stmt.target, env)
        first = QubitRef(start[0], block=start[1], logical=start[2], span=stmt.span)
        synthetic = MapOp(
            span=stmt.span,
            name="X",
            targets=[first] * count,
        )
        return self._estimate_map_op(synthetic, context, env)

    def _apply_operation(
        self,
        context: ErrorContext,
        active_block: BlockId,
        scheme: QecScheme,
        op: RuntimeOperation,
        label: str,
    ) -> ResourceEstimate:
        time = max(0.0, scheme.eval(op.time))
        op_ancilla = max(0, int(math.ceil(scheme.eval(op.ancilla))))
        if op.kind == "logical":
            n_se = max(0.0, scheme.eval(op.n_se or "0"))
            p = _prob(n_se * scheme.p_l_value())
            space = op_ancilla + self.anc_se(context)
        else:
            p = _prob(scheme.eval(op.p_error or "0"))
            space = op_ancilla + self.anc_se(context, set(context.blocks) - {active_block})
        epsilon = _oplus(context.blocks[active_block].epsilon, p)
        updated = self.step(context, {active_block}, epsilon, time)
        return self._result(updated, time, space, label)

    def _estimate_parallel(
        self, stmt: Parallel, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        left_blocks = self.footprint(stmt.left, env)
        right_blocks = set(context.blocks) - left_blocks
        left_ctx = context.restrict(left_blocks)
        right_ctx = context.restrict(right_blocks)
        left = self.estimate(stmt.left, left_ctx, env.copy())
        right = self.estimate(stmt.right, right_ctx, env.copy())
        if left.time <= right.time:
            left_final = self.wait(left.context, right.time - left.time)
            final_context = left_final.merge(right.context)
            space = max(left.space, self.anc_se(context, left_blocks)) + right.space
        else:
            right_final = self.wait(right.context, left.time - right.time)
            final_context = left.context.merge(right_final)
            space = left.space + max(right.space, self.anc_se(context, right_blocks))
        time = max(left.time, right.time)
        trace = [
            TraceEntry("parallel begin", 0.0, space, context.copy()),
            *left.trace,
            *right.trace,
            TraceEntry("parallel synchronize", time, space, final_context.copy()),
        ]
        return ResourceEstimate(final_context, time, space, trace)

    def _estimate_conditional(
        self, stmt: Conditional, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        if _is_pauli(stmt.measurement.name):
            return self._estimate_joint_pauli_conditional(stmt, context, env)
        return self._estimate_primitive_measurement_conditional(stmt, context, env)

    def _estimate_primitive_measurement_conditional(
        self, stmt: Conditional, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.measurement.name)
        assert op is not None
        measurement = self._apply_operation(
            context, block, scheme, op, f"measure {stmt.measurement.display()} on {_fmt_block(block)}"
        )
        measurement_epsilon = measurement.context.blocks[block].epsilon
        during = self.estimate(stmt.during, measurement.context, env.copy())
        if op.kind == "logical":
            wait_time = max(0.0, scheme.t_dec_value() - during.time)
            then_context = self.wait(during.context, wait_time)
            then = self.estimate(stmt.then_branch, then_context, env.copy())
            time = measurement.time + max(during.time, scheme.t_dec_value()) + then.time
        else:
            then = self.estimate(stmt.then_branch, during.context, env.copy())
            time = measurement.time + during.time + then.time
        propagated = self.upd(
            then.context,
            self.footprint(stmt.then_branch, env) - {block},
            measurement_epsilon,
        )
        space = max(measurement.space, during.space, then.space)
        trace = [
            *measurement.trace,
            *during.trace,
            *then.trace,
            TraceEntry(
                f"conditional {stmt.measurement.display()} complete",
                time,
                space,
                propagated.copy(),
            ),
        ]
        return ResourceEstimate(propagated, time, space, trace)

    def _estimate_joint_pauli_conditional(
        self, stmt: Conditional, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        logicals = [self._resolve_logical(ref, env) for ref in stmt.targets]
        participating = _distinct_blocks(logicals)
        target_schemes = [context.blocks[item[:2]].scheme for item in logicals]
        a = len(stmt.targets)
        d_joint = min(s.d for s in target_schemes)
        se_ancilla = self.anc_se(context)
        aux_graph = sum(s.n_aux_value() for s in target_schemes)
        adapter = max(0, a - 1) * 3 * d_joint
        space = int(math.ceil(se_ancilla + aux_graph + adapter))
        time_meas = d_joint * max(2.0 * s.t_se_value() for s in target_schemes)
        time_dec = a * max(s.t_dec_value() for s in target_schemes)
        max_pl = max(s.p_l_value() for s in target_schemes)
        unified = _oplus_many(
            [context.blocks[item[:2]].epsilon for item in logicals]
            + [_prob(d_joint * max_pl)]
        )

        after_measure = self.step(context, set(participating), unified, time_meas)

        during = self.estimate(stmt.during, after_measure, env.copy())
        then_context = self.wait(during.context, max(0.0, time_dec - during.time))
        then = self.estimate(stmt.then_branch, then_context, env.copy())
        propagated = self.upd(
            then.context,
            self.footprint(stmt.then_branch, env) - set(participating),
            unified,
        )
        time = time_meas + max(during.time, time_dec) + then.time
        total_space = max(space, during.space, then.space)
        trace = [
            TraceEntry(
                f"joint Pauli {stmt.measurement.display()} on {', '.join(_fmt_block(b) for b in participating)}",
                time_meas,
                space,
                after_measure.copy(),
            ),
            *during.trace,
            *then.trace,
            TraceEntry(
                f"conditional {stmt.measurement.display()} complete",
                time,
                total_space,
                propagated.copy(),
            ),
        ]
        return ResourceEstimate(propagated, time, total_space, trace)

    def _estimate_repeat_until(
        self, stmt: RepeatUntil, context: ErrorContext, env: EstimateEnv
    ) -> ResourceEstimate:
        target_blocks = self.footprint(stmt.body, env)
        sample = self.estimate(stmt.body, context.copy(), env.copy())
        failures = [
            sample.context.blocks[block].epsilon
            for block in sorted(target_blocks)
            if block in sample.context.blocks
        ]
        calibration_failures = {
            block: sample.context.blocks[block].epsilon
            for block in target_blocks
            if block in sample.context.blocks
        }
        per_run_failure = max(failures, default=0.0)
        repetitions = _repeat_count(per_run_failure, stmt.threshold)

        total_time = 0.0
        max_space = 0
        current = context.copy()
        trace: list[TraceEntry] = [
            TraceEntry(
                (
                    f"repeat until {stmt.threshold:g}: "
                    f"estimated {repetitions} repetition(s), per-run failure {per_run_failure:.6g}"
                ),
                0.0,
                0,
                current.copy(),
            )
        ]
        for index in range(repetitions):
            trace.append(
                TraceEntry(
                    f"repeat iteration {index + 1}/{repetitions}",
                    0.0,
                    0,
                    current.copy(),
                )
            )
            result = self.estimate(stmt.body, current, env.copy())
            current = result.context
            total_time += result.time
            max_space = max(max_space, result.space)
            trace.extend(result.trace)

        final_context = current.copy()
        for block, calibration_failure in calibration_failures.items():
            state = final_context.blocks.get(block)
            if state is not None:
                state.epsilon = _prob(calibration_failure**repetitions)
        trace.append(
            TraceEntry(
                f"repeat until {stmt.threshold:g} complete",
                total_time,
                max_space,
                final_context.copy(),
            )
        )
        return ResourceEstimate(final_context, total_time, max_space, trace)

    def step(
        self, context: ErrorContext, active: set[BlockId], active_failure: float, elapsed: float
    ) -> ErrorContext:
        updated = context.copy()
        for block, state in updated.blocks.items():
            if block in active:
                state.epsilon = _prob(active_failure)
            else:
                state.epsilon = _oplus(state.epsilon, self.idle(context, block, elapsed))
        return updated

    def upd(self, context: ErrorContext, blocks: Iterable[BlockId], epsilon: float) -> ErrorContext:
        updated = context.copy()
        for block in blocks:
            state = updated.blocks.get(block)
            if state is not None:
                state.epsilon = _oplus(state.epsilon, epsilon)
        return updated

    def wait(self, context: ErrorContext, elapsed: float) -> ErrorContext:
        if elapsed <= 0:
            return context.copy()
        return self.step(context, set(), 0.0, elapsed)

    def idle(self, context: ErrorContext, block: BlockId, elapsed: float) -> float:
        if elapsed <= 0:
            return 0.0
        scheme = context.blocks[block].scheme
        t_se = scheme.t_se_value()
        if t_se <= 0:
            return 0.0
        return _prob(math.ceil(elapsed / t_se) * scheme.p_l_value())

    def anc_se(self, context: ErrorContext, domain: Iterable[BlockId] | None = None) -> int:
        blocks = context.blocks if domain is None else domain
        return sum(
            context.blocks[block].scheme.se_ancilla_value()
            for block in blocks
            if block in context.blocks
        )

    def footprint(self, stmt: Statement, env: EstimateEnv) -> set[BlockId]:
        if isinstance(stmt, Skip):
            return set()
        if isinstance(stmt, Sequence):
            out: set[BlockId] = set()
            for child in stmt.statements:
                out |= self.footprint(child, env)
            return out
        if isinstance(stmt, Allocation):
            scheme = self.schemes[stmt.scheme]
            child_env = env.copy()
            child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
            local = {(stmt.register, i) for i in range(stmt.count)}
            return self.footprint(stmt.body, child_env) - local
        if isinstance(stmt, Reset):
            return {self._resolve_block(stmt.target, env)}
        if isinstance(stmt, PrimitiveOp):
            return {self._resolve_logical(ref, env)[:2] for ref in stmt.targets}
        if isinstance(stmt, MapOp):
            return {self._resolve_logical(ref, env)[:2] for ref in stmt.targets}
        if isinstance(stmt, XorConst):
            return {self._resolve_logical(stmt.target, env)[:2]}
        if isinstance(stmt, Parallel):
            return self.footprint(stmt.left, env) | self.footprint(stmt.right, env)
        if isinstance(stmt, Conditional):
            return (
                {self._resolve_logical(ref, env)[:2] for ref in stmt.targets}
                | self.footprint(stmt.during, env)
                | self.footprint(stmt.then_branch, env)
            )
        if isinstance(stmt, RepeatUntil):
            return self.footprint(stmt.body, env)
        if isinstance(stmt, Call):
            decl = self.subcircuits.get(stmt.name)
            if decl is None:
                return set()
            return self.footprint(self._instantiate_call(stmt, decl, env), env.copy())
        return set()

    def _instantiate_call(self, call: Call, decl: SubcircuitDecl, env: EstimateEnv) -> Statement:
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
        return substitute_statement(
            decl.body or Skip(),
            qsubs,
            psubs,
            reserved=set(env.registers),
            rename_tag=_call_rename_tag(call),
        )

    def _resolve_block(self, ref: QubitRef, env: EstimateEnv) -> BlockId:
        binding = env.registers[ref.name]
        idx = ref.block if ref.block is not None else 0
        return (binding.name, idx)

    def _resolve_logical(self, ref: QubitRef, env: EstimateEnv) -> tuple[str, int, int]:
        binding = env.registers[ref.name]
        if ref.logical is None and ref.block is not None and binding.count == 1:
            return (binding.name, 0, ref.block)
        block = self._resolve_block(ref, env)
        logical = ref.logical if ref.logical is not None else 0
        return (block[0], block[1], logical)

    def _result(self, context: ErrorContext, time: float, space: int, label: str) -> ResourceEstimate:
        return ResourceEstimate(
            context=context,
            time=time,
            space=space,
            trace=[TraceEntry(label, time, space, context.copy())],
        )


def _oplus(epsilon: float, p: float) -> float:
    return _prob(1.0 - (1.0 - _prob(epsilon)) * (1.0 - _prob(p)))


def _oplus_many(values: Iterable[float]) -> float:
    result = 0.0
    for value in values:
        result = _oplus(result, value)
    return result


def _distinct_blocks(logicals: Iterable[tuple[str, int, int]]) -> list[BlockId]:
    blocks: list[BlockId] = []
    seen: set[BlockId] = set()
    for reg, block, _ in logicals:
        item = (reg, block)
        if item not in seen:
            seen.add(item)
            blocks.append(item)
    return blocks


def _is_static_single_block_body(
    stmt: Statement,
    subcircuits: dict[str, SubcircuitDecl],
    cache: dict[str, bool],
) -> bool:
    """Whether a body is a deterministic sequence of gates and static calls."""

    if isinstance(stmt, (Skip, PrimitiveOp, MapOp, XorConst)):
        return True
    if isinstance(stmt, Sequence):
        return all(
            _is_static_single_block_body(child, subcircuits, cache)
            for child in stmt.statements
        )
    if isinstance(stmt, Call):
        cached = cache.get(stmt.name)
        if cached is not None:
            return cached
        decl = subcircuits.get(stmt.name)
        if decl is None:
            return False
        cache[stmt.name] = False
        result = _is_static_single_block_body(
            decl.body or Skip(),
            subcircuits,
            cache,
        )
        cache[stmt.name] = result
        return result
    return False


def _prob(value: float) -> float:
    return min(1.0, max(0.0, value))


def _repeat_count(per_run_failure: float, threshold: float) -> int:
    if threshold >= 1.0 or per_run_failure <= 0.0:
        return 1
    if per_run_failure >= 1.0:
        raise ResourceEstimationError(
            "repeat-until has per-run failure probability 1, so no finite repetition count "
            f"can reach threshold {threshold:g}"
        )
    return max(1, int(math.ceil(math.log(threshold) / math.log(per_run_failure))))


def _is_pauli(value: str) -> bool:
    return bool(value) and all(ch in "IXYZ" for ch in value)


def _fmt_block(block: BlockId) -> str:
    return f"{block[0]}[{block[1]}]"
