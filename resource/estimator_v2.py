"""Source-aware resource and error semantics (estimator v2).

Implements arXiv-2608.04573 §5.1–5.4: every codeblock carries a set of
independent fault sources instead of a scalar error rate, so a fault shared by
several codeblocks (e.g. through a joint measurement) is counted once.

Time and space follow exactly the same rules as v1 (`estimator.py`); only the
error analysis differs. Each `_rule_*` method names the paper rule it
implements. Design notes: ESTIMATOR_V2_DESIGN.md.

Two representations of a block's source set are supported:

* ``compress=True`` (default): sources held by exactly one block are merged into
  one number (``private_logq``). This is exact, because sources that always sit
  in the same blocks act as a single source. A block's private part gets a
  source id only when it is shared (measurement snapshot, Ctl envelope).
* ``compress=False``: every fault event gets its own id. Slow on long
  programs; kept as the reference implementation for tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

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
    RepeatUntil,
    Reset,
    Sequence,
    Skip,
    Statement,
    SubcircuitDecl,
    XorConst,
)
from ftqp_prototype.resource import distill
from ftqp_prototype.resource.estimator import (
    BlockId,
    ResourceEstimate,
    ResourceEstimationError,
    ResourceEstimator,
    TraceEntry,
    _distinct_blocks,
    _fmt_block,
    _is_pauli,
    _is_static_single_block_body,
    _prob,
    _repeat_count,
)
from ftqp_prototype.resource.models import QecScheme, RuntimeOperation, joint_measurement_risk
from ftqp_prototype.resource.sources import (
    SourceTable,
    log_survival,
    risk_from_log_survival,
)
from ftqp_prototype.typechecker.checker import (
    CheckResult,
    RegisterBinding,
    _call_rename_tag,
    call_arg_formal_pairs,
    pauli_operation_as_parallel,
    substitute_statement,
)

# Reserved register name for ancillas: never reported as a program output.
ANCILLA_REGISTER = "anc"


# ---------------------------------------------------------------------------
# Error contexts
# ---------------------------------------------------------------------------


class BlockState(NamedTuple):
    """Entry Q :_D S of an error context.

    D is ``shared`` (source ids) plus, when compressing, one merged source of
    log-survival ``private_logq`` that no other block holds.
    """

    scheme: QecScheme
    shared: frozenset[int] = frozenset()
    shared_logq: float = 0.0
    private_logq: float = 0.0
    idle_time: float = 0.0  # time since the end of this block's last operation

    @property
    def epsilon(self) -> float:
        """ε_Q = R(D_Q)."""
        return risk_from_log_survival(self.shared_logq + self.private_logq)

    @property
    def source_count(self) -> int:
        return len(self.shared) + (1 if self.private_logq < 0.0 else 0)


@dataclass
class ErrorContextV2:
    blocks: dict[BlockId, BlockState] = field(default_factory=dict)

    # Block states are immutable, so shallow copies are enough.
    def copy(self) -> "ErrorContextV2":
        return ErrorContextV2(dict(self.blocks))

    def restrict(self, domain: Iterable[BlockId]) -> "ErrorContextV2":
        return ErrorContextV2({b: self.blocks[b] for b in domain if b in self.blocks})

    def merge(self, other: "ErrorContextV2") -> "ErrorContextV2":
        merged = dict(self.blocks)
        merged.update(other.blocks)
        return ErrorContextV2(merged)

    def format(self) -> str:
        if not self.blocks:
            return "(empty)"
        return "; ".join(
            f"{reg}[{idx}]: eps={st.epsilon:.6g}, sources={st.source_count}, scheme={st.scheme.name}"
            for (reg, idx), st in sorted(self.blocks.items())
        )


@dataclass
class EstimateEnvV2:
    registers: dict[str, RegisterBinding] = field(default_factory=dict)
    # True only along Sequence / Allocation / Parallel from the program body,
    # i.e. outside conditionals, calls and repeat loops.
    program_level: bool = True

    def copy(self, *, program_level: bool | None = None) -> "EstimateEnvV2":
        return EstimateEnvV2(
            dict(self.registers),
            self.program_level if program_level is None else program_level,
        )


@dataclass
class ResourceEstimateV2(ResourceEstimate):
    outputs: dict[BlockId, BlockState] = field(default_factory=dict)
    output_risk: float | None = None
    naive_output_risk: float | None = None
    distillations: list[distill.DistillationReport] = field(default_factory=list)
    source_ids: int = 0
    semantics: str = "v2"


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class ResourceEstimatorV2:
    def __init__(
        self,
        symbols: CheckResult,
        *,
        compress: bool = True,
        outputs: Iterable[str] | None = None,
        detect_distillation: bool = True,
    ) -> None:
        self.schemes = symbols.schemes
        self.subcircuits = symbols.subcircuits
        self.compress = compress
        self.requested_outputs = set(outputs) if outputs is not None else None
        self.detect_distillation = detect_distillation
        self.table = SourceTable()
        self.call_profile_cache: dict[tuple[object, ...], tuple[float, int, float, bool]] = {}
        self.static_body_cache: dict[str, bool] = {}
        self.call_footprint_cache: dict[tuple[object, ...], frozenset[BlockId]] = {}
        self.call_resets_cache: dict[tuple[object, ...], frozenset[BlockId]] = {}
        self.output_allocs: set[int] = set()
        self.captured_outputs: dict[BlockId, BlockState] = {}
        self.distillations: list[distill.DistillationReport] = []

    # -- program -----------------------------------------------------------

    def estimate_program(self, program: Program) -> ResourceEstimateV2:
        if program.body is None:
            return ResourceEstimateV2(
                ErrorContextV2(), 0.0, 0, [], time_cycles=0.0, time_unit=0.0
            )
        env = EstimateEnvV2()
        self.output_allocs = self._select_output_allocs(program.body, env)
        result = self.estimate(program.body, ErrorContextV2(), env)
        # SE-cycle annotation does not depend on the error model; reuse v1's.
        annotated = ResourceEstimator._annotate_se_cycles(self, result)  # type: ignore[arg-type]
        outputs = dict(self.captured_outputs)
        output_risk, naive = self._output_risk(outputs)
        return ResourceEstimateV2(
            annotated.context,
            annotated.time,
            annotated.space,
            annotated.trace,
            time_cycles=annotated.time_cycles,
            time_unit=annotated.time_unit,
            time_unit_scheme=annotated.time_unit_scheme,
            mixed_time_units=annotated.mixed_time_units,
            outputs=outputs,
            output_risk=output_risk,
            naive_output_risk=naive,
            distillations=list(self.distillations),
            source_ids=len(self.table),
        )

    def _output_risk(
        self, outputs: dict[BlockId, BlockState]
    ) -> tuple[float | None, float | None]:
        """§5.4: ε_O = R(⋃_{Q∈O} D_Q), next to the naive ⊕_Q ε_Q."""
        if not outputs:
            return None, None
        union: set[int] = set()
        private = 0.0
        naive_logq = 0.0
        for st in outputs.values():
            union |= st.shared
            private += st.private_logq  # private parts are disjoint by construction
            naive_logq += log_survival(st.epsilon)
        logq = self.table.log_survival(union) + private
        return risk_from_log_survival(logq), risk_from_log_survival(naive_logq)

    # -- dispatch ----------------------------------------------------------

    def estimate(
        self, stmt: Statement, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        if isinstance(stmt, Skip):
            return self._result(context, 0.0, 0, "skip")
        if isinstance(stmt, Sequence):
            return self._rule_seq(stmt, context, env)
        if isinstance(stmt, Allocation):
            return self._rule_alloc(stmt, context, env)
        if isinstance(stmt, Reset):
            return self._rule_prep(stmt, context, env)
        if isinstance(stmt, PrimitiveOp):
            expanded = pauli_operation_as_parallel(stmt)
            if expanded is not None:
                return self.estimate(expanded, context, env)
            return self._rule_op(stmt, context, env)
        if isinstance(stmt, MapOp):
            return self._estimate_map_op(stmt, context, env)
        if isinstance(stmt, XorConst):
            return self._estimate_xor_const(stmt, context, env)
        if isinstance(stmt, Parallel):
            return self._rule_par(stmt, context, env)
        if isinstance(stmt, Conditional):
            if _is_pauli(stmt.measurement.name):
                return self._rule_if_joint(stmt, context, env)
            return self._rule_if_primitive(stmt, context, env)
        if isinstance(stmt, RepeatUntil):
            return self._estimate_repeat_until(stmt, context, env)
        if isinstance(stmt, Call):
            return self._rule_call(stmt, context, env)
        return self._result(context, 0.0, 0, f"unsupported {type(stmt).__name__}")

    # -- §5.1 primitives, preparation, allocation ---------------------------

    def _rule_op(
        self, stmt: PrimitiveOp, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Op-L) / (Op-F)."""
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.name)
        assert op is not None
        label = f"{op.kind} {stmt.name}[{', '.join(ref.display() for ref in stmt.targets)}]"
        return self._apply_operation(context, block, scheme, op, label)

    def _apply_operation(
        self,
        context: ErrorContextV2,
        block: BlockId,
        scheme: QecScheme,
        op: RuntimeOperation,
        label: str,
    ) -> ResourceEstimate:
        time, space, risk = self._op_costs(context, block, scheme, op)
        kind = "op" if op.kind == "logical" else "faulty"
        state = self._add_fresh(context.blocks[block], risk, kind, label)
        updated = self.step(context, {block: state}, time, label)
        return self._result(updated, time, space, label)

    def _op_costs(
        self,
        context: ErrorContextV2,
        block: BlockId,
        scheme: QecScheme,
        op: RuntimeOperation,
    ) -> tuple[float, int, float]:
        """T_X, N_X (Eq. costs) and the risk p(s_X) of one primitive."""
        time = max(0.0, scheme.eval(op.time))
        ancilla = max(0, int(math.ceil(scheme.eval(op.ancilla))))
        if op.kind == "logical":
            risk = self._rounds_risk(scheme, scheme.eval(op.n_se or "0"))
            space = ancilla + self.anc_se(context)
        else:
            risk = _prob(scheme.eval(op.p_error or "0"))
            space = ancilla + self.anc_se(context, set(context.blocks) - {block})
        return time, space, risk

    def _rule_prep(
        self, stmt: Reset, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Prep): the block's source set is replaced by {s_prep}."""
        block = self._resolve_block(stmt.target, env)
        old = context.blocks[block]
        scheme = old.scheme
        risk = self._rounds_risk(scheme, scheme.eval(scheme.prep_n_se))
        time = max(0.0, scheme.eval(scheme.prep_time))
        space = max(0, int(math.ceil(scheme.eval(scheme.prep_ancilla)))) + self.anc_se(context)
        label = f"{stmt.keyword}({_fmt_block(block)})"
        fresh = BlockState(scheme, idle_time=old.idle_time)
        state = self._add_fresh(fresh, risk, "prep", label)
        updated = self.step(context, {block: state}, time, label)
        return self._result(updated, time, space, label)

    def _rule_alloc(
        self, stmt: Allocation, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Alloc), plus output capture (§5.4) and distillation detection."""
        scheme = self.schemes[stmt.scheme]
        extended = context.copy()
        local = [(stmt.register, i) for i in range(stmt.count)]
        for block in local:
            extended.blocks[block] = BlockState(scheme)
        child_env = env.copy()
        child_env.registers[stmt.register] = RegisterBinding(stmt.register, stmt.count, scheme)
        body = self.estimate(stmt.body, extended, child_env)
        trace_tail: list[TraceEntry] = []

        if self.detect_distillation and distill.tail_looks_like_checks(stmt):
            analysis = distill.analyze(self, stmt, child_env)
            if analysis is not None:
                body_context = body.context.copy()
                if len(analysis.output_blocks) == 1:
                    (block,) = analysis.output_blocks
                    old = body_context.blocks[block]
                    fresh = BlockState(old.scheme, idle_time=old.idle_time)
                    body_context.blocks[block] = self._add_fresh(
                        fresh, analysis.report.accepted_error, "distilled", analysis.report.label
                    )
                else:
                    for block, ids in self._distilled_sources(analysis).items():
                        old = body_context.blocks[block]
                        fresh = BlockState(old.scheme, idle_time=old.idle_time)
                        body_context.blocks[block] = self._union(fresh, ids)
                body = ResourceEstimate(body_context, body.time, body.space, body.trace)
                self.distillations.append(analysis.report)
                trace_tail.append(
                    TraceEntry(
                        f"distillation accepted: A={analysis.report.acceptance:.6g}, "
                        f"eps_acc={analysis.report.accepted_error:.6g}",
                        body.time,
                        0,
                        body_context.copy(),
                    )
                )

        if env.program_level and id(stmt) in self.output_allocs:
            for block in local:
                self.captured_outputs[block] = body.context.blocks[block]

        final_context = body.context.restrict(context.blocks)
        data_qubits = scheme.n * stmt.count
        trace = [
            TraceEntry(
                f"allocate {stmt.register}[{stmt.count}] with {stmt.scheme}",
                0.0,
                data_qubits,
                extended.copy(),
            ),
            *body.trace,
            *trace_tail,
            TraceEntry(
                f"release {stmt.register}",
                body.time,
                data_qubits + body.space,
                final_context.copy(),
            ),
        ]
        return ResourceEstimate(final_context, body.time, data_qubits + body.space, trace)

    def _estimate_map_op(
        self, stmt: MapOp, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """`map`: count × (Op-L/Op-F) on one block, merged into one source (exact)."""
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.name)
        assert op is not None
        count = len(stmt.targets)
        per_time, space, per_risk = self._op_costs(context, block, scheme, op)
        total_time = count * per_time
        combined = risk_from_log_survival(count * log_survival(per_risk))
        label = f"map {stmt.name} sequentially over {count} targets"
        state = self._add_fresh(context.blocks[block], combined, "op", label)
        updated = self.step(context, {block: state}, total_time, label)
        return self._result(updated, total_time, space, label)

    def _estimate_xor_const(
        self, stmt: XorConst, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        count = stmt.value.bit_count()
        if count == 0:
            return self._result(context, 0.0, 0, "xor zero mask")
        start = self._resolve_logical(stmt.target, env)
        first = QubitRef(start[0], block=start[1], logical=start[2], span=stmt.span)
        return self._estimate_map_op(
            MapOp(span=stmt.span, name="X", targets=[first] * count), context, env
        )

    # -- §5.2 conditionals ---------------------------------------------------

    def _rule_if_primitive(
        self, stmt: Conditional, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(If-F) / (If-L)."""
        block = self._resolve_logical(stmt.targets[0], env)[:2]
        scheme = context.blocks[block].scheme
        op = scheme.find_operation(stmt.measurement.name)
        assert op is not None
        label = f"measure {stmt.measurement.display()} on {_fmt_block(block)}"
        time_m, space_m, risk_m = self._op_costs(context, block, scheme, op)

        snapshot = self._snapshot(context, [block], risk_m, label)
        measured = self.step(context, {block: self._with_shared(context.blocks[block], snapshot)}, time_m, label)
        during = self.estimate(stmt.during, measured, env.copy(program_level=False))
        if op.kind == "logical":
            t_dec = scheme.t_dec_value()
            ready = self.wait(during.context, t_dec - during.time, "wait for decoding")
            elapsed = time_m + max(during.time, t_dec)
        else:
            ready = during.context
            elapsed = time_m + during.time
        then = self._rule_ctl(stmt.then_branch, ready, env, snapshot)
        time = elapsed + then.time
        space = max(space_m, during.space, then.space)
        trace = [
            TraceEntry(label, time_m, space_m, measured.copy()),
            *during.trace,
            *then.trace,
            TraceEntry(
                f"conditional {stmt.measurement.display()} complete",
                time,
                space,
                then.context.copy(),
            ),
        ]
        return ResourceEstimate(then.context, time, space, trace)

    def _rule_if_joint(
        self, stmt: Conditional, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(If-J): joint Pauli measurement; costs as in Eq. (njoint)."""
        logicals = [self._resolve_logical(ref, env) for ref in stmt.targets]
        participating = _distinct_blocks(logicals)
        operand_schemes = [context.blocks[item[:2]].scheme for item in logicals]
        a = len(stmt.targets)
        d_joint = min(s.d for s in operand_schemes)
        space = int(
            math.ceil(
                self.anc_se(context)
                + sum(s.n_aux_value() for s in operand_schemes)
                + max(0, a - 1) * 3 * d_joint
            )
        )
        time_meas = d_joint * max(2.0 * s.t_se_value() for s in operand_schemes)
        time_dec = a * max(s.t_dec_value() for s in operand_schemes)
        risk_m = joint_measurement_risk(operand_schemes)
        label = (
            f"joint Pauli {stmt.measurement.display()} on "
            f"{', '.join(_fmt_block(b) for b in participating)}"
        )

        snapshot = self._snapshot(context, participating, risk_m, label)
        active = {b: self._with_shared(context.blocks[b], snapshot) for b in participating}
        measured = self.step(context, active, time_meas, label)
        during = self.estimate(stmt.during, measured, env.copy(program_level=False))
        ready = self.wait(during.context, time_dec - during.time, "wait for decoding")
        then = self._rule_ctl(stmt.then_branch, ready, env, snapshot)
        time = time_meas + max(during.time, time_dec) + then.time
        total_space = max(space, during.space, then.space)
        trace = [
            TraceEntry(label, time_meas, space, measured.copy()),
            *during.trace,
            *then.trace,
            TraceEntry(
                f"conditional {stmt.measurement.display()} complete",
                time,
                total_space,
                then.context.copy(),
            ),
        ]
        return ResourceEstimate(then.context, time, total_space, trace)

    def _snapshot(
        self,
        context: ErrorContextV2,
        measured: Iterable[BlockId],
        risk: float,
        label: str,
    ) -> frozenset[int]:
        """U_M = ⋃_{Q measured} D_Q ∪ {s_M} (Eq. snapshot), frozen at measurement time."""
        ids: set[int] = set()
        for block in measured:
            ids |= self._materialize(context.blocks[block]).shared
        if risk > 0.0:
            ids.add(self.table.fresh(risk, "meas", label))
        return frozenset(ids)

    def _rule_ctl(
        self,
        stmt: Statement,
        context: ErrorContextV2,
        env: EstimateEnvV2,
        snapshot: frozenset[int],
    ) -> ResourceEstimate:
        """(Ctl): run S under the control of a measurement with snapshot U.

        Q ∈ Rst(S):                D' ∪ U ∪ D^Γ ∪ {s_{Q,t}}
        Q ∈ ActBlock(S) \\ Rst(S):  D' ∪ U
        otherwise:                 D'
        """
        active = self.footprint(stmt, env) & context.blocks.keys()
        resets = self.resets(stmt, env) & active
        if resets and self.compress:
            # Give the pre-S private parts an id now, so the envelope below
            # and any sharing inside S refer to the same source.
            context = context.copy()
            for block in resets:
                context.blocks[block] = self._materialize(context.blocks[block])
        result = self.estimate(stmt, context, env.copy(program_level=False))
        if not active:
            return result
        blocks = dict(result.context.blocks)
        for block in active:
            state = self._union(blocks[block], snapshot)
            if block in resets:
                before = context.blocks[block]
                state = self._union(state, before.shared)
                skip_risk = self._idle_risk(before, result.time)
                state = self._add_fresh(
                    state, skip_risk, "idle-skip", f"skip-path idle {_fmt_block(block)}"
                )
            blocks[block] = state
        return ResourceEstimate(ErrorContextV2(blocks), result.time, result.space, result.trace)

    # -- §5.3 composition and calls -----------------------------------------

    def _rule_seq(
        self, stmt: Sequence, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Seq)."""
        total_time = 0.0
        max_space = 0
        trace: list[TraceEntry] = []
        current = context
        for child in stmt.statements:
            child_result = self.estimate(child, current, env)
            current = child_result.context
            total_time += child_result.time
            max_space = max(max_space, child_result.space)
            trace.extend(child_result.trace)
        return ResourceEstimate(current, total_time, max_space, trace)

    def _rule_par(
        self, stmt: Parallel, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Par) with D1 := ActBlock(S1), D2 := dom Γ \\ D1."""
        left_blocks = self.footprint(stmt.left, env) & context.blocks.keys()
        right_blocks = set(context.blocks) - left_blocks
        left = self.estimate(stmt.left, context.restrict(left_blocks), env.copy())
        right = self.estimate(stmt.right, context.restrict(right_blocks), env.copy())
        if left.time <= right.time:
            final_context = self.wait(left.context, right.time - left.time, "parallel sync").merge(
                right.context
            )
            space = max(left.space, self.anc_se(context, left_blocks)) + right.space
        else:
            final_context = left.context.merge(
                self.wait(right.context, left.time - right.time, "parallel sync")
            )
            space = left.space + max(right.space, self.anc_se(context, right_blocks))
        time = max(left.time, right.time)
        trace = [
            TraceEntry("parallel begin", 0.0, space, context.copy()),
            *left.trace,
            *right.trace,
            TraceEntry("parallel synchronize", time, space, final_context.copy()),
        ]
        return ResourceEstimate(final_context, time, space, trace)

    def _rule_call(
        self, stmt: Call, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """(Call): substitute and estimate the body; every call gets fresh sources.

        Cache (as in v1): for a gate-only body in a single-block context, the
        added sources only sit in that block, so they merge into one fresh
        source per call, and time and space do not depend on the context.
        """
        decl = self.subcircuits[stmt.name]
        single = len(context.blocks) == 1
        cache_key = (
            stmt.name,
            tuple(sorted((block, st.scheme.name) for block, st in context.blocks.items())),
            tuple(arg.value for arg in stmt.args if isinstance(arg, PauliLiteral)),
        )
        cached = self.call_profile_cache.get(cache_key) if single else None
        label = f"call {stmt.name}"
        if cached is not None:
            time, space, added_logq, resets_idle = cached
            block = next(iter(context.blocks))
            state = self._add_fresh_logq(context.blocks[block], added_logq, "call", label)
            if resets_idle:
                state = state._replace(idle_time=0.0)
            updated = ErrorContextV2({block: state})
            return self._result(updated, time, space, f"{label} (cached gate-level profile)")

        body = self._instantiate_call(stmt, decl, env)
        cacheable = single and _is_static_single_block_body(
            body, self.subcircuits, self.static_body_cache
        )
        result = self.estimate(body, context, env.copy(program_level=False))
        result.trace.insert(0, TraceEntry(label, 0.0, 0, context.copy()))
        if cacheable:
            block = next(iter(context.blocks))
            before = context.blocks[block]
            after = result.context.blocks[block]
            added = (after.shared_logq + after.private_logq) - (
                before.shared_logq + before.private_logq
            )
            self.call_profile_cache[cache_key] = (
                result.time,
                result.space,
                min(0.0, added),
                result.time > 0.0,
            )
        return result

    # -- prototype extension (no rule in the paper) -------------------------

    def _estimate_repeat_until(
        self, stmt: RepeatUntil, context: ErrorContextV2, env: EstimateEnvV2
    ) -> ResourceEstimate:
        """`repeat S until ε`: provisional rule carried over from v1.

        Postselection is what §5.5 (distillation) models; until a rule exists,
        the target blocks end with one fresh source of risk calib^reps.
        """
        targets = self.footprint(stmt.body, env)
        loop_env = env.copy(program_level=False)
        sample = self.estimate(stmt.body, context, loop_env)
        calibration = {
            block: sample.context.blocks[block].epsilon
            for block in targets
            if block in sample.context.blocks
        }
        per_run = max(calibration.values(), default=0.0)
        repetitions = _repeat_count(per_run, stmt.threshold)
        total_time = 0.0
        max_space = 0
        current = context
        trace = [
            TraceEntry(
                f"repeat until {stmt.threshold:g} (provisional rule): "
                f"estimated {repetitions} repetition(s), per-run failure {per_run:.6g}",
                0.0,
                0,
                current.copy(),
            )
        ]
        for _ in range(repetitions):
            result = self.estimate(stmt.body, current, loop_env)
            current = result.context
            total_time += result.time
            max_space = max(max_space, result.space)
            trace.extend(result.trace)
        final = current.copy()
        for block, failure in calibration.items():
            old = final.blocks.get(block)
            if old is not None:
                fresh = BlockState(old.scheme, idle_time=old.idle_time)
                final.blocks[block] = self._add_fresh(
                    fresh, _prob(failure**repetitions), "repeat", "repeat-until"
                )
        trace.append(
            TraceEntry(f"repeat until {stmt.threshold:g} complete", total_time, max_space, final.copy())
        )
        return ResourceEstimate(final, total_time, max_space, trace)

    # -- context algebra -------------------------------------------------------

    def step(
        self,
        context: ErrorContextV2,
        active: dict[BlockId, BlockState],
        elapsed: float,
        label: str = "",
    ) -> ErrorContextV2:
        """Step_Γ(K, T) (Eq. step). Active blocks take their new state; the
        others get a fresh idle source for the same interval."""
        blocks: dict[BlockId, BlockState] = {}
        for block, state in context.blocks.items():
            new = active.get(block)
            if new is not None:
                # The idle period of an active block restarts when its
                # operation ends (zero-length operations do not interrupt it).
                blocks[block] = new._replace(idle_time=0.0 if elapsed > 0 else state.idle_time)
            else:
                blocks[block] = self._idle(state, block, elapsed, label)
        return ErrorContextV2(blocks)

    def wait(self, context: ErrorContextV2, elapsed: float, label: str = "wait") -> ErrorContextV2:
        """Wait_Γ(T) := Step_Γ(∅, T), and Γ itself for T ≤ 0."""
        if elapsed <= 0:
            return context
        return self.step(context, {}, elapsed, label)

    def _idle(self, state: BlockState, block: BlockId, elapsed: float, label: str) -> BlockState:
        if elapsed <= 0:
            return state
        risk = self._idle_risk(state, elapsed)
        state = self._add_fresh(state, risk, "idle", f"idle {_fmt_block(block)} during {label}")
        return state._replace(idle_time=state.idle_time + elapsed)

    def _idle_risk(self, state: BlockState, elapsed: float) -> float:
        """Idle_Γ(Q, T) = r_S(n_Q(T)), with rounds counted cumulatively from
        the end of the block's last operation."""
        t_se = state.scheme.t_se_value()
        if elapsed <= 0 or t_se <= 0:
            return 0.0
        rounds = _ceil_rounds((state.idle_time + elapsed) / t_se) - _ceil_rounds(
            state.idle_time / t_se
        )
        return self._rounds_risk(state.scheme, rounds)

    @staticmethod
    def _rounds_risk(scheme: QecScheme, rounds: float) -> float:
        """r_S(n) = 1 - (1 - p_L)^n (Eq. idle)."""
        rounds = max(0.0, rounds)
        p_l = scheme.p_l_value()
        if rounds == 0 or p_l <= 0:
            return 0.0
        return risk_from_log_survival(rounds * log_survival(p_l))

    def _distilled_sources(self, analysis: "distill.DistillationAnalysis") -> dict[BlockId, frozenset[int]]:
        """Sources of the accepted outputs of a protocol with several output blocks.

        The outputs fail together, so they must not get one independent source
        each. Every error pattern e ≠ 0 (probability w_e given acceptance) becomes
        one source, held by exactly the blocks with a wrong output in e. Its
        parameter is p_e = 1 - (1 - ε_acc)^(w_e / ε_acc), so that

        * the risk of all outputs together is ε_acc, exactly;
        * the risk of one block is 1 - (1 - ε_acc)^(m/ε_acc) ≥ m, its marginal
          error m (equal up to O(ε_acc²)).
        """
        report = analysis.report
        held: dict[BlockId, set[int]] = {block: set() for block in analysis.output_blocks}
        total = math.fsum(report.output_patterns.values())
        if total <= 0.0 or report.accepted_error <= 0.0:
            return {block: frozenset() for block in held}
        joint_logq = log_survival(report.accepted_error)
        for pattern, weight in report.output_patterns.items():
            blocks = {
                qubit[:2] for i, qubit in enumerate(analysis.outputs) if pattern >> i & 1
            } & held.keys()
            if not blocks:
                continue
            logq = joint_logq * weight / total
            sid = self.table.fresh(risk_from_log_survival(logq), "distilled", report.label)
            self.table.logq[sid] = logq
            for block in blocks:
                held[block].add(sid)
        return {block: frozenset(ids) for block, ids in held.items()}

    def _add_fresh(self, state: BlockState, risk: float, kind: str, label: str) -> BlockState:
        """D ∪ {s} for a fresh source s with p(s) = risk (omitted when 0)."""
        if risk <= 0.0:
            return state
        return self._add_fresh_logq(state, log_survival(risk), kind, label)

    def _add_fresh_logq(self, state: BlockState, logq: float, kind: str, label: str) -> BlockState:
        if logq >= 0.0:
            return state
        if self.compress:
            return state._replace(private_logq=state.private_logq + logq)
        sid = self.table.fresh(risk_from_log_survival(logq), kind, label)
        self.table.logq[sid] = logq
        return state._replace(shared=state.shared | {sid}, shared_logq=state.shared_logq + logq)

    def _materialize(self, state: BlockState) -> BlockState:
        """Give the merged private sources one id (needed before sharing them)."""
        if state.private_logq >= 0.0:
            return state
        sid = self.table.fresh(risk_from_log_survival(state.private_logq), "merged")
        self.table.logq[sid] = state.private_logq
        return state._replace(
            shared=state.shared | {sid},
            shared_logq=state.shared_logq + state.private_logq,
            private_logq=0.0,
        )

    def _with_shared(self, state: BlockState, ids: frozenset[int]) -> BlockState:
        """The state whose source set is exactly `ids`."""
        return state._replace(shared=ids, shared_logq=self.table.log_survival(ids), private_logq=0.0)

    def _union(self, state: BlockState, ids: frozenset[int]) -> BlockState:
        """D ∪ ids; shared sources already in D are counted once."""
        new = ids - state.shared
        if not new:
            return state
        return state._replace(
            shared=state.shared | new,
            shared_logq=state.shared_logq + self.table.log_survival(new),
        )

    def anc_se(self, context: ErrorContextV2, domain: Iterable[BlockId] | None = None) -> int:
        blocks = context.blocks if domain is None else domain
        return sum(
            context.blocks[block].scheme.se_ancilla_value()
            for block in blocks
            if block in context.blocks
        )

    # -- ActBlock and Rst -----------------------------------------------------

    def footprint(self, stmt: Statement, env: EstimateEnvV2) -> set[BlockId]:
        """ActBlock(S), with calls unfolded: ActBlock(call) := ActBlock(S'[P̄/P̄', q̄/q̄'])."""
        if isinstance(stmt, Skip):
            return set()
        if isinstance(stmt, Sequence):
            out: set[BlockId] = set()
            for child in stmt.statements:
                out |= self.footprint(child, env)
            return out
        if isinstance(stmt, Allocation):
            child_env = self._alloc_env(stmt, env)
            local = {(stmt.register, i) for i in range(stmt.count)}
            return self.footprint(stmt.body, child_env) - local
        if isinstance(stmt, Reset):
            return {self._resolve_block(stmt.target, env)}
        if isinstance(stmt, (PrimitiveOp, MapOp)):
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
            key = self._call_key(stmt, env)
            cached = self.call_footprint_cache.get(key)
            if cached is None:
                body = self._instantiate_call(stmt, self.subcircuits[stmt.name], env)
                cached = frozenset(self.footprint(body, env.copy()))
                self.call_footprint_cache[key] = cached
            return set(cached)
        return set()

    def resets(self, stmt: Statement, env: EstimateEnvV2) -> set[BlockId]:
        """Rst(S): blocks prepared in S (calls unfolded), minus blocks allocated in S."""
        if isinstance(stmt, Reset):
            return {self._resolve_block(stmt.target, env)}
        if isinstance(stmt, Sequence):
            out: set[BlockId] = set()
            for child in stmt.statements:
                out |= self.resets(child, env)
            return out
        if isinstance(stmt, Allocation):
            child_env = self._alloc_env(stmt, env)
            local = {(stmt.register, i) for i in range(stmt.count)}
            return self.resets(stmt.body, child_env) - local
        if isinstance(stmt, Parallel):
            return self.resets(stmt.left, env) | self.resets(stmt.right, env)
        if isinstance(stmt, Conditional):
            return self.resets(stmt.during, env) | self.resets(stmt.then_branch, env)
        if isinstance(stmt, RepeatUntil):
            return self.resets(stmt.body, env)
        if isinstance(stmt, Call):
            key = self._call_key(stmt, env)
            cached = self.call_resets_cache.get(key)
            if cached is None:
                body = self._instantiate_call(stmt, self.subcircuits[stmt.name], env)
                cached = frozenset(self.resets(body, env.copy()))
                self.call_resets_cache[key] = cached
            return set(cached)
        return set()

    def _call_key(self, stmt: Call, env: EstimateEnvV2) -> tuple[object, ...]:
        """Bodies are closed (checker), so a call's footprint only depends on its arguments."""
        parts: list[object] = [stmt.name]
        for arg in stmt.args:
            if isinstance(arg, QubitRef) and arg.name in env.registers:
                parts.append(("q", self._resolve_logical(arg, env)))
            elif isinstance(arg, IdentifierArg) and arg.name in env.registers:
                parts.append(("r", env.registers[arg.name].name, env.registers[arg.name].count))
            elif isinstance(arg, PauliLiteral):
                parts.append(("p", arg.value))
            elif isinstance(arg, IdentifierArg):
                parts.append(("p", arg.name))
            else:
                parts.append(("?", repr(arg)))
        return tuple(parts)

    # -- outputs (§5.4) -------------------------------------------------------

    def _select_output_allocs(self, body: Statement, env: EstimateEnvV2) -> set[int]:
        """Allocations whose registers are program outputs.

        Default, two levels:
        1. A register named `anc` (reserved identifier) is always an ancilla.
        2. Otherwise a program-level register is an output if it is released
           only after all circuit operations have finished (only releases
           follow its release). Registers released mid-program, or allocated
           inside conditionals, calls or loops, are ancillas. A final
           measurement on a data register does not make it an ancilla: the
           error of that readout is part of the output error.
        With `outputs=` given, every program-level allocation of those names.
        """
        selected: set[int] = set()
        found: set[str] = set()

        def walk(stmt: Statement, at_end: bool, walk_env: EstimateEnvV2) -> None:
            if isinstance(stmt, Allocation):
                child_env = self._alloc_env(stmt, walk_env)
                if self.requested_outputs is not None:
                    if stmt.register in self.requested_outputs:
                        selected.add(id(stmt))
                        found.add(stmt.register)
                elif at_end and stmt.register != ANCILLA_REGISTER:
                    selected.add(id(stmt))
                walk(stmt.body, at_end, child_env)
            elif isinstance(stmt, Sequence):
                last = len(stmt.statements) - 1
                for index, child in enumerate(stmt.statements):
                    walk(child, at_end and index == last, walk_env)
            elif isinstance(stmt, Parallel):
                walk(stmt.left, at_end, walk_env)
                walk(stmt.right, at_end, walk_env)

        walk(body, True, env)
        if self.requested_outputs is not None:
            missing = sorted(self.requested_outputs - found)
            if missing:
                raise ResourceEstimationError(
                    "requested output register(s) not allocated at program level: "
                    + ", ".join(missing)
                )
        return selected

    # -- helpers shared with v1 -------------------------------------------------

    def _alloc_env(self, stmt: Allocation, env: EstimateEnvV2) -> EstimateEnvV2:
        child_env = env.copy()
        child_env.registers[stmt.register] = RegisterBinding(
            stmt.register, stmt.count, self.schemes[stmt.scheme]
        )
        return child_env

    def _instantiate_call(self, call: Call, decl: SubcircuitDecl, env: EstimateEnvV2) -> Statement:
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

    def _resolve_block(self, ref: QubitRef, env: EstimateEnvV2) -> BlockId:
        binding = env.registers[ref.name]
        idx = ref.block if ref.block is not None else 0
        return (binding.name, idx)

    def _resolve_logical(self, ref: QubitRef, env: EstimateEnvV2) -> tuple[str, int, int]:
        binding = env.registers[ref.name]
        if ref.logical is None and ref.block is not None and binding.count == 1:
            return (binding.name, 0, ref.block)
        block = self._resolve_block(ref, env)
        logical = ref.logical if ref.logical is not None else 0
        return (block[0], block[1], logical)

    def _result(
        self, context: ErrorContextV2, time: float, space: int, label: str
    ) -> ResourceEstimate:
        return ResourceEstimate(
            context=context,
            time=time,
            space=space,
            trace=[TraceEntry(label, time, space, context.copy())],
        )


def _ceil_rounds(x: float) -> int:
    """ceil with a small relative tolerance, so 390·tSE/tSE does not become 391."""
    return max(0, math.ceil(x - 1e-9 * max(1.0, abs(x))))
