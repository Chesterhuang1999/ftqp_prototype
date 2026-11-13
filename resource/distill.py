"""Error model of rotation-form distillation protocols (arXiv-2608.04573 §5.5).

Pattern (as in examples/distillation/15to1-*.ft): an allocation scope of local
ancillas whose body

* applies rotations P_{j,π/8}, each a joint measurement M(P_j·Z)[q̄, a'] with a
  fresh ancilla a' followed by `if M(faulty)[a']; then P_j[q̄]`, plus
  preparations and logical (Clifford) gates, and
* ends with check measurements `(if M(B)[a_i]; then skip) || …` on the scope's
  own ancillas, right before `release`.

`extract` reads the protocol (rotations, checks, outputs) from such a scope and
`evaluate` runs the recursion of Eq. (distill-rec) for the acceptance
probability A and the accepted-output error ε_acc. Two users:

* `estimator_distill.DistillationEstimator`, the standalone estimator for
  distillation programs;
* `estimator_v2` (plug-in): an accepted output continues with one fresh source
  of risk ε_acc.

Assumption (design doc, Q5): the output observable is in the same basis as the
checks, so P_j flips an output when it anticommutes with that basis there.

Storage errors (optional, on by default; appendix of the paper). Faults on the
outputs and checks themselves enter the recursion as further events between
the rotations, from the protocol's timeline:

* a block stored for n QEC rounds has an error with r = 1 - (1 - p_L)^n, split
  half X and half Z;
* an error E on block Q that anticommutes with Q's basis flips Q's bit;
* an error that anticommutes with rotations on Q is moved to the nearer end of
  the protocol: each rotation it crosses becomes P_{j,-θ}, an extra error
  P_{j,-π/4} that flips (h_j, l_j) with probability 1/2 after twirling; at the
  start it vanishes (the qubits start in the +1 eigenstate of their basis), at
  the end it is harmless on a check and leaves a magic-state output wrong with
  probability 1/2 (|<m|X|m>|^2 = 1/2);
* storage of a rotation's own ancilla before its faulty measurement (risk a)
  adds 3a/4 to q_j (Z gives P_j, X gives P_{j,-π/4}).

On Litinski's 15-to-1 schedule this reproduces his density-matrix output error
within 1e-5 (scripts/distill_storage.py validate). `evaluate(..., storage=False)`
is the plain recursion of §5.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftqp_prototype.resource.models import joint_measurement_risk

from ftqp_prototype.parser.ast import (
    Allocation,
    Call,
    Conditional,
    MapOp,
    Parallel,
    PrimitiveOp,
    RepeatUntil,
    Reset,
    Sequence,
    Skip,
    Statement,
    XorConst,
)
from ftqp_prototype.typechecker.checker import pauli_operation_as_parallel

if TYPE_CHECKING:  # pragma: no cover
    from ftqp_prototype.resource.estimator_v2 import EstimateEnvV2, ResourceEstimatorV2

Logical = tuple[str, int, int]
BlockId = tuple[str, int]


@dataclass
class Rotation:
    """One rotation P_{j,π/8} and its error, Eq. (distill-q)."""

    axis: dict[Logical, str]
    p_f: float  # error rate of the faulty T measurement
    q_m: float  # risk of the joint measurement M(P_j·Z)
    check_flips: int = 0  # l_j as a bit mask over the checks
    output_flips: int = 0  # h_j as a bit mask over the outputs
    time: float = 0.0  # when the rotation acts (end of its joint measurement), from scope start
    q_storage: float = 0.0  # extra flip probability from storage of the rotation ancilla

    @property
    def q_f(self) -> float:
        """X, Y, Z each with p_F/3; two of them leave the Pauli error P_j."""
        return 2.0 * self.p_f / 3.0

    @property
    def q(self) -> float:
        """q_j = q_F (1 - q_m/2) + (1 - q_F) q_m/2."""
        return self.q_f * (1.0 - self.q_m / 2.0) + (1.0 - self.q_f) * self.q_m / 2.0

    @property
    def q_total(self) -> float:
        """q_j with the storage of the rotation ancilla: q_j ⊕ q_storage."""
        return _xor(self.q, self.q_storage)


@dataclass
class Protocol:
    scope: str
    rotations: list[Rotation]
    checks: list[tuple[Logical, str]]  # (qubit, basis)
    outputs: list[Logical]
    output_blocks: list[BlockId] = field(default_factory=list)
    end: float = 0.0  # duration of the scope body (the checks included)
    # (p_L per round, t_SE) of each output and check qubit, for storage errors
    storage_rates: dict[Logical, tuple[float, float]] = field(default_factory=dict)


@dataclass
class DistillationReport:
    label: str
    rotations: int
    checks: list[tuple[Logical, str]]
    outputs: list[Logical]
    acceptance: float
    accepted_error: float
    rotation_error: float  # largest q_j (with ancilla storage when storage is on)
    protocol: Protocol | None = None
    storage: bool = False  # storage errors included in acceptance / accepted_error
    acceptance_no_storage: float | None = None  # plain §5.5 recursion
    accepted_error_no_storage: float | None = None
    # Several outputs: `accepted_error` is the probability that at least one
    # output is wrong. `output_errors[i]` is the marginal error of outputs[i],
    # and `output_patterns[e]` the probability of the error pattern e ≠ 0 (bit
    # i = outputs[i] wrong); all three are conditional on acceptance.
    output_errors: list[float] = field(default_factory=list)
    output_patterns: dict[int, float] = field(default_factory=dict)


@dataclass
class DistillationAnalysis:
    report: DistillationReport
    output_blocks: list[BlockId] = field(default_factory=list)
    outputs: list[Logical] = field(default_factory=list)  # bit order of output_patterns


def tail_looks_like_checks(stmt: Allocation) -> bool:
    """Cheap syntactic pre-filter: does the scope end with checks on its own register?"""
    tail = stmt.body
    while isinstance(tail, Sequence) and tail.statements:
        tail = tail.statements[-1]
    leaves = _parallel_leaves(tail)
    return bool(leaves) and all(
        _is_check_shape(leaf) and leaf.targets[0].name == stmt.register for leaf in leaves
    )


def extract(
    est: "ResourceEstimatorV2", stmt: Allocation, env: "EstimateEnvV2", scope: str = ""
) -> Protocol | None:
    """Read rotations, checks and outputs from a scope; None if it does not match."""
    atoms: list[tuple[Statement, "EstimateEnvV2", float, float]] = []
    local_blocks: set[BlockId] = {(stmt.register, i) for i in range(stmt.count)}
    alloc_time: dict[BlockId, float] = {block: 0.0 for block in local_blocks}
    scope_end = _flatten(est, stmt.body, env, atoms, local_blocks, 0.0, alloc_time)
    if scope_end is None:
        return None

    checks: list[tuple[Logical, str]] = []
    rates: dict[Logical, tuple[float, float]] = {}
    end = len(atoms)
    while end > 0:
        node, node_env, _, _ = atoms[end - 1]
        if not _is_check_shape(node):
            break
        logical = est._resolve_logical(node.targets[0], node_env)
        if logical[:2] not in local_blocks:
            break
        checks.insert(0, (logical, node.measurement.name))
        rates[logical] = _rates(node_env.registers[logical[0]].scheme)
        end -= 1
    if not checks:
        return None

    joint_risk: dict[Logical, float] = {}
    joint_end: dict[Logical, float] = {}
    rotations: list[Rotation] = []
    for node, node_env, start, duration in atoms[:end]:
        if isinstance(node, Reset):
            if est._resolve_block(node.target, node_env) not in local_blocks:
                return None
            continue
        if isinstance(node, PrimitiveOp):
            block = est._resolve_logical(node.targets[0], node_env)[:2]
            op = node_env.registers[block[0]].scheme.find_operation(node.name)
            if op is None or op.kind != "logical":
                return None  # only Clifford/logical gates between rotations
            continue
        if not isinstance(node, Conditional) or not isinstance(node.during, Skip):
            return None
        logicals = [est._resolve_logical(ref, node_env) for ref in node.targets]
        if _is_pauli_name(node.measurement.name):
            # Joint measurement M(P·Z)[q̄, a'] of a rotation gadget; its
            # correction may only touch local ancillas. Risk as in Eq. (njoint).
            then_axis = _pauli_axis(est, node.then_branch, node_env)
            if then_axis is None or any(q[:2] not in local_blocks for q in then_axis):
                return None
            schemes = [node_env.registers[q[0]].scheme for q in logicals]
            risk = joint_measurement_risk(schemes)
            # The rotation acts when the measurement ends, before its correction.
            measured = start + duration - _duration(est, node.then_branch, node_env)
            for logical in logicals:
                joint_risk[logical] = risk
                joint_end[logical] = measured
            continue
        # Faulty single-qubit measurement on the rotation ancilla, then P_j[q̄].
        if len(logicals) != 1:
            return None
        scheme = node_env.registers[logicals[0][0]].scheme
        op = scheme.find_operation(node.measurement.name)
        if op is None or op.kind != "faulty":
            return None
        axis = _pauli_axis(est, node.then_branch, node_env)
        if not axis:
            return None
        p_f = min(1.0, max(0.0, scheme.eval(op.p_error or "0")))
        ancilla = logicals[0]
        p_l, t_se = _rates(scheme)
        stored = start - alloc_time.get(ancilla[:2], start)  # ancilla alive before its faulty measurement
        q_storage = 0.75 * _storage_risk(p_l, stored / t_se if t_se > 0 else 0.0)
        for qubit in axis:
            rates.setdefault(qubit, _rates(node_env.registers[qubit[0]].scheme))
        rotations.append(
            Rotation(
                axis,
                p_f,
                joint_risk.get(ancilla, 0.0),
                time=joint_end.get(ancilla, start),
                q_storage=q_storage,
            )
        )

    outputs = sorted({q for rot in rotations for q in rot.axis if q[:2] not in local_blocks})
    if not rotations or not outputs:
        return None

    out_basis = checks[0][1]
    for rot in rotations:
        rot.check_flips = sum(
            1 << i
            for i, (qubit, basis) in enumerate(checks)
            if _anticommutes(rot.axis.get(qubit, "I"), basis)
        )
        rot.output_flips = sum(
            1 << i
            for i, qubit in enumerate(outputs)
            if _anticommutes(rot.axis.get(qubit, "I"), out_basis)
        )
    return Protocol(
        scope or f"{stmt.register}[{stmt.count}]",
        rotations,
        checks,
        outputs,
        sorted({q[:2] for q in outputs}),
        end=scope_end,
        storage_rates={q: rates[q] for q in [*outputs, *(c for c, _ in checks)] if q in rates},
    )


Pattern = tuple[int, int]  # (e, c): output errors and check flips


def evaluate(protocol: Protocol, *, storage: bool = True) -> tuple[float, float]:
    """Eq. (distill-rec), with storage events when `storage`: return (A, ε_acc).

    μ_j(e, c) = (1-q_j) μ_{j-1}(e, c) + q_j μ_{j-1}(e⊕h_j, c⊕l_j),
    A = Σ_e μ_n(e, 0),  ε_acc = Σ_{e≠0} μ_n(e, 0) / A.
    """
    return _metrics(_final_distribution(protocol, storage=storage))


def output_patterns(protocol: Protocol, *, storage: bool = True) -> dict[int, float]:
    """Pr[output error pattern = e | accepted] for every e ≠ 0 (bit i = outputs[i])."""
    mu = _final_distribution(protocol, storage=storage)
    acceptance = sum(w for (e, c), w in mu.items() if c == 0)
    if acceptance <= 0.0:
        return {}
    return {e: w / acceptance for (e, c), w in sorted(mu.items()) if c == 0 and e != 0 and w > 0.0}


def _final_distribution(protocol: Protocol, *, storage: bool) -> dict[Pattern, float]:
    """μ_n of Eq. (distill-rec), with storage events when `storage`."""
    mu: dict[Pattern, float] = {(0, 0): 1.0}
    if not storage:
        for rot in protocol.rotations:
            mu = _flip(mu, (rot.output_flips, rot.check_flips), rot.q)
        return mu

    rotations = sorted(protocol.rotations, key=lambda rot: rot.time)
    events = _storage_events(protocol, rotations)
    k = 0
    for j in range(len(rotations) + 1):
        while k < len(events) and events[k][0] == j:
            _, qubit, pauli, p = events[k]
            mu = _storage_event(protocol, rotations, mu, j, qubit, pauli, p)
            k += 1
        if j < len(rotations):
            rot = rotations[j]
            mu = _flip(mu, (rot.output_flips, rot.check_flips), rot.q_total)
    return mu


def _metrics(mu: dict[Pattern, float]) -> tuple[float, float]:
    acceptance = sum(w for (e, c), w in mu.items() if c == 0)
    if acceptance <= 0.0:
        return 0.0, 1.0
    wrong = sum(w for (e, c), w in mu.items() if c == 0 and e != 0)
    return acceptance, min(1.0, wrong / acceptance)


def report_for(protocol: Protocol, *, storage: bool = True) -> DistillationReport:
    plain_acceptance, plain_error = evaluate(protocol, storage=False)
    acceptance, accepted_error = (
        evaluate(protocol, storage=True) if storage else (plain_acceptance, plain_error)
    )
    patterns = output_patterns(protocol, storage=storage)
    marginals = [
        min(1.0, math.fsum(w for e, w in patterns.items() if e >> i & 1))
        for i in range(len(protocol.outputs))
    ]
    return DistillationReport(
        output_errors=marginals,
        output_patterns=patterns,
        label=f"distillation in scope {protocol.scope}",
        rotations=len(protocol.rotations),
        checks=protocol.checks,
        outputs=protocol.outputs,
        acceptance=acceptance,
        accepted_error=accepted_error,
        rotation_error=max((rot.q_total if storage else rot.q) for rot in protocol.rotations),
        protocol=protocol,
        storage=storage,
        acceptance_no_storage=plain_acceptance,
        accepted_error_no_storage=plain_error,
    )


def analyze(
    est: "ResourceEstimatorV2", stmt: Allocation, env: "EstimateEnvV2"
) -> DistillationAnalysis | None:
    """Plug-in used by estimator v2: extract + evaluate, or None."""
    protocol = extract(est, stmt, env)
    if protocol is None:
        return None
    # Estimator v2 has no option for this; a caller may set `est.distill_storage`.
    storage = getattr(est, "distill_storage", True)
    return DistillationAnalysis(
        report_for(protocol, storage=storage), protocol.output_blocks, protocol.outputs
    )


def find_protocols(est: "ResourceEstimatorV2", body: Statement, env: "EstimateEnvV2") -> list[Protocol]:
    """All distillation scopes reachable from `body`, with calls unfolded."""
    found: list[Protocol] = []

    def walk(stmt: Statement, walk_env: "EstimateEnvV2", where: str) -> None:
        if isinstance(stmt, Allocation):
            child_env = est._alloc_env(stmt, walk_env)
            if tail_looks_like_checks(stmt):
                scope = f"{where}{stmt.register}[{stmt.count}]"
                protocol = extract(est, stmt, child_env, scope)
                if protocol is not None:
                    found.append(protocol)
                    return
            walk(stmt.body, child_env, where)
        elif isinstance(stmt, Sequence):
            for child in stmt.statements:
                walk(child, walk_env, where)
        elif isinstance(stmt, Parallel):
            walk(stmt.left, walk_env, where)
            walk(stmt.right, walk_env, where)
        elif isinstance(stmt, Conditional):
            walk(stmt.during, walk_env, where)
            walk(stmt.then_branch, walk_env, where)
        elif isinstance(stmt, RepeatUntil):
            walk(stmt.body, walk_env, where)
        elif isinstance(stmt, Call):
            body = est._instantiate_call(stmt, est.subcircuits[stmt.name], walk_env)
            walk(body, walk_env.copy(), f"{where}{stmt.name}: ")

    walk(body, env, "")
    return found


def format_qubit(q: Logical) -> str:
    return f"{q[0]}[{q[1]}]" if q[2] == 0 else f"{q[0]}[{q[1]}][{q[2]}]"


def format_axis(axis: dict[Logical, str]) -> str:
    """e.g. 'Z(anc[0]) Z(q[0])'."""
    return " ".join(f"{letter}({format_qubit(q)})" for q, letter in sorted(axis.items()))


def _flatten(
    est: "ResourceEstimatorV2",
    stmt: Statement,
    env: "EstimateEnvV2",
    out: list,
    local_blocks: set[BlockId],
    t0: float,
    alloc_time: dict[BlockId, float],
) -> float | None:
    """Unfold sequences, parallels, calls and nested allocations into atoms
    (stmt, env, start, duration), timed like (Seq) and (Par); return the end
    time, or None if the scope has a statement the pattern does not allow."""
    if isinstance(stmt, Skip):
        return t0
    if isinstance(stmt, Sequence):
        t: float | None = t0
        for child in stmt.statements:
            t = _flatten(est, child, env, out, local_blocks, t, alloc_time)
            if t is None:
                return None
        return t
    if isinstance(stmt, Parallel):
        left = _flatten(est, stmt.left, env, out, local_blocks, t0, alloc_time)
        right = _flatten(est, stmt.right, env, out, local_blocks, t0, alloc_time)
        return None if left is None or right is None else max(left, right)
    if isinstance(stmt, Allocation):
        child_env = est._alloc_env(stmt, env)
        for i in range(stmt.count):
            local_blocks.add((stmt.register, i))
            alloc_time[(stmt.register, i)] = t0
        return _flatten(est, stmt.body, child_env, out, local_blocks, t0, alloc_time)
    if isinstance(stmt, Call):
        body = est._instantiate_call(stmt, est.subcircuits[stmt.name], env)
        return _flatten(est, body, env.copy(), out, local_blocks, t0, alloc_time)
    if isinstance(stmt, PrimitiveOp):
        expanded = pauli_operation_as_parallel(stmt)
        if expanded is not None:
            return _flatten(est, expanded, env, out, local_blocks, t0, alloc_time)
    if isinstance(stmt, (PrimitiveOp, Reset, Conditional)):
        duration = _duration(est, stmt, env)
        out.append((stmt, env, t0, duration))
        return t0 + duration
    return None  # MapOp, XorConst, RepeatUntil and anything else


def _duration(est: "ResourceEstimatorV2", stmt: Statement, env: "EstimateEnvV2") -> float:
    """Time of `stmt` under the estimator's own rules (time does not depend on
    errors). Runs on a scratch context and source table, so the estimator's
    state is left as it was."""
    from ftqp_prototype.resource.estimator_v2 import BlockState, ErrorContextV2
    from ftqp_prototype.resource.sources import SourceTable

    blocks = {
        block: BlockState(env.registers[block[0]].scheme)
        for block in est.footprint(stmt, env)
        if block[0] in env.registers
    }
    saved = (est.table, est.detect_distillation, est.distillations)
    est.table, est.detect_distillation, est.distillations = SourceTable(), False, []
    try:
        return est.estimate(stmt, ErrorContextV2(blocks), env).time
    finally:
        est.table, est.detect_distillation, est.distillations = saved


# -- storage errors (see module docstring) ------------------------------------


def _rates(scheme) -> tuple[float, float]:
    return scheme.p_l_value(), scheme.t_se_value()


def _storage_risk(p_l: float, rounds: float) -> float:
    """r = 1 - (1 - p_L)^n for a (possibly fractional) number of rounds n."""
    if rounds <= 0 or p_l <= 0:
        return 0.0
    if p_l >= 1:
        return 1.0
    return -math.expm1(rounds * math.log1p(-p_l))


def _xor(*ps: float) -> float:
    """Flip probability of independent flips."""
    r = 0.0
    for p in ps:
        r = r * (1.0 - p) + (1.0 - r) * p
    return r


def _flip(mu: dict[Pattern, float], f: Pattern, p: float) -> dict[Pattern, float]:
    """F_f^p: flip the pattern by f with probability p."""
    if p <= 0.0 or f == (0, 0):
        return mu
    nxt: dict[Pattern, float] = {}
    for (e, c), weight in mu.items():
        nxt[(e, c)] = nxt.get((e, c), 0.0) + (1.0 - p) * weight
        key = (e ^ f[0], c ^ f[1])
        nxt[key] = nxt.get(key, 0.0) + p * weight
    return nxt


def _storage_events(protocol: Protocol, rotations: list[Rotation]) -> list[tuple[int, Logical, str, float]]:
    """(position, qubit, X|Z, probability): storage of each output and check
    between consecutive rotation times, from the scope start to its end."""
    times = sorted({rot.time for rot in rotations})
    bounds = [0.0, *times, max(protocol.end, times[-1] if times else 0.0)]
    events: list[tuple[int, Logical, str, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        position = sum(1 for rot in rotations if rot.time <= a)
        for qubit, (p_l, t_se) in protocol.storage_rates.items():
            r = _storage_risk(p_l, (b - a) / t_se if t_se > 0 else 0.0)
            if r > 0.0:
                events.append((position, qubit, "X", r / 2.0))
                events.append((position, qubit, "Z", r / 2.0))
    return events


def _storage_event(
    protocol: Protocol,
    rotations: list[Rotation],
    mu: dict[Pattern, float],
    position: int,
    qubit: Logical,
    pauli: str,
    p: float,
) -> dict[Pattern, float]:
    """μ ← (1-p) μ + p T(μ) for an error `pauli` on `qubit` after `position` rotations."""
    if qubit in protocol.outputs:
        index = protocol.outputs.index(qubit)
        own: Pattern = (1 << index, 0)
        basis = protocol.checks[0][1]
        is_output = True
    else:
        index = next(i for i, (q, _) in enumerate(protocol.checks) if q == qubit)
        own = (0, 1 << index)
        basis = protocol.checks[index][1]
        is_output = False

    def crosses(rot: Rotation) -> bool:
        return _anticommutes(rot.axis.get(qubit, "I"), pauli)

    before = [rot for rot in rotations[:position] if crosses(rot)]
    after = [rot for rot in rotations[position:] if crosses(rot)]
    flips_basis = _anticommutes(pauli, basis)
    transformed = mu
    if len(before) < len(after):
        # Moved to the start: the qubit starts in the +1 eigenstate of its basis.
        for rot in before:
            transformed = _flip(transformed, (rot.output_flips, rot.check_flips), 0.5)
        if flips_basis:
            transformed = _flip(transformed, own, 1.0)
    else:
        for rot in after:
            transformed = _flip(transformed, (rot.output_flips, rot.check_flips), 0.5)
        if flips_basis:
            transformed = _flip(transformed, own, 1.0)
        elif is_output:
            transformed = _flip(transformed, own, 0.5)  # |<m|X|m>|^2 = 1/2
    keys = set(mu) | set(transformed)
    return {
        key: (1.0 - p) * mu.get(key, 0.0) + p * transformed.get(key, 0.0) for key in keys
    }


def _pauli_axis(
    est: "ResourceEstimatorV2", stmt: Statement, env: "EstimateEnvV2"
) -> dict[Logical, str] | None:
    """The Pauli operator applied by a correction, as {logical qubit: letter}."""
    if isinstance(stmt, Skip):
        return {}
    if isinstance(stmt, PrimitiveOp):
        expanded = pauli_operation_as_parallel(stmt)
        if expanded is not None:
            return _pauli_axis(est, expanded, env)
        if stmt.name in {"X", "Y", "Z"} and len(stmt.targets) == 1:
            return {est._resolve_logical(stmt.targets[0], env): stmt.name}
        return None
    if isinstance(stmt, (Parallel, Sequence)):
        parts = [stmt.left, stmt.right] if isinstance(stmt, Parallel) else stmt.statements
        axis: dict[Logical, str] = {}
        for part in parts:
            sub = _pauli_axis(est, part, env)
            if sub is None:
                return None
            for qubit, letter in sub.items():
                axis[qubit] = _multiply(axis.get(qubit, "I"), letter)
        return {q: letter for q, letter in axis.items() if letter != "I"}
    return None


def _parallel_leaves(stmt: Statement) -> list[Statement]:
    if isinstance(stmt, Parallel):
        return _parallel_leaves(stmt.left) + _parallel_leaves(stmt.right)
    return [stmt]


def _is_check_shape(stmt: Statement) -> bool:
    return (
        isinstance(stmt, Conditional)
        and isinstance(stmt.during, Skip)
        and isinstance(stmt.then_branch, Skip)
        and len(stmt.targets) == 1
        and stmt.measurement.name in {"X", "Y", "Z"}
    )


def _is_pauli_name(name: str) -> bool:
    return bool(name) and all(ch in "IXYZ" for ch in name)


def _anticommutes(a: str, b: str) -> bool:
    return a != "I" and b != "I" and a != b


def _multiply(a: str, b: str) -> str:
    """Pauli product up to phase."""
    if a == "I":
        return b
    if b == "I":
        return a
    if a == b:
        return "I"
    return ({"X", "Y", "Z"} - {a, b}).pop()
