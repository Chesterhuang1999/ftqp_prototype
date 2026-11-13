"""Runtime models for hardware and QEC declarations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ftqp_prototype.parser.ast import HardwareDecl, OperationDecl, QECDecl
from ftqp_prototype.resource.expressions import ExpressionEvaluator


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    p1: float
    p2: float
    pm: float
    t1: float
    t2: float
    tm: float

    @classmethod
    def from_decl(cls, decl: HardwareDecl) -> "HardwareConfig":
        return cls(
            name=decl.name,
            p1=decl.params["p1"],
            p2=decl.params["p2"],
            pm=decl.params["pm"],
            t1=decl.params["t1"],
            t2=decl.params["t2"],
            tm=decl.params["tm"],
        )

    def variables(self) -> dict[str, float]:
        return {
            "p1": self.p1,
            "p2": self.p2,
            "pm": self.pm,
            "t1": self.t1,
            "t2": self.t2,
            "tm": self.tm,
        }


@dataclass(frozen=True)
class RuntimeOperation:
    name: str
    arity: int
    time: str
    ancilla: str
    kind: str
    n_se: str | None = None
    p_error: str | None = None

    @classmethod
    def from_decl(cls, decl: OperationDecl) -> "RuntimeOperation":
        return cls(
            name=decl.name,
            arity=decl.arity,
            time=decl.time,
            ancilla=decl.ancilla,
            kind=decl.kind,
            n_se=decl.n_se,
            p_error=decl.p_error,
        )


@dataclass
class QecScheme:
    name: str
    hardware: HardwareConfig
    n: int
    k: int
    d: int
    se_ancilla: str
    t_se: str
    prep_time: str
    prep_n_se: str
    prep_ancilla: str
    logical_ops: dict[str, RuntimeOperation] = field(default_factory=dict)
    faulty_ops: dict[str, RuntimeOperation] = field(default_factory=dict)
    t_dec: str = "0"
    p_l: str = "0"
    n_aux: str = "0"

    @classmethod
    def from_decl(cls, decl: QECDecl, hardware: HardwareConfig) -> "QecScheme":
        return cls(
            name=decl.name,
            hardware=hardware,
            n=decl.n,
            k=decl.k,
            d=decl.d,
            se_ancilla=decl.se_ancilla,
            t_se=decl.t_se,
            prep_time=decl.prep_time,
            prep_n_se=decl.prep_n_se,
            prep_ancilla=decl.prep_ancilla,
            logical_ops={k: RuntimeOperation.from_decl(v) for k, v in decl.logical_ops.items()},
            faulty_ops={k: RuntimeOperation.from_decl(v) for k, v in decl.faulty_ops.items()},
            t_dec=decl.t_dec,
            p_l=decl.p_l,
            n_aux=decl.n_aux,
        )

    def variables(self) -> dict[str, float]:
        vars_: dict[str, float] = {
            "n": float(self.n),
            "k": float(self.k),
            "d": float(self.d),
            **self.hardware.variables(),
        }
        t_se = ExpressionEvaluator(vars_).eval(self.t_se)
        vars_.update(
            {
                "tSE": t_se,
                "t_SE": t_se,
                "tse": t_se,
                "p": max(self.hardware.p1, self.hardware.p2, self.hardware.pm),
            }
        )
        try:
            vars_["pL"] = ExpressionEvaluator(vars_).eval(self.p_l)
            vars_["p_l"] = vars_["pL"]
        except Exception:
            # Validation reports the real error. Keeping variables partial makes
            # one bad expression less likely to hide unrelated diagnostics.
            pass
        return vars_

    def eval(self, expr: str | int | float) -> float:
        return ExpressionEvaluator(self.variables()).eval(expr)

    def t_se_value(self) -> float:
        return self.eval(self.t_se)

    def se_ancilla_value(self) -> int:
        return max(0, int(math.ceil(self.eval(self.se_ancilla))))

    def p_l_value(self) -> float:
        return _probability(self.eval(self.p_l))

    def t_dec_value(self) -> float:
        return max(0.0, self.eval(self.t_dec))

    def n_aux_value(self) -> int:
        return max(0, int(math.ceil(self.eval(self.n_aux))))

    def find_operation(self, name: str) -> RuntimeOperation | None:
        return self.logical_ops.get(name) or self.faulty_ops.get(name)


def joint_measurement_times(schemes: list["QecScheme"]) -> tuple[int, float]:
    """d_joint and T_meas of a joint Pauli measurement (Eq. njoint)."""
    d_joint = min(s.d for s in schemes)
    return d_joint, d_joint * max(2.0 * s.t_se_value() for s in schemes)


def joint_measurement_risk(schemes: list["QecScheme"]) -> float:
    """r_M of Eq. (njoint): max over operands of 1 - (1 - p_L)^ceil(T_meas / t_SE).

    Each operand is charged at least the risk it would accrue idling for
    T_meas, as rule (Ctl) requires. To first order this is 2 d_joint max p_L.
    A scheme with t_SE = 0 is charged 2 d_joint rounds.
    """
    d_joint, time_meas = joint_measurement_times(schemes)
    worst = 0.0
    for s in schemes:
        p_l = s.p_l_value()
        if p_l <= 0.0:
            continue
        t_se = s.t_se_value()
        x = time_meas / t_se if t_se > 0 else 2.0 * d_joint
        rounds = max(0, math.ceil(x - 1e-9 * max(1.0, abs(x))))
        if p_l >= 1.0:
            risk = 1.0 if rounds > 0 else 0.0
        else:
            risk = -math.expm1(rounds * math.log1p(-p_l))
        worst = max(worst, risk)
    return _probability(worst)


def _probability(value: float) -> float:
    return min(1.0, max(0.0, value))
