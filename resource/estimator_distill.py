"""Standalone estimator for distillation programs (arXiv-2608.04573 §5.5).

Time and space come from the normal rules (the same derivation as estimator
v2, so they equal v1 and v2). Only the error analysis differs: instead of the
disturbance risk of §5.4, which ignores postselection, each rotation-form
distillation protocol found in the program is summarized by

* the acceptance probability A of one attempt, and
* the error ε_acc of an accepted output,

from the recursion of Eq. (distill-rec). The disturbance risk is reported as
well, for comparison.

Model limits (as stated in §5.5): faulty-T errors are reduced exactly and the
coherent part of measurement errors is twirled away. Storage errors of the
outputs and checks are included by default (`storage=False`: the plain §5.5
recursion); see the module docstring of `distill`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, fields

from ftqp_prototype.parser.ast import Program
from ftqp_prototype.resource import distill
from ftqp_prototype.resource.estimator import ResourceEstimationError
from ftqp_prototype.resource.estimator_v2 import (
    EstimateEnvV2,
    ResourceEstimateV2,
    ResourceEstimatorV2,
)
from ftqp_prototype.typechecker.checker import CheckResult


@dataclass
class DistillationEstimate(ResourceEstimateV2):
    protocols: list[distill.DistillationReport] = field(default_factory=list)
    accepted_output_error: float | None = None  # ε_acc (combined over protocols)
    acceptance: float | None = None  # A (product over protocols)
    disturbance_risk: float | None = None  # §5.4 output risk, no postselection


class DistillationEstimator:
    def __init__(
        self, symbols: CheckResult, *, outputs: Iterable[str] | None = None, storage: bool = True
    ) -> None:
        self.symbols = symbols
        self.outputs = list(outputs) if outputs is not None else None
        self.storage = storage

    def estimate_program(self, program: Program) -> DistillationEstimate:
        # Resources: the normal rules, without the distillation plug-in, so the
        # output errors below are the plain §5.4 disturbance risks.
        base = ResourceEstimatorV2(
            self.symbols, outputs=self.outputs, detect_distillation=False
        ).estimate_program(program)

        helper = ResourceEstimatorV2(self.symbols)
        protocols = (
            distill.find_protocols(helper, program.body, EstimateEnvV2())
            if program.body is not None
            else []
        )
        if not protocols:
            raise ResourceEstimationError(
                "no rotation-form distillation protocol found: expected an allocation scope "
                "whose body applies rotations (joint M(P·Z) with an ancilla, then a faulty "
                "measurement controlling P) and ends with check measurements "
                "`(if M(B)[anc[i]]; then skip) || …` on its own ancillas"
            )
        reports = [distill.report_for(protocol, storage=self.storage) for protocol in protocols]

        values = {f.name: getattr(base, f.name) for f in fields(ResourceEstimateV2)}
        values["semantics"] = "distill"
        return DistillationEstimate(
            **values,
            protocols=reports,
            accepted_output_error=-math.expm1(sum(math.log1p(-r.accepted_error) for r in reports)),
            acceptance=math.prod(r.acceptance for r in reports),
            disturbance_risk=base.output_risk,
        )
