"""Tests for estimator v2 (source-aware error analysis) and the closed-body check.

Run (from anywhere):
    python3 ftqp_prototype/tests/test_estimator_v2.py -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR.parent))

from ftqp_prototype.parser.ft_parser import parse_file, parse_text  # noqa: E402
from ftqp_prototype.resource.estimator import ResourceEstimator  # noqa: E402
from ftqp_prototype.resource.estimator import ResourceEstimationError  # noqa: E402
from ftqp_prototype.resource.estimator_distill import DistillationEstimator  # noqa: E402
from ftqp_prototype.resource.estimator_v2 import ResourceEstimatorV2  # noqa: E402
from ftqp_prototype.typechecker import TypeChecker, ValidationError  # noqa: E402

EXAMPLES = PACKAGE_DIR / "examples"
# Deliberately invalid, or too slow for a unit test (eea.ft: ~2M lines).
SKIP = {"invalid_parallel.ft", "naive.ft", "eea.ft"}

# d = 27 surface code on the logicalT.ft hardware: tSE = 1.42 µs, pL ≈ 1.2516e-5.
HEADER = """
hardware hw = { t1: 1e-8, t2: 1e-7, tm: 1e-6, p1: 1e-3, p2: 2e-3, pm: 3e-3 };
qecc sc using hw = {
  [[n,k,d]]: [729, 1, 27],
  SE: { "#ancilla": "n-1", tSE: "2t1+4t2+tm" },
  PREP: { time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
  OPL: {
    I: { arity: 1, time: "0", "#SE": "0", "#ancilla": "0" },
    X: { arity: 1, time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
    Z: { arity: 1, time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
    H: { arity: 1, time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
    half: { arity: 1, time: "0.5*tSE", "#SE": "0", "#ancilla": "0" }
  },
  OPF: { faultyT: { arity: 1, time: "2t1+2tm", pF: "max(p1, p2, pm)", "#ancilla": "0" } },
  tdec: "390*tSE",
  pL: "0.1*(max(p1,p2,pm)/0.0057)^((d+1)/2)",
  naux: "d*d"
};
qecc sc2 using hw = {
  [[n,k,d]]: [1458, 2, 27],
  SE: { "#ancilla": "n-1", tSE: "2t1+4t2+tm" },
  PREP: { time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
  OPL: { X: { arity: 1, time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" } },
  OPF: { },
  tdec: "0*tSE",
  pL: "0.1*(max(p1,p2,pm)/0.0057)^((d+1)/2)",
  naux: "d*d"
};
"""
P_L = 0.1 * (3e-3 / 0.0057) ** 14
R_M = 1 - (1 - P_L) ** 54  # joint-measurement risk: 2 d_joint = 54 rounds at d_joint = 27


def r(n: float) -> float:
    return 1 - (1 - P_L) ** n


def risk(*ps: float) -> float:
    return 1 - math.prod(1 - p for p in ps)


def check(text: str):
    program = parse_text(text, "<test>")
    return program, TypeChecker().check(program)


def run_v2(text: str, **kwargs):
    program, symbols = check(text)
    return ResourceEstimatorV2(symbols, **kwargs).estimate_program(program)


def output_eps(estimate) -> dict[str, float]:
    return {f"{reg}[{idx}]": st.epsilon for (reg, idx), st in estimate.outputs.items()}


class Close(unittest.TestCase):
    def assertClose(self, actual: float, expected: float, rel: float = 1e-9) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-300),
            f"{actual!r} != {expected!r} (rel_tol={rel})",
        )


class TimeSpaceParity(unittest.TestCase):
    """The paper keeps the time and space rules, so v2 must match v1 exactly."""

    def test_all_examples(self):
        files = sorted(p for p in EXAMPLES.rglob("*.ft") if p.name not in SKIP)
        self.assertGreater(len(files), 50)
        for path in files:
            with self.subTest(example=str(path.relative_to(EXAMPLES))):
                program = parse_file(path)
                symbols = TypeChecker().check(program)
                v1 = ResourceEstimator(symbols).estimate_program(program)
                v2 = ResourceEstimatorV2(symbols).estimate_program(program)
                self.assertTrue(math.isclose(v1.time, v2.time, rel_tol=1e-12, abs_tol=0.0))
                self.assertEqual(v1.space, v2.space)


class Rules(Close):
    def test_logicalT_step_008_counts_shared_source_once(self):
        program = parse_file(EXAMPLES / "logicalT.ft")
        estimate = ResourceEstimatorV2(TypeChecker().check(program)).estimate_program(program)
        entry = next(e for e in estimate.trace if e.label == "conditional M(X) complete")
        q = next(st for (reg, _), st in entry.context.blocks.items() if reg == "q")
        # prepare(anc) [shared into q by the ZZ snapshot], q idling through it
        # (t1 + d tSE: 28 rounds), M(ZZ), M(X), q idling through M(X) (54 rounds),
        # the decoding wait (390) and the correction Z.
        expected = risk(r(27), r(28), R_M, R_M, r(54), r(390), r(27))
        self.assertClose(q.epsilon, expected)

    def test_same_block_joint_measurements_are_not_doubled(self):
        body = "  if M(ZZ)[q[0], q[1]]; then skip;\n" * 4
        text = HEADER + "program {\n  logical q := new sc2;\n" + body + "  release(q);\n}"
        # q ends with a measurement; it is still tracked (its readout error counts).
        est = run_v2(text)
        self.assertClose(output_eps(est)["q[0]"], risk(*[R_M] * 4))

    def test_idle_rounds_are_counted_cumulatively(self):
        text = HEADER + """
program {
  logical a := new sc;
  logical b := new sc;
  half[a]; half[a];
  release(b);
  release(a);
}"""
        # b idles 2 × 0.5 tSE = 1 round in total (v1 would charge 2).
        self.assertClose(output_eps(run_v2(text))["b[0]"], r(1))

    def test_zero_length_op_does_not_restart_idle_period(self):
        text = HEADER + """
program {
  logical a := new sc;
  logical b := new sc;
  half[a]; I[b]; half[a];
  release(b);
  release(a);
}"""
        self.assertClose(output_eps(run_v2(text))["b[0]"], r(1))

    def test_prepare_wipes_only_its_own_block(self):
        text = HEADER + """
program {
  logical a := new sc;
  logical b := new sc;
  if M(ZZ)[a, b]; then skip;
  prepare(a);
  release(b);
  release(a);
}"""
        eps = output_eps(run_v2(text))
        prep_time_rounds = math.ceil((1e-8 + 27 * 1.42e-6) / 1.42e-6)  # b idles during prepare(a)
        self.assertClose(eps["a[0]"], r(27))
        self.assertClose(eps["b[0]"], risk(R_M, r(2 * 390), r(prep_time_rounds)))

    def test_released_ancilla_leaves_shared_source_behind(self):
        text = HEADER + """
program {
  logical q := new sc;
  logical anc := new sc;
  if M(ZZ)[q, anc]; then skip;
  release(anc);
  X[q];
  release(q);
}"""
        # anc is released mid-program, so it is an ancilla; q keeps s_ZZ.
        est = run_v2(text)
        self.assertEqual(set(output_eps(est)), {"q[0]"})
        self.assertClose(est.output_risk, risk(R_M, r(2 * 390), r(27)))

    def test_each_call_creates_fresh_sources(self):
        text = HEADER + """
def flip(t: Codeblock) { X[t]; }
program {
  logical a := new sc;
  logical b := new sc;
  call flip(a);
  call flip(a);
  release(b);
  release(a);
}"""
        self.assertClose(output_eps(run_v2(text))["a[0]"], risk(r(27), r(27)))

    def test_ctl_envelope_for_reset_blocks(self):
        # The branch re-prepares b, so b keeps its pre-branch sources (skip path)
        # plus an idle source for the branch duration, on top of s_prep and U.
        text = HEADER + """
program {
  logical a := new sc;
  logical b := new sc;
  if M(ZZ)[a, b]; then skip;
  if M(Z)[a]; then { prepare(b); };
  release(b);
  release(a);
}"""
        eps = output_eps(run_v2(text))
        prep_rounds = math.ceil((1e-8 + 27 * 1.42e-6) / 1.42e-6)
        executing = [r(27)]  # s_prep
        snapshot = [R_M, r(2 * 390), R_M]  # U = D_a = {s_ZZ, a's first wait} ∪ {s_M(Z)}
        # b before the branch: s_ZZ (already in U, counted once), its own first
        # wait, idle during M(Z) and its decoding wait; then the skip-path idle.
        skip_path = [r(2 * 390), r(54), r(390), r(prep_rounds)]
        expected = risk(*executing, *snapshot, *skip_path)
        self.assertClose(eps["b[0]"], expected)

    def test_untouched_call_argument_does_not_absorb_snapshot(self):
        text = HEADER + """
def fix(t: Codeblock, spare: Codeblock) { X[t]; }
program {
  logical a := new sc;
  logical b := new sc;
  logical c := new sc;
  if M(ZZ)[a, b]; then call fix(b, c);
  release(c);
  release(b);
  release(a);
}"""
        eps = output_eps(run_v2(text))
        self.assertClose(eps["c[0]"], risk(r(54), r(780), r(28)))


class Outputs(unittest.TestCase):
    def outputs_of(self, relative: str) -> set[str]:
        program = parse_file(EXAMPLES / relative)
        estimate = ResourceEstimatorV2(TypeChecker().check(program)).estimate_program(program)
        return {reg for reg, _ in estimate.outputs}

    def test_default_outputs(self):
        # Level 1: the reserved name `anc` is an ancilla, wherever it is released.
        self.assertEqual(self.outputs_of("cnot.ft"), {"ctl", "tgt"})
        self.assertEqual(self.outputs_of("miscellaneous/idle.ft"), {"q"})
        # Level 2: q ends with a measurement on itself but is still an output.
        self.assertEqual(self.outputs_of("miscellaneous/if.ft"), {"q"})
        self.assertEqual(self.outputs_of("logicalT.ft"), {"q"})
        self.assertEqual(self.outputs_of("distillation/15to1-d11.ft"), {"q"})

    def test_data_register_released_mid_program_is_an_ancilla(self):
        text = HEADER + """
program {
  logical q := new sc;
  logical tmp := new sc;
  if M(ZZ)[q, tmp]; then skip;
  release(tmp);
  X[q];
  release(q);
}"""
        self.assertEqual(set(output_eps(run_v2(text))), {"q[0]"})

    def test_requested_outputs(self):
        program = parse_file(EXAMPLES / "cnot.ft")
        symbols = TypeChecker().check(program)
        est = ResourceEstimatorV2(symbols, outputs=["anc"]).estimate_program(program)
        self.assertEqual({reg for reg, _ in est.outputs}, {"anc"})


class Distillation(Close):
    def test_15to1_matches_closed_form(self):
        program = parse_file(EXAMPLES / "distillation/15to1-d11.ft")
        est = ResourceEstimatorV2(TypeChecker().check(program)).estimate_program(program)
        self.assertEqual(len(est.distillations), 1)
        report = est.distillations[0]
        self.assertEqual((report.rotations, len(report.checks), len(report.outputs)), (15, 4, 1))
        q = max(r.q for r in report.protocol.rotations)
        self.assertClose(report.acceptance_no_storage, (1 + 15 * (1 - 2 * q) ** 8) / 16)
        self.assertAlmostEqual(report.accepted_error_no_storage / (35 * q**3), 1.0, delta=0.01)
        self.assertClose(est.output_risk, report.accepted_error)

    def test_several_outputs_match_bravyi_haah(self):
        """(3k+8)-to-k: acceptance 1-(3k+8)q, one output (3k+1)q², any output (4+3k(k-1)/2)q²."""
        for k in (2, 4, 6, 8):
            with self.subTest(k=k):
                program = parse_file(EXAMPLES / f"distillation/bh/bh{k}.ft")
                symbols = TypeChecker().check(program)
                est = ResourceEstimatorV2(symbols).estimate_program(program)
                (report,) = est.distillations
                self.assertEqual(
                    (report.rotations, len(report.checks), len(report.outputs)), (3 * k + 8, 3, k)
                )
                q = max(r.q for r in report.protocol.rotations)
                any_coeff = 4 + 3 * k * (k - 1) // 2
                self.assertAlmostEqual(report.acceptance, 1 - (3 * k + 8) * q, delta=1e-5)
                self.assertAlmostEqual(report.accepted_error / (any_coeff * q * q), 1.0, delta=2e-3)
                self.assertEqual(len(report.output_errors), k)
                for marginal in report.output_errors:
                    self.assertAlmostEqual(marginal / ((3 * k + 1) * q * q), 1.0, delta=2e-3)
                # The outputs fail together: the joint error is eps_acc, not k times it,
                # and each output block carries (a bound on) its own marginal error.
                self.assertClose(est.output_risk, report.accepted_error)
                self.assertEqual(len(est.outputs), k)
                for state, marginal in zip(est.outputs.values(), report.output_errors):
                    self.assertGreaterEqual(state.epsilon, marginal * (1 - 1e-12))
                    self.assertAlmostEqual(state.epsilon / marginal, 1.0, delta=1e-6)
                # Same numbers in the reference (uncompressed) mode.
                ref = ResourceEstimatorV2(symbols, compress=False).estimate_program(program)
                self.assertClose(ref.output_risk, est.output_risk)

    def test_not_detected_in_t_gate_examples(self):
        for path in sorted((EXAMPLES / "logicalT-exp").rglob("*.ft")):
            with self.subTest(example=path.name):
                program = parse_file(path)
                est = ResourceEstimatorV2(TypeChecker().check(program)).estimate_program(program)
                self.assertEqual(est.distillations, [])


class StandaloneDistillationEstimator(Close):
    def estimate(self, relative: str):
        program = parse_file(EXAMPLES / relative)
        symbols = TypeChecker().check(program)
        return (
            DistillationEstimator(symbols).estimate_program(program),
            ResourceEstimatorV2(symbols).estimate_program(program),
        )

    def test_all_15to1_examples(self):
        for path in sorted((EXAMPLES / "distillation").glob("15to1-d*.ft")):
            with self.subTest(example=path.name):
                est, v2 = self.estimate(f"distillation/{path.name}")
                # Resources follow the normal rules.
                self.assertEqual(est.space, v2.space)
                self.assertTrue(math.isclose(est.time, v2.time, rel_tol=1e-12))
                (report,) = est.protocols
                rots = report.protocol.rotations
                self.assertEqual((len(rots), len(report.checks), len(report.outputs)), (15, 4, 1))
                # 15-to-1: 7 rotations touch the output; every non-trivial check
                # pattern appears once (the [[15,1,3]] structure).
                self.assertEqual(sum(1 for r in rots if r.output_flips), 7)
                self.assertEqual(sorted(r.check_flips for r in rots), list(range(1, 16)))
                q = max(r.q for r in rots)
                self.assertClose(report.acceptance_no_storage, (1 + 15 * (1 - 2 * q) ** 8) / 16)
                if q > 1e-4:  # small d: q_m is no longer negligible against q_F
                    continue
                self.assertAlmostEqual(
                    report.accepted_error_no_storage / (35 * q**3), 1.0, delta=0.01
                )
                # Storage errors can only add to the accepted error.
                self.assertGreaterEqual(report.accepted_error, report.accepted_error_no_storage)
                # Same core as the v2 plug-in.
                self.assertClose(report.accepted_error, v2.distillations[0].accepted_error)
                self.assertGreater(est.disturbance_risk, report.accepted_error)

    def test_paper_example_values(self):
        est, _ = self.estimate("distillation/15to1-d11.ft")
        (report,) = est.protocols
        self.assertAlmostEqual(report.acceptance_no_storage, 0.99900, delta=5e-6)
        self.assertAlmostEqual(report.accepted_error_no_storage / 1.04e-11, 1.0, delta=0.01)
        # With storage errors (appendix): 4.6e-11 at d = 11.
        self.assertAlmostEqual(est.accepted_output_error / 4.6083e-11, 1.0, delta=0.001)

    def test_rejects_program_without_protocol(self):
        with self.assertRaises(ResourceEstimationError):
            self.estimate("logicalT.ft")


class DistillationStorage(Close):
    """Storage errors of outputs and checks in the recursion (distill.py)."""

    # Independent storage recursion with r_M = 1-(1-p_L)^(2d), on the schedule written
    # out by hand (prepare/H 2d, 12 rotation slots of 4d + 2.02, checks 2d).
    PROTOTYPE = {3: 1.0802e-3, 5: 1.7034e-5, 7: 2.3261e-7, 9: 2.9589e-9, 11: 4.6083e-11, 21: 1.037e-11}

    def estimate(self, d: int, **kwargs):
        program = parse_file(EXAMPLES / f"distillation/15to1-d{d}.ft")
        return DistillationEstimator(TypeChecker().check(program), **kwargs).estimate_program(program)

    def test_matches_hand_written_schedule(self):
        for d, expected in self.PROTOTYPE.items():
            with self.subTest(d=d):
                self.assertClose(self.estimate(d).accepted_output_error, expected, rel=1e-3)

    def test_rotation_times_follow_the_schedule(self):
        (report,) = self.estimate(11).protocols
        cycle = 1e-6  # fixed one-microsecond SE round
        times = sorted({round(r.time / cycle, 6) for r in report.protocol.rotations})
        self.assertEqual(len(times), 12)  # four parallel rotations, then eleven
        self.assertClose(times[0], 2 * 11 + 22, rel=1e-12)
        self.assertClose(times[1] - times[0], 4 * 11 + 2.02, rel=1e-7)

    def test_off_switch_gives_plain_recursion(self):
        on, off = self.estimate(5), self.estimate(5, storage=False)
        (r_on,), (r_off,) = on.protocols, off.protocols
        self.assertFalse(r_off.storage)
        self.assertEqual(off.accepted_output_error, r_on.accepted_error_no_storage)
        self.assertEqual(off.acceptance, r_on.acceptance_no_storage)
        self.assertGreater(on.accepted_output_error, 1e3 * off.accepted_output_error)
        self.assertLess(on.acceptance, off.acceptance)  # checks catch stored errors too
        self.assertEqual((on.time, on.space), (off.time, off.space))

    def test_v2_plugin_switch_and_untouched_resources(self):
        program = parse_file(EXAMPLES / "distillation/15to1-d7.ft")
        symbols = TypeChecker().check(program)
        default = ResourceEstimatorV2(symbols).estimate_program(program)
        plain_est = ResourceEstimatorV2(symbols)
        plain_est.distill_storage = False
        plain = plain_est.estimate_program(program)
        self.assertEqual((default.time, default.space), (plain.time, plain.space))
        self.assertClose(default.output_risk, self.estimate(7).accepted_output_error)
        self.assertClose(plain.output_risk, self.estimate(7, storage=False).accepted_output_error)
        # Timing the protocol must not leave sources behind in the estimator.
        from ftqp_prototype.resource import distill
        from ftqp_prototype.resource.estimator_v2 import EstimateEnvV2

        sources = len(plain_est.table.p)
        self.assertEqual(len(distill.find_protocols(plain_est, program.body, EstimateEnvV2())), 1)
        self.assertEqual(len(plain_est.table.p), sources)
        self.assertEqual(plain_est.distillations[0].storage, False)


class CompressedMatchesReference(Close):
    """Merging private sources must not change any error number."""

    def test_examples(self):
        files = sorted(
            p
            for p in EXAMPLES.rglob("*.ft")
            if p.name not in SKIP and "compare_ms" not in p.parts
        )
        for path in files:
            with self.subTest(example=str(path.relative_to(EXAMPLES))):
                program = parse_file(path)
                symbols = TypeChecker().check(program)
                fast = ResourceEstimatorV2(symbols, compress=True).estimate_program(program)
                ref = ResourceEstimatorV2(symbols, compress=False).estimate_program(program)
                self.assertEqual(len(fast.trace), len(ref.trace))
                for a, b in zip(fast.trace, ref.trace):
                    for block, st in a.context.blocks.items():
                        self.assertAlmostEqual(
                            st.epsilon, b.context.blocks[block].epsilon, delta=1e-13
                        )
                if ref.output_risk is not None:
                    self.assertClose(fast.output_risk, ref.output_risk)


class ClosedSubcircuitBodies(unittest.TestCase):
    def test_free_register_is_rejected(self):
        text = HEADER + """
def fix(t: Codeblock, spare: Codeblock) { X[t]; X[g]; }
program {
  logical g := new sc;
  logical b := new sc;
  logical c := new sc;
  (call fix(b, c)) || X[g];
  release(c);
  release(b);
  release(g);
}"""
        with self.assertRaises(ValidationError) as ctx:
            check(text)
        self.assertIn("uses register 'g'", str(ctx.exception))

    def test_parameters_and_local_allocations_are_fine(self):
        text = HEADER + """
def gadget(t: Codeblock) {
  logical anc := new sc;
  if M(ZZ)[t, anc]; then X[t];
  release(anc);
}
program {
  logical q := new sc;
  call gadget(q);
  release(q);
}"""
        check(text)


if __name__ == "__main__":
    unittest.main()
