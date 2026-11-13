#!/usr/bin/env python3
"""Litinski's cost model of the one-level 15-to-1 protocol, and a comparison
with our estimators on examples/distillation/15to1-d*.ft.

Model: D. Litinski, "Magic State Distillation: Not as Costly as You Think",
Quantum 3, 205 (2019), and its code (github.com/litinski/magicstates,
Python/onelevel15to1.py, Apache-2.0). This is an independent reimplementation
of the same model:

  * 5 qubits in |+>, qubit 1 is the output, qubits 2-5 are the checks;
  * 15 rotations exp(i pi/8 P_j) in 6 time steps (Fig. 11 of the paper);
  * a faulty T measurement turns pi/8 into 5pi/8 (Pauli error P_j),
    -pi/8 or 3pi/8, each with probability p/3;
  * storage errors (distances d_X, d_Z) and lattice-surgery errors (d_m) with
    p_L(d) = 0.1 (100 p)^((d+1)/2) per code cycle, added to those channels or
    applied as X/Z errors on the five patches;
  * accept when qubits 2-5 are found in |+>; p_out = 1 - <m|rho_out|m>.

Cost: 2((d_X + 4 d_Z) 3 d_X + 2 d_m) qubits, 6 d_m / (1 - p_fail) code cycles.

The density matrix is diagonal-friendly: every rotation is diagonal in the Z
basis, so all channels act entrywise or by index permutation. We keep 50 digits
(decimal) because p_out ~ 1e-11 cancels against O(1) entries.

Usage:
  scripts/litinski15to1.py validate            # Table 1 of Litinski (2019)
  scripts/litinski15to1.py compare [--output DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 50

PACKAGE_DIR = Path(__file__).resolve().parents[1]
EXAMPLES = PACKAGE_DIR / "examples" / "distillation"

ONE = Decimal(1)
HALF = Decimal("0.5")
SQRT2_2 = Decimal(2).sqrt() / 2  # cos(pi/4) = sin(pi/4)

# e^{i 2 theta} for theta = pi/8, 5pi/8, -pi/8, 3pi/8 (ideal, Pauli, -pi/4, +pi/4 error)
PHASE2 = [
    (SQRT2_2, SQRT2_2),  # e^{i pi/4}
    (-SQRT2_2, -SQRT2_2),  # e^{i 5pi/4}
    (SQRT2_2, -SQRT2_2),  # e^{-i pi/4}
    (-SQRT2_2, SQRT2_2),  # e^{i 3pi/4}
]

N = 5
DIM = 1 << N


def bit(q: int) -> int:
    """Mask of qubit q (1-based, qubit 1 is the most significant, as in kron)."""
    return 1 << (N - q)


def plog(p: Decimal, d: int) -> Decimal:
    return Decimal("0.1") * (100 * p) ** ((d + 1) / Decimal(2))


class State:
    """32x32 density matrix as a dict-free list of complex (re, im) Decimals."""

    def __init__(self) -> None:
        v = ONE / DIM
        self.re = [[v] * DIM for _ in range(DIM)]
        self.im = [[Decimal(0)] * DIM for _ in range(DIM)]

    def rotation(self, axis: str, p1: Decimal, p2: Decimal, p3: Decimal) -> None:
        """Mixture of exp(i theta P) for theta = pi/8, 5pi/8, -pi/8, 3pi/8."""
        probs = [ONE - p1 - p2 - p3, p1, p2, p3]
        cre = sum(pk * ph[0] for pk, ph in zip(probs, PHASE2))
        cim = sum(pk * ph[1] for pk, ph in zip(probs, PHASE2))
        mask = sum(bit(i + 1) for i, c in enumerate(axis) if c == "z")
        sign = [1 - 2 * (bin(j & mask).count("1") & 1) for j in range(DIM)]
        for j in range(DIM):
            rj, ij = self.re[j], self.im[j]
            for l in range(DIM):
                if sign[j] == sign[l]:
                    continue
                s = cim if sign[j] > 0 else -cim  # conj for sigma = -1
                a, b = rj[l], ij[l]
                rj[l], ij[l] = a * cre - b * s, a * s + b * cre

    def pauli_z(self, q: int, p: Decimal) -> None:
        if p == 0:
            return
        m = bit(q)
        f = ONE - 2 * p
        for j in range(DIM):
            for l in range(DIM):
                if (j ^ l) & m:
                    self.re[j][l] *= f
                    self.im[j][l] *= f

    def pauli_x(self, q: int, p: Decimal) -> None:
        if p == 0:
            return
        m = bit(q)
        re, im = self.re, self.im
        nre = [[(ONE - p) * re[j][l] + p * re[j ^ m][l ^ m] for l in range(DIM)] for j in range(DIM)]
        nim = [[(ONE - p) * im[j][l] + p * im[j ^ m][l ^ m] for l in range(DIM)] for j in range(DIM)]
        self.re, self.im = nre, nim

    def storage(self, x: list[Decimal], z: list[Decimal]) -> None:
        for q in range(1, N + 1):
            self.pauli_x(q, x[q - 1])
        for q in range(1, N + 1):
            self.pauli_z(q, z[q - 1])

    def output_after_checks(self) -> tuple[list[list[tuple[Decimal, Decimal]]], Decimal]:
        """Project qubits 2-5 onto |+>: unnormalized 2x2 output state and its trace."""
        out = [[(Decimal(0), Decimal(0))] * 2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                sr = si = Decimal(0)
                for c in range(16):
                    for c2 in range(16):
                        sr += self.re[a * 16 + c][b * 16 + c2]
                        si += self.im[a * 16 + c][b * 16 + c2]
                out[a][b] = (sr / 16, si / 16)
        return out, out[0][0][0] + out[1][1][0]


def run(p: Decimal, dx: int, dz: int, dm: int) -> State:
    """The six time steps of Litinski's one-level 15-to-1 protocol."""
    P = p / 3
    px, pz, pm = plog(p, dx), plog(p, dz), plog(p, dm)
    a = HALF * Decimal(dz) / dx * px * dm  # X storage of a check qubit
    b = HALF * Decimal(dx) / dz * pz * dm  # Z storage of a check qubit
    M = HALF * pm * dm
    single = (P + HALF * Decimal(dm) / dz * pz * dm, P + HALF * dz * pm, P)

    def multi(width: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        return (P + M, P + M + HALF * width * dx / dm * pm, P)

    s = State()
    zero = Decimal(0)
    # Step 1: rotations 1-3, 5 on the check qubits.
    s.rotation("1z111", *single)
    s.rotation("11z11", *single)
    s.rotation("111z1", *single)
    s.rotation("1zzz1", *multi(Decimal(3 * dz)))
    s.storage([zero, a, a, a, zero], [zero, b, b, b, zero])
    # Step 2: rotations 6-7; qubit 1 enters.
    s.rotation("zzz11", *multi(Decimal(dx + 2 * dz)))
    s.rotation("zz1z1", *multi(Decimal(dx + 3 * dz)))
    s.pauli_z(1, HALF * Decimal((dx + 2 * dz) + (dx + 3 * dz)) / dx * px * dm)
    s.storage([HALF * px * dm, a, a, a, zero], [HALF * px * dm, b, b, b, zero])
    # Step 3: rotations 8-9 and the single rotation on qubit 5.
    s.rotation("z1zz1", *multi(Decimal(dx + 3 * dz)))
    s.rotation("z11zz", *multi(Decimal(dx + 4 * dz)))
    s.rotation("1111z", *single)
    s.pauli_z(1, HALF * Decimal((dx + 3 * dz) + (dx + 4 * dz)) / dx * px * dm)
    s.storage([HALF * px * dm, a, a, a, a], [HALF * px * dm, b, b, b, b])
    # Step 4: rotations 10-11.
    s.rotation("zz11z", *multi(Decimal(dx + 4 * dz)))
    s.rotation("z1z1z", *multi(Decimal(dx + 4 * dz)))
    s.pauli_z(1, HALF * Decimal(2 * (dx + 4 * dz)) / dx * px * dm)
    s.storage([HALF * px * dm, a, a, a, a], [HALF * px * dm, b, b, b, b])
    # Step 5: rotations 12-13; the output is moved out (extra 2 d_X cycles).
    s.rotation("zzzzz", *multi(Decimal(dx + 4 * dz)))
    s.rotation("11zzz", *multi(Decimal(3 * dz)))
    s.pauli_z(1, HALF * Decimal(dx + 4 * dz) / dx * px * dm)
    out_store = HALF * px * (dm + 2 * dx)
    s.storage([out_store, a, a, a, a], [out_store, b, b, b, b])
    # Step 6: rotations 14-15 on the check qubits.
    s.rotation("1z1zz", *multi(Decimal(4 * dz)))
    s.rotation("1zz1z", *multi(Decimal(4 * dz)))
    s.storage([zero, a, a, a, a], [zero, b, b, b, b])
    return s


_IDEAL: list[tuple[Decimal, Decimal]] | None = None


def ideal_output() -> list[tuple[Decimal, Decimal]]:
    """Pure output state of the noiseless protocol, as a normalized vector."""
    global _IDEAL
    if _IDEAL is None:
        out, tr = run(Decimal(0), 3, 3, 3).output_after_checks()
        # rho = |v><v| with v = (1, e^{i phi})/sqrt 2; read e^{i phi} from rho[1][0].
        r10 = out[1][0]
        norm = out[0][0][0]
        _IDEAL = [(1 / Decimal(2).sqrt(), Decimal(0)), (r10[0] / norm / Decimal(2).sqrt(), r10[1] / norm / Decimal(2).sqrt())]
        assert abs(tr - ONE) < Decimal("1e-40"), "ideal protocol must always accept"
    return _IDEAL


def litinski(p: float, dx: int, dz: int, dm: int, *, logical_errors: bool = True) -> dict:
    """p_fail, p_out, qubits and code cycles of (15-to-1)_{dx,dz,dm} at physical rate p."""
    pp = Decimal(str(p))
    s = run(pp, dx, dz, dm) if logical_errors else _run_no_logical(pp)
    out, tr = s.output_after_checks()
    v = ideal_output()
    # <v|rho|v> for a 2x2 rho, conj(v_a) rho_ab v_b
    fid = Decimal(0)
    for a_ in range(2):
        for b_ in range(2):
            va, vb, r = v[a_], v[b_], out[a_][b_]
            # conj(va) * r * vb, real part
            t_re = va[0] * r[0] + va[1] * r[1]
            t_im = va[0] * r[1] - va[1] * r[0]
            fid += t_re * vb[0] - t_im * vb[1]
    pfail = ONE - tr
    pout = ONE - fid / tr
    return {
        "p_fail": float(pfail),
        "p_out": float(pout),
        "qubits": 2 * ((dx + 4 * dz) * 3 * dx + 2 * dm),
        "cycles": 6 * dm / float(1 - pfail),
        "cycles_per_attempt": 6 * dm,
    }


def _run_no_logical(p: Decimal) -> State:
    """Same circuit with faulty T measurements only (p_L = 0)."""
    P = p / 3
    s = State()
    for axis in ("1z111", "11z11", "111z1", "1zzz1", "zzz11", "zz1z1", "z1zz1", "z11zz",
                 "1111z", "zz11z", "z1z1z", "zzzzz", "11zzz", "1z1zz", "1zz1z"):
        s.rotation(axis, P, P, P)
    return s


# ---------------------------------------------------------------------------


# Table 1 of Litinski (2019), p_phys = 1e-4: (d_X, d_Z, d_m) -> (p_out, qubits, cycles)
TABLE1 = {(7, 3, 3): (4.4e-8, 810, 18.1), (9, 3, 3): (9.3e-10, 1150, 18.1), (11, 5, 5): (1.9e-11, 2070, 30.0)}


def validate(_: argparse.Namespace) -> int:
    print("Litinski (2019) Table 1, p_phys = 1e-4: published vs this reimplementation")
    print(f"{'(dX,dZ,dm)':12} {'p_out pub':>10} {'p_out ours':>11} {'qubits pub':>10} {'ours':>6} {'cycles pub':>10} {'ours':>7}")
    for (dx, dz, dm), (pout, qubits, cycles) in TABLE1.items():
        r = litinski(1e-4, dx, dz, dm)
        print(f"{str((dx, dz, dm)):12} {pout:10.2g} {r['p_out']:11.3g} {qubits:10d} {r['qubits']:6d} {cycles:10.1f} {r['cycles']:7.2f}")
    r = litinski(1e-4, 99, 99, 99, logical_errors=False)
    q = 2e-4 / 3
    print(f"\nfaulty-T errors only: p_out = {r['p_out']:.6g}, 35 q^3 with q = 2p/3: {35 * q**3:.6g}; p_fail = {r['p_fail']:.6g}")
    return 0


def compare(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(PACKAGE_DIR.parent))
    from ftqp_prototype.parser import parse_file  # noqa: E402
    from ftqp_prototype.resource.estimator import ResourceEstimator  # noqa: E402
    from ftqp_prototype.resource.estimator_distill import DistillationEstimator  # noqa: E402
    from ftqp_prototype.typechecker import TypeChecker  # noqa: E402

    rows = []
    p = 1e-4
    for d in args.distances:
        program = parse_file(EXAMPLES / f"15to1-d{d}.ft")
        symbols = TypeChecker().check(program)
        ours = DistillationEstimator(symbols).estimate_program(program)  # storage errors on
        (report,) = ours.protocols
        v1 = ResourceEstimator(symbols).estimate_program(program)
        assert v1.space == ours.space and abs(v1.time_cycles - ours.time_cycles) < 1e-6
        lit = litinski(p, d, d, d)
        lit_t = litinski(p, d, d, d, logical_errors=False)
        rows.append({
            "d": d,
            "ours_qubits": ours.space,
            "lit_qubits": lit["qubits"],
            "ours_cycles_attempt": ours.time_cycles,
            "ours_cycles_expected": ours.time_cycles / ours.acceptance,
            "lit_cycles_attempt": lit["cycles_per_attempt"],
            "lit_cycles_expected": lit["cycles"],
            "ours_acceptance": ours.acceptance,
            "lit_acceptance": 1 - lit["p_fail"],
            "ours_eps_acc": ours.accepted_output_error,
            "ours_eps_acc_no_storage": report.accepted_error_no_storage,
            "ours_acceptance_no_storage": report.acceptance_no_storage,
            "lit_p_out": lit["p_out"],
            "lit_p_out_faultyT_only": lit_t["p_out"],
            "ours_no_postselection": ours.disturbance_risk,
        })

    print("15-to-1, p = 1e-4, d = d_X = d_Z = d_m. Ours = FTQP estimator (§5.5 plug-in); Litinski = his 2019 model.\n")
    hdr = (f"{'d':>3} {'qubits ours':>11} {'Lit.':>7} {'cycles ours':>11} {'Lit.':>6} "
           f"{'A ours':>9} {'1-pfail Lit.':>12} {'eps_acc ours':>12} {'no storage':>11} "
           f"{'p_out Lit.':>11} {'Lit. T-only':>11}")
    print(hdr)
    for r in rows:
        print(f"{r['d']:>3} {r['ours_qubits']:>11} {r['lit_qubits']:>7} {r['ours_cycles_attempt']:>11.0f} "
              f"{r['lit_cycles_attempt']:>6} {r['ours_acceptance']:>9.6f} {r['lit_acceptance']:>12.6f} "
              f"{r['ours_eps_acc']:>12.4e} {r['ours_eps_acc_no_storage']:>11.4e} "
              f"{r['lit_p_out']:>11.4e} {r['lit_p_out_faultyT_only']:>11.4e}")
    print("\neps_acc ours includes storage errors (distill.py, default); 'no storage' is the plain")
    print("§5.5 recursion. Cycles are per attempt; expected cycles = per attempt / acceptance (CSV).")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "distill_compare.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        tex = args.output / "distill_compare_tables.tex"
        tex.write_text(pgf_tables(rows), encoding="utf-8")
        print(f"wrote {path} and {tex}")
    return 0


def pgf_tables(rows: list[dict]) -> str:
    lines = ["% 15-to-1 error comparison (scripts/litinski15to1.py compare), p = 1e-4",
             "\\pgfplotstableread{", "d ours oursNoStorage litinski litinskiT"]
    for r in rows:
        lines.append(f"{r['d']} {r['ours_eps_acc']:.6e} {r['ours_eps_acc_no_storage']:.6e} "
                     f"{r['lit_p_out']:.6e} {r['lit_p_out_faultyT_only']:.6e}")
    lines.append("}\\distillErrorTable")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="reproduce Table 1 of Litinski (2019)")
    cmp_ = sub.add_parser("compare", help="compare with our estimators on examples/distillation")
    cmp_.add_argument("--distances", type=lambda s: [int(x) for x in s.split(",")], default=[3, 5, 7, 9, 11, 21, 31, 41, 51, 61])
    cmp_.add_argument("--output", type=Path)
    args = parser.parse_args()
    return validate(args) if args.command == "validate" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
