#!/usr/bin/env python3
"""Prototype of the storage-error recursion (15-to-1), now in resource/distill.py.

Kept as an independent check: `validate` runs the recursion on Litinski's own
schedule against his density matrix; `ours` writes our programs' schedule out
by hand (tests/test_estimator_v2.py compares the estimator with it).

The §5.5 recursion tracks the flip pattern (e, c) of outputs and checks under
independent rotation errors P_j. Storage errors are added as further events:

* Z storage on block Q (prob z): Z commutes with every Z-type rotation, so it
  only flips Q's check bit or output bit. Exact.
* X storage on block Q at time t (prob x): push X_Q to the nearer end of the
  protocol. Every rotation P_j ∋ Q it crosses becomes P_{j,-π/8}, i.e. it gets
  the error P_{j,-π/4}, which after twirling flips (h_j, l_j) with prob 1/2.
  At the start X_Q is harmless (X|+> = |+>); at the end it is harmless on a check
  (measured in X) and randomizes the output bit (|<m|X|m>|^2 = 1/2). Exact when
  no rotation is crossed.
* Storage of a rotation's own ancilla before its faulty T measurement (prob a):
  Z gives P_j, X gives P_{j,-π/4}; the rotation flips with 3a/4 more.

Storage risk of a block over n QEC rounds: r = 1 - (1 - p_L)^n, split half X,
half Z (as in Litinski 2019).

Usage:
  scripts/distill_storage.py validate     # vs Litinski's density matrix
  scripts/distill_storage.py ours         # our 15-to-1 programs, d = 3..11
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litinski15to1 as lit  # noqa: E402

# Qubit 0 = output, 1..4 = checks (Litinski's qubits 1..5). Bit layout of a state
# index: bit 4 = output error e, bits 0..3 = check flips c.
OUT_BIT = 1 << 4


def qbit(q: int) -> int:
    return OUT_BIT if q == 0 else 1 << (q - 1)


def flips(axis: str) -> int:
    """(h_j, l_j) of a Z-type axis over 5 qubits, e.g. 'z1zz1'."""
    return sum(qbit(i) for i, c in enumerate(axis) if c == "z")


@dataclass
class Schedule:
    """Rotations in time order, and storage events placed between them."""

    axes: list[str] = field(default_factory=list)
    q: list[float] = field(default_factory=list)  # flip prob of each rotation
    q_base: list[float] = field(default_factory=list)  # same, without storage terms
    # (position, qubit, pauli, prob): after `position` rotations have happened
    storage: list[tuple[int, int, str, float]] = field(default_factory=list)

    def rot(self, axis: str, q: float, q_base: float | None = None) -> None:
        self.axes.append(axis)
        self.q.append(q)
        self.q_base.append(q if q_base is None else q_base)

    def store(self, qubit: int, pauli: str, p: float) -> None:
        if p > 0:
            self.storage.append((len(self.axes), qubit, pauli, p))


def evaluate(s: Schedule, *, storage: bool = True) -> tuple[float, float]:
    """Extended recursion; returns (A, eps_acc)."""
    mu = [0.0] * 32
    mu[0] = 1.0

    def flip(mu: list[float], f: int, p: float) -> list[float]:
        return [(1 - p) * mu[i] + p * mu[i ^ f] for i in range(32)]

    def x_event(mu: list[float], pos: int, qubit: int, p: float) -> list[float]:
        before = [j for j in range(pos) if s.axes[j][qubit] == "z"]
        after = [j for j in range(pos, len(s.axes)) if s.axes[j][qubit] == "z"]
        crossed, end_bits = (before, 0) if len(before) < len(after) else (after, OUT_BIT if qubit == 0 else 0)
        t = mu
        for j in crossed:
            t = flip(t, flips(s.axes[j]), 0.5)
        if end_bits:
            t = flip(t, end_bits, 0.5)
        return [(1 - p) * a + p * b for a, b in zip(mu, t)]

    events = sorted(s.storage, key=lambda e: e[0]) if storage else []
    k = 0
    for j in range(len(s.axes) + 1):
        while k < len(events) and events[k][0] == j:
            _, qubit, pauli, p = events[k]
            mu = flip(mu, qbit(qubit), p) if pauli == "Z" else x_event(mu, j, qubit, p)
            k += 1
        if j < len(s.axes):
            mu = flip(mu, flips(s.axes[j]), s.q[j] if storage else s.q_base[j])
    acc = sum(mu[i] for i in range(32) if i & 0xF == 0)
    return acc, mu[OUT_BIT] / acc


def xor(*ps: float) -> float:
    """Flip probability of independent flips."""
    r = 0.0
    for p in ps:
        r = r * (1 - p) + (1 - r) * p
    return r


# ---------------------------------------------------------------------------
# Litinski's schedule (same terms as litinski15to1.run), for validation
# ---------------------------------------------------------------------------


def litinski_schedule(p: float, dx: int, dz: int, dm: int) -> Schedule:
    P = p / 3
    pl = lambda d: 0.1 * (100 * p) ** ((d + 1) / 2)  # noqa: E731
    px, pz, pm = pl(dx), pl(dz), pl(dm)
    a = 0.5 * dz / dx * px * dm
    b = 0.5 * dx / dz * pz * dm
    M = 0.5 * pm * dm
    s = Schedule()

    def rot(axis: str, p1: float, p2: float) -> None:
        # p1: Pauli error (flip); p2: -pi/8 and P: +pi/4 (twirled: flip 1/2)
        s.rot(axis.replace("1", "-").replace("-", "1"), p1 + (p2 + P) / 2)

    single = (P + 0.5 * dm / dz * pz * dm, P + 0.5 * dz * pm)
    multi = lambda w: (P + M, P + M + 0.5 * w * dx / dm * pm)  # noqa: E731

    def storage(xs: list[float], zs: list[float]) -> None:
        for q in range(5):
            s.store(q, "X", xs[q])
            s.store(q, "Z", zs[q])

    rot("1z111", *single); rot("11z11", *single); rot("111z1", *single)  # noqa: E702
    rot("1zzz1", *multi(3 * dz))
    storage([0, a, a, a, 0], [0, b, b, b, 0])
    rot("zzz11", *multi(dx + 2 * dz)); rot("zz1z1", *multi(dx + 3 * dz))  # noqa: E702
    s.store(0, "Z", 0.5 * ((dx + 2 * dz) + (dx + 3 * dz)) / dx * px * dm)
    storage([0.5 * px * dm, a, a, a, 0], [0.5 * px * dm, b, b, b, 0])
    rot("z1zz1", *multi(dx + 3 * dz)); rot("z11zz", *multi(dx + 4 * dz)); rot("1111z", *single)  # noqa: E702
    s.store(0, "Z", 0.5 * ((dx + 3 * dz) + (dx + 4 * dz)) / dx * px * dm)
    storage([0.5 * px * dm] + [a] * 4, [0.5 * px * dm] + [b] * 4)
    rot("zz11z", *multi(dx + 4 * dz)); rot("z1z1z", *multi(dx + 4 * dz))  # noqa: E702
    s.store(0, "Z", 0.5 * 2 * (dx + 4 * dz) / dx * px * dm)
    storage([0.5 * px * dm] + [a] * 4, [0.5 * px * dm] + [b] * 4)
    rot("zzzzz", *multi(dx + 4 * dz)); rot("11zzz", *multi(3 * dz))  # noqa: E702
    s.store(0, "Z", 0.5 * (dx + 4 * dz) / dx * px * dm)
    o = 0.5 * px * (dm + 2 * dx)
    storage([o] + [a] * 4, [o] + [b] * 4)
    rot("1z1zz", *multi(4 * dz)); rot("1zz1z", *multi(4 * dz))  # noqa: E702
    storage([0] + [a] * 4, [0] + [b] * 4)
    return s


def validate(_: argparse.Namespace) -> int:
    print("Extended recursion on Litinski's schedule vs his 5-qubit density matrix")
    print(f"{'p':>6} {'(dX,dZ,dm)':>12} {'p_out DM':>11} {'recursion':>11} {'ratio':>6} {'1-A DM':>10} {'1-A rec':>10} {'rec, no storage':>16}")
    configs = [(1e-4, 7, 3, 3), (1e-4, 9, 3, 3), (1e-4, 11, 5, 5), (1e-3, 17, 7, 7)]
    configs += [(1e-4, d, d, d) for d in (3, 5, 7, 9, 11, 21)]
    for p, dx, dz, dm in configs:
        dm_ = lit.litinski(p, dx, dz, dm)
        sch = litinski_schedule(p, dx, dz, dm)
        acc, eps = evaluate(sch)
        _, eps0 = evaluate(sch, storage=False)
        print(f"{p:6.0e} {str((dx, dz, dm)):>12} {dm_['p_out']:11.3e} {eps:11.3e} {eps / dm_['p_out']:6.2f} "
              f"{dm_['p_fail']:10.3e} {1 - acc:10.3e} {eps0:16.3e}")
    return 0


# ---------------------------------------------------------------------------
# Our program: examples/distillation/15to1-d*.ft, schedule of the v2 estimator
# ---------------------------------------------------------------------------

# Hardware of the examples: fixed tSE = 1 us; times below in SE rounds.
T1, T2, TM = 1e-8, 1e-7, 1e-6
TSE = 1e-6
EPS1 = 0.0  # t1 is absorbed into the first SE round
T_FAULTY = (2 * T1 + 2 * TM) / TSE

# Rotations in program order (qubit 0 = q, 1..4 = anc[0..3]); the first four run in parallel.
OUR_AXES = ["1z111", "11z11", "111z1", "1111z",
            "1zzz1", "zzz11", "zz1z1", "z1zz1", "z11zz", "zz11z", "z1z1z", "zzzzz",
            "11zzz", "1z1zz", "1zz1z"]


def our_schedule(p: float, d: int) -> tuple[Schedule, float]:
    """Storage events from the program's timeline (52d+24.24 rounds)."""
    pL = 0.1 * (p / 0.01) ** ((d + 1) / 2)
    r = lambda n: 1 - (1 - pL) ** n  # noqa: E731
    p_f = p
    q_f = 2 * p_f / 3
    q_m = -math.expm1(2 * d * math.log1p(-pL))  # Eq. (njoint)
    gadget = 2 * d + (d + EPS1) + T_FAULTY + (d + EPS1)  # M(PZ), X[anc], M(faultyT), P[q]
    anc_rounds = 2 * d + (d + EPS1)  # rotation ancilla alive before its faulty T measurement
    q_rot = xor(q_f, q_m / 2, 0.75 * r(anc_rounds))
    q_rot0 = xor(q_f, q_m / 2)  # Eq. (distill-q)

    start = 2 * (d + EPS1)  # prepare || ..., H || ...
    slots = [OUR_AXES[:4]] + [[a] for a in OUR_AXES[4:]]
    # The rotation acts when its joint measurement ends.
    times = [start + i * gadget + 2 * d for i in range(len(slots))]
    total = start + len(slots) * gadget + 2 * d  # + check measurements M(X)
    s = Schedule()
    prev = 0.0
    for t, slot in zip(times, slots):
        for q in range(5):
            s.store(q, "X", r(t - prev) / 2)
            s.store(q, "Z", r(t - prev) / 2)
        for axis in slot:
            s.rot(axis, q_rot, q_rot0)
        prev = t
    for q in range(5):
        s.store(q, "X", r(total - prev) / 2)
        s.store(q, "Z", r(total - prev) / 2)
    return s, total


def ours(args: argparse.Namespace) -> int:
    p = 1e-4
    print("Our 15-to-1 programs, p = 1e-4: §5.5 recursion without and with storage errors,")
    print("and Litinski's model at the same uniform distance d = dX = dZ = dm.")
    print(f"{'d':>3} {'cycles':>7} {'eps_acc §5.5':>13} {'+ storage':>11} {'1-A':>10} {'Litinski p_out':>15} {'Lit. 1-A':>10}")
    for d in args.distances:
        sch, total = our_schedule(p, d)
        _, eps0 = evaluate(sch, storage=False)
        acc, eps = evaluate(sch)
        L = lit.litinski(p, d, d, d)
        print(f"{d:>3} {total:7.1f} {eps0:13.3e} {eps:11.3e} {1 - acc:10.3e} {L['p_out']:15.3e} {L['p_fail']:10.3e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    o = sub.add_parser("ours")
    o.add_argument("--distances", type=lambda v: [int(x) for x in v.split(",")], default=[3, 5, 7, 9, 11, 21])
    args = parser.parse_args()
    return validate(args) if args.command == "validate" else ours(args)


if __name__ == "__main__":
    raise SystemExit(main())
