#!/usr/bin/env python3
"""Bravyi-Haah (3k+8)-to-k family (Sec. 6.2, RQ2): our estimate vs the analytic error law.

Runs examples/distillation/bh/bh{2,4,6,8}.ft with the distillation estimator
(--semantics distill, storage errors on) and compares, for each k,

  * acceptance A                   with 1 - (3k+8) q
  * error of one output            with the exact value and (3k+1) q^2
  * error on at least one output   with the exact value and (4 + 3k(k-1)/2) q^2

where q = 2 pF / 3 is the probability that one rotation is faulty. "Exact" is
the recursion `exact()` of scripts/bravyi_haah.py; for k <= 4 it is also
checked against brute-force enumeration `brute()`. The pass criteria T1-T5 of
the task brief are checked, and the script exits with status 1 if one fails.

Comparison with G(k) of Bravyi and Haah (arXiv:1209.2426, Sec. VII, Eqs. 24-26)
------------------------------------------------------------------------------
`matrix` rebuilds G(k) from Eqs. (24)-(25) of the paper (blocks L, M, S1, S2)
and compares it with `columns(k)` of scripts/bravyi_haah.py. Result, for
k = 2, 4, 6, 8 (rerun with `bh_compare.py matrix`):

  * The output rows (G1) are identical up to column order: each output row has
    the 4 all-ones columns and 3 columns of its own, weight 7.
  * The check rows (G0) differ only by an invertible change of basis T of the
    3 check rows, again up to column order. Our per-output check parts
    {110, 101, 011} are the non-zero vectors of the even-weight subspace of
    F_2^3; the paper's {100, 010, 110} (from S2) are the non-zero vectors of
    span(100, 010). Any T mapping the first subspace onto the second also maps
    the remaining four check parts {100, 010, 001, 111} onto the paper's S1
    columns {001, 101, 011, 111}. T is printed by `matrix`.
  * A change of basis of the check rows keeps their row space, so the two
    matrices define the same code, the same acceptance event (all checks +1)
    and the same distribution of output flips. The script confirms this by
    running exact() on both column sets.
  * Weight enumerators, for both matrices: W_{G0}(x) = 1 + x^8 + 6 x^(4+2k)
    and W_{G0 + g_a}(x) = 1 + 2x^7 + x^8 + 6x^(3+2k) + 6x^(4+2k) for every
    output row g_a, as in Eq. (26) of the paper.

So bh{k}.ft implement the Bravyi-Haah protocol itself, not only a matrix with
the same parameters, and the family can be cited by name. The paper gives
P_s = 1 - (8+3k)p + ... and the per-output error (1+3k)p^2 + ... (Sec. VII)
but no closed form for the error on at least one output; (4 + 3k(k-1)/2) q^2
is counted from the matrix (undetected weight-2 faults) and confirmed by
brute force for k = 2, 4.

Usage:
  scripts/bh_compare.py matrix
  scripts/bh_compare.py compare [--ks 2,4,6,8] [--no-brute] [--output DIR]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
EXAMPLES = PACKAGE_DIR / "examples" / "distillation" / "bh"
sys.path.insert(0, str(SCRIPT_DIR))

from bravyi_haah import brute, columns, exact, tri  # noqa: E402

P_F = 1e-4
Q = 2 * P_F / 3  # the estimator's q_F

# Pass criteria of the task brief
TOL_ACCEPT_ABS = 1e-5  # T1
TOL_EXACT_REL = 0.01  # T2, T3: within 1% of exact
TOL_LEAD_REL = 0.002  # T2, T3: within 0.2% of leading order
TOL_SLOPE = 0.02  # T4: |slope - 3|
TOL_INTERCEPT = 0.1  # T4: |intercept - 1|


# ---------------------------------------------------------------------------
# Matrix comparison with Bravyi-Haah G(k)


def bh_columns(k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Columns (output bits, check bits) of G(k), Eqs. (24)-(25) of Bravyi-Haah."""
    if k % 2:
        raise ValueError("G(k) is defined for even k")
    s1 = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]  # columns of S1
    s2 = [(1, 0, 0), (0, 1, 0), (1, 1, 0)] * 2  # columns of S2
    cols = [((0,) * k, c) for c in s1]  # first block: 0 over S1
    cols += [((1,) * k, c) for c in s1]  # second block: L over S1
    for j in range(k // 2):  # six-column block j: M on output rows 2j, 2j+1
        for t, c in enumerate(s2):
            row = 2 * j + (t >= 3)
            cols.append((tuple(int(i == row) for i in range(k)), c))
    return cols


def rows(cols, k: int) -> tuple[list[list[int]], list[list[int]]]:
    g1 = [[o[r] for o, _ in cols] for r in range(k)]
    g0 = [[c[r] for _, c in cols] for r in range(3)]
    return g1, g0


def weight_enumerator(generators: list[list[int]]) -> Counter:
    n = len(generators[0])
    counts: Counter = Counter()
    for coeffs in itertools.product((0, 1), repeat=len(generators)):
        v = [0] * n
        for a, g in zip(coeffs, generators):
            if a:
                v = [x ^ y for x, y in zip(v, g)]
        counts[sum(v)] += 1
    return counts


def gl3():
    """All invertible 3x3 matrices over F_2, as functions on 3-tuples."""
    for entries in itertools.product((0, 1), repeat=9):
        m = [entries[0:3], entries[3:6], entries[6:9]]
        det = (m[0][0] * (m[1][1] * m[2][2] + m[1][2] * m[2][1])
               + m[0][1] * (m[1][0] * m[2][2] + m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] + m[1][1] * m[2][0])) % 2
        if det:
            yield m


def apply(m, c):
    return tuple(sum(m[r][j] * c[j] for j in range(3)) % 2 for r in range(3))


def check_matrix(k: int) -> tuple[bool, list[str]]:
    ours, paper = columns(k), bh_columns(k)
    lines: list[str] = []
    ok = True

    g_paper = [r for part in rows(paper, k) for r in part]
    w1 = [sum(r) for r in rows(paper, k)[0]]
    lines.append(f"  paper G({k}): {k + 3} x {len(paper)}, triorthogonal = {tri(g_paper)}, output row weights {sorted(set(w1))}")
    ok &= tri(g_paper) and set(w1) == {7}

    # Output part: same multiset of output columns
    same_g1 = Counter(o for o, _ in ours) == Counter(o for o, _ in paper)
    lines.append(f"  output rows equal up to column order: {same_g1}")
    ok &= same_g1

    # Check part: find T in GL(3, 2) with {(o, T c)} = paper's columns
    target = Counter(paper)
    found = next((m for m in gl3() if Counter((o, apply(m, c)) for o, c in ours) == target), None)
    if found is None:
        lines.append("  no change of basis of the check rows maps ours onto G(k)")
        ok = False
    else:
        lines.append(f"  check rows: ours = G(k) after T = {[list(r) for r in found]} and a column permutation")

    # Weight enumerators, Eq. (26)
    # Built term by term: for k = 2 the terms x^8 and 6x^(4+2k) coincide.
    expect0 = Counter([0, 8] + [4 + 2 * k] * 6)
    expect1 = Counter([0, 7, 7, 8] + [3 + 2 * k] * 6 + [4 + 2 * k] * 6)
    for name, cols in (("ours", ours), ("paper", paper)):
        g1, g0 = rows(cols, k)
        w0 = weight_enumerator(g0)
        w_a = [weight_enumerator(g0 + [g]) for g in g1]
        good = w0 == expect0 and all(w == expect1 for w in w_a)
        lines.append(f"  {name:5}: W_G0 = {fmt_poly(w0)}; W_G0+g_a = {fmt_poly(w_a[0])} (same for all a: {len({tuple(sorted(w.items())) for w in w_a}) == 1}) -> Eq. (26): {good}")
        ok &= good

    # Same fault distribution
    e_ours, e_paper = exact(ours, k, Q), exact(paper, k, Q)
    same = all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(e_ours, e_paper))
    lines.append(f"  exact(A, any, one output) identical: {same}")
    ok &= same
    return ok, lines


def fmt_poly(c: Counter) -> str:
    return " + ".join(("1" if w == 0 else (f"{n}x^{w}" if n > 1 else f"x^{w}")) for w, n in sorted(c.items()))


def matrix(args: argparse.Namespace) -> int:
    all_ok = True
    for k in args.ks:
        ok, lines = check_matrix(k)
        print(f"k = {k}: {'MATCHES' if ok else 'DIFFERS FROM'} G(k) of Bravyi-Haah")
        print("\n".join(lines))
        all_ok &= ok
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Estimator vs baselines


def leading(k: int, q: float) -> dict:
    return {"A": 1 - (3 * k + 8) * q, "one": (3 * k + 1) * q * q, "any": (4 + 3 * k * (k - 1) / 2) * q * q}


def run_estimator(k: int) -> dict:
    sys.path.insert(0, str(PACKAGE_DIR.parent))
    from ftqp_prototype.parser import parse_file  # noqa: E402
    from ftqp_prototype.resource.estimator_distill import DistillationEstimator  # noqa: E402
    from ftqp_prototype.typechecker import TypeChecker  # noqa: E402

    program = parse_file(EXAMPLES / f"bh{k}.ft")
    symbols = TypeChecker().check(program)
    est = DistillationEstimator(symbols).estimate_program(program)  # storage errors on
    (report,) = est.protocols
    if len(report.outputs) != k or len(report.checks) != 3:
        raise SystemExit(f"bh{k}.ft: estimator found {len(report.outputs)} outputs and {len(report.checks)} checks")
    return {
        "A": est.acceptance,
        "any": est.accepted_output_error,
        "one": max(report.output_errors),
        "one_spread": max(report.output_errors) - min(report.output_errors),
        "A_no_storage": report.acceptance_no_storage,
        "any_no_storage": report.accepted_error_no_storage,
        "rotations": report.rotations,
        "q": report.rotation_error,
        "cycles": est.time_cycles,
        "qubits": est.space,
    }


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def compare(args: argparse.Namespace) -> int:
    rows_: list[dict] = []
    for k in args.ks:
        ours = run_estimator(k)
        cols = columns(k)
        A, any_, one = exact(cols, k, Q)
        row = {"k": k, "n": len(cols), "ours": ours,
               "exact": {"A": A, "any": any_, "one": one}, "lead": leading(k, Q)}
        if not args.no_brute and k <= 4:
            bA, bany, bone = brute(cols, k, Q)
            row["brute"] = {"A": bA, "any": bany, "one": bone}
        rows_.append(row)

    print(f"Bravyi-Haah (3k+8)-to-k, pF = {P_F:g}, q = 2pF/3 = {Q:.4e}, d = 21. ours / exact / leading order\n")
    print(f"{'k':>2} {'n':>3} {'A':>30} {'one output':>36} {'at least one output':>36} {'cycles':>7} {'qubits':>7}")
    for r in rows_:
        o, e, l = r["ours"], r["exact"], r["lead"]
        print(f"{r['k']:>2} {r['n']:>3} {o['A']:.6f} / {e['A']:.6f} / {l['A']:.6f}"
              f"  {o['one']:.4e} / {e['one']:.4e} / {l['one']:.4e}"
              f"  {o['any']:.4e} / {e['any']:.4e} / {l['any']:.4e}"
              f"  {o['cycles']:>7.0f} {o['qubits']:>7}")
        if "brute" in r:
            b = r["brute"]
            print(f"{'':>6} brute force: A = {b['A']:.6f}, one = {b['one']:.4e}, any = {b['any']:.4e}")

    failures = check_targets(rows_)
    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  " + f)
    else:
        print("T1-T5 hold for all k.")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        tex = args.output / "bh_compare.tex"
        tex.write_text(latex_table(rows_), encoding="utf-8")
        print(f"wrote {tex}")
    return 1 if failures else 0


def check_targets(rows_: list[dict]) -> list[str]:
    fail: list[str] = []
    for r in rows_:
        k, o, e, l = r["k"], r["ours"], r["exact"], r["lead"]
        if r["n"] != 3 * k + 8 or o["rotations"] != 3 * k + 8:
            fail.append(f"k={k}: {o['rotations']} rotations, expected {3 * k + 8}")
        if not math.isclose(o["q"], Q, rel_tol=1e-6):
            fail.append(f"k={k}: estimator's q_j = {o['q']:.6e}, expected 2pF/3 = {Q:.6e}")
        if o["one_spread"] > 1e-6 * o["one"]:
            fail.append(f"k={k}: outputs have different marginal errors (spread {o['one_spread']:.2e})")
        # T1
        if abs(o["A"] - l["A"]) >= TOL_ACCEPT_ABS:
            fail.append(f"T1 k={k}: A = {o['A']:.6f} vs 1-(3k+8)q = {l['A']:.6f}")
        # T2, T3
        for key, name in (("one", "T2"), ("any", "T3")):
            if rel(o[key], e[key]) >= TOL_EXACT_REL:
                fail.append(f"{name} k={k}: {o[key]:.4e} vs exact {e[key]:.4e} ({100 * rel(o[key], e[key]):.3f}%)")
            if rel(o[key], l[key]) >= TOL_LEAD_REL:
                fail.append(f"{name} k={k}: {o[key]:.4e} vs leading {l[key]:.4e} ({100 * rel(o[key], l[key]):.3f}%)")
        if "brute" in r:
            for key in ("A", "one", "any"):
                if not math.isclose(r["brute"][key], e[key], rel_tol=1e-9):
                    fail.append(f"k={k}: exact {key} = {e[key]:.6e} disagrees with brute force {r['brute'][key]:.6e}")
    # T4: per-output coefficient c_k = eps_1 / q^2 against k
    ks = [r["k"] for r in rows_]
    cs = [r["ours"]["one"] / Q**2 for r in rows_]
    if len(ks) >= 2:
        kbar, cbar = sum(ks) / len(ks), sum(cs) / len(cs)
        slope = sum((k - kbar) * (c - cbar) for k, c in zip(ks, cs)) / sum((k - kbar) ** 2 for k in ks)
        intercept = cbar - slope * kbar
        print(f"\nT4: eps_1 / q^2 = {', '.join(f'{c:.3f}' for c in cs)}; fit slope {slope:.4f}, intercept {intercept:.4f}")
        if abs(slope - 3) >= TOL_SLOPE or abs(intercept - 1) >= TOL_INTERCEPT:
            fail.append(f"T4: slope {slope:.4f}, intercept {intercept:.4f} (expected 3, 1)")
    # T5: joint error well below the union bound k * eps_1 for k >= 4
    ratios = []
    for r in rows_:
        k, o = r["k"], r["ours"]
        ratio = o["any"] / (k * o["one"])
        expected = (4 + 3 * k * (k - 1) / 2) / (k * (3 * k + 1))
        ratios.append(f"k={k}: {ratio:.3f} (leading order {expected:.3f})")
        if k >= 4 and not ratio < 1:
            fail.append(f"T5 k={k}: ratio {ratio:.3f} is not below 1")
        if abs(ratio - expected) > 0.01:
            fail.append(f"T5 k={k}: ratio {ratio:.3f} differs from leading order {expected:.3f}")
    print("T5: eps_any / (k eps_1): " + "; ".join(ratios))
    return fail


# ---------------------------------------------------------------------------
# LaTeX draft table (booktabs)


def latex_table(rows_: list[dict]) -> str:
    """Errors in units of q^2, so the leading-order coefficients 3k+1 and
    4+3k(k-1)/2 can be read off directly (the three-values-per-cell layout with
    absolute errors is 200pt too wide for acmsmall)."""
    q2 = Q * Q
    m, e = f"{q2:.2e}".split("e")
    lines = [
        "% Generated by scripts/bh_compare.py compare --output DIR. Draft for Sec. 6.2; not yet in the paper.",
        "% Needs booktabs. Errors are divided by q^2, q = 2 p_F / 3.",
        "\\begin{table}[t]",
        "  \\centering\\small",
        "  \\caption{The Bravyi--Haah \\((3k{+}8)\\)-to-\\(k\\) protocols at \\(d=21\\) and \\(p=p_F=10^{-4}\\).",
        "    Errors are given in units of \\(q^2\\), where \\(q=2p_F/3\\approx 6.67\\times10^{-5}\\) is the probability that a",
        f"    rotation is faulty (\\(q^2\\approx {m}\\times10^{{{int(e)}}}\\)). \\emph{{Ours}}: the estimator on the program text;",
        "    \\emph{exact}: independent rotation faults, all orders; the last column of each group is the leading-order term.}",
        "  \\label{tab:distill-bh}",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{r cc ccc ccc rr}",
        "    \\toprule",
        "    & \\multicolumn{2}{c}{Acceptance \\(A\\)} & \\multicolumn{3}{c}{Error of one output \\(/q^2\\)}",
        "      & \\multicolumn{3}{c}{Error of any output \\(/q^2\\)} & & \\\\",
        "    \\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){7-9}",
        "    \\(k\\) & ours & \\(1-(3k{+}8)q\\) & ours & exact & \\(3k{+}1\\)",
        "      & ours & exact & \\(4+\\frac{3k(k-1)}{2}\\) & Cycles & Qubits \\\\",
        "    \\midrule",
    ]
    for r in rows_:
        k, o, ex, l = r["k"], r["ours"], r["exact"], r["lead"]
        cells = [
            str(k),
            f"{o['A']:.6f}", f"{l['A']:.6f}",
            f"{o['one'] / q2:.3f}", f"{ex['one'] / q2:.3f}", f"{3 * k + 1}",
            f"{o['any'] / q2:.3f}", f"{ex['any'] / q2:.3f}", f"{4 + 3 * k * (k - 1) // 2}",
            f"{round(o['cycles']):,}".replace(",", "\\,"), f"{o['qubits']:,}".replace(",", "\\,"),
        ]
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ks = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    m = sub.add_parser("matrix", help="compare the generator's matrices with G(k) of Bravyi-Haah")
    m.add_argument("--ks", type=ks, default=[2, 4, 6, 8])
    c = sub.add_parser("compare", help="run the estimator on bh<k>.ft and check T1-T5")
    c.add_argument("--ks", type=ks, default=[2, 4, 6, 8])
    c.add_argument("--no-brute", action="store_true", help="skip brute-force enumeration for k <= 4")
    c.add_argument("--output", type=Path)
    args = parser.parse_args()
    return matrix(args) if args.command == "matrix" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
