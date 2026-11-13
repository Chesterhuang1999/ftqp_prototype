#!/usr/bin/env bash
# 15-to-1 distillation (Sec. 6.2, Fig. 15): our estimate vs Litinski (2019).
#
# Runs examples/distillation/15to1-d{3,...,11,21,...,61}.ft with the distillation
# estimator (time, space, acceptance A, accepted error eps_acc with and without
# storage errors), and Litinski's
# one-level 15-to-1 model with d_X = d_Z = d_m = d at p = 1e-4 (qubits, cycles,
# acceptance 1 - p_fail, output error p_out). The paper's comparison used only
# Litinski's closed-form space/time formulas; scripts/litinski15to1.py is an
# independent reimplementation of his error model (github.com/litinski/magicstates)
# and first checks it against Table 1 of his paper.
#
# Usage: scripts/run_distill_compare.sh [--out DIR]   (default results/distillation)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUT="$PKG_DIR/results/distillation"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
echo "== Validation of the Litinski model"
"$PYTHON" "$SCRIPT_DIR/litinski15to1.py" validate
echo
echo "== Storage-error recursion on Litinski's schedule"
"$PYTHON" "$SCRIPT_DIR/distill_storage.py" validate
echo
echo "== Comparison"
"$PYTHON" "$SCRIPT_DIR/litinski15to1.py" compare --output "$OUT"
