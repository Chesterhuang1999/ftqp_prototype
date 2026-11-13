#!/usr/bin/env bash
# Bravyi-Haah (3k+8)-to-k distillation (Sec. 6.2, RQ2): our estimate vs the analytic error law.
#
# Checks that examples/distillation/bh/bh{2,4,6,8}.ft encode G(k) of Bravyi and
# Haah (arXiv:1209.2426, Eqs. 24-26), then runs them with the distillation
# estimator (--semantics distill, storage errors on, d = 21, p = pF = 1e-4) and
# compares acceptance, the error of one output and the error on at least one
# output with the exact values and the leading-order formulas, with
# q = 2 pF / 3. Fails if the matrix check or any pass criterion T1-T5 fails.
# Writes a draft LaTeX table bh_compare.tex (no CSV).
#
# Usage: scripts/run_bh_compare.sh [--out DIR] [--no-brute]   (default results/distillation)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUT="$PKG_DIR/results/distillation"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --no-brute) EXTRA+=(--no-brute); shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
echo "== Matrices vs G(k) of Bravyi-Haah"
"$PYTHON" "$SCRIPT_DIR/bh_compare.py" matrix
echo
echo "== Estimator vs baselines"
"$PYTHON" "$SCRIPT_DIR/bh_compare.py" compare --output "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
