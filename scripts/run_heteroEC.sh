#!/usr/bin/env bash
# Heterogeneous error correction: best error rate under a space budget when
# codeblocks may use different code distances (Table 1 and Table 3 of
# arXiv-2608.04573). Mean decoding time in both.
#
# Table 1 (Sec. 2): naive logical T gate of Fig. 5 (no trailing Hadamards),
#   data and ancilla distances (d_data, d_anc) in {27, 29, 31}^2 (9 runs),
#   budgets < 4500, 5000, 5500, 6000 qubits.
#   Sweep: scripts/heteroec.py table1-sweep (new; no generator existed before).
#
# Table 3 (Sec. 6.1): auto-corrected logical T gate followed by 36 Hadamards,
#   (d_q, d_anc[0], d_anc[1]) in {23, 25, 27, 29, 31}^3 (125 runs),
#   budgets < 5000, 5500, 6000, 6500, 7000, 7500 qubits.
#   Sweep: examples/logicalT-exp/hybrid/collect_mean_autocorrected.py.
#
# "Optimal" = lowest error of q among configurations whose total physical
# qubit count is strictly below the budget.
#
# Usage: scripts/run_heteroEC.sh [--semantics v1|v2|both] [--out DIR] [--mean-dir DIR] [--table-format markdown|latex]
#   v1 reproduces the numbers of the first paper version; v2 is the
#   source-aware error analysis of the revised Sec. 5 (default).
# Override budgets with TABLE1_BUDGETS / TABLE3_BUDGETS (comma-separated).
set -euo pipefail

SEMANTICS="v2"
OUT=""
MEAN_DIR=""
TABLE_FORMAT="markdown"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --semantics) SEMANTICS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --mean-dir) MEAN_DIR="$2"; shift 2 ;;
    --table-format) TABLE_FORMAT="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$PKG_DIR/results/heteroEC}"
MEAN_DIR="${MEAN_DIR:-$PKG_DIR/examples/logicalT-exp/mean}"
case "$TABLE_FORMAT" in
  markdown) TABLE_EXT=md ;;
  latex) TABLE_EXT=tex ;;
  *) echo "unsupported table format" >&2; exit 2 ;;
esac
TABLE1_BUDGETS="${TABLE1_BUDGETS:-4500,5000,5500,6000}"
TABLE3_BUDGETS="${TABLE3_BUDGETS:-5000,5500,6000,6500,7000,7500}"

case "$SEMANTICS" in
  v1|v2) SEMANTICS_LIST=("$SEMANTICS") ;;
  both) SEMANTICS_LIST=(v1 v2) ;;
  *) echo "--semantics must be v1, v2 or both" >&2; exit 2 ;;
esac

for sem in "${SEMANTICS_LIST[@]}"; do
  dir="$OUT/$sem"
  mkdir -p "$dir"

  echo "== Table 1: naive logical T, (d_data, d_anc) sweep, error semantics $sem"
  "$PYTHON" "$SCRIPT_DIR/heteroec.py" table1-sweep --semantics "$sem" \
    --distances 27,29,31 --mean-dir "$MEAN_DIR" --artifacts-dir "$dir/raw/table1" --output "$dir/table1_sweep.csv"
  echo
  "$PYTHON" "$SCRIPT_DIR/heteroec.py" optimal --csv "$dir/table1_sweep.csv" \
    --budgets "$TABLE1_BUDGETS" --style table1 --format "$TABLE_FORMAT" --output "$dir/table1.$TABLE_EXT"
  echo

  echo "== Table 3: auto-corrected logical T + 36 H, (d_q, d_anc[0], d_anc[1]) sweep, error semantics $sem"
  echo "   125 runs, about a minute"
  "$PYTHON" "$PKG_DIR/examples/logicalT-exp/hybrid/collect_mean_autocorrected.py" \
    --semantics "$sem" --mean-dir "$MEAN_DIR" --artifacts-dir "$dir/raw/table3" --output "$dir/table3_sweep.csv"
  echo
  "$PYTHON" "$SCRIPT_DIR/heteroec.py" optimal --csv "$dir/table3_sweep.csv" \
    --budgets "$TABLE3_BUDGETS" --style table3 --format "$TABLE_FORMAT" --output "$dir/table3.$TABLE_EXT"
  echo
done
echo "Results in $OUT"
