#!/usr/bin/env bash
# Render Fig. 13 (logical T + 36 H; Fig. 14 in the first paper version) as a
# standalone PDF/PNG with pgfplots, from the tables written by run_figure14.sh,
# and write a drop-in replacement for the paper's contents/exp.tex.
#
# The plotting code is the paper's own, with the bottom row chosen by --row2:
#   latency     scripts/figure13_template_latency.tex (default): top row unchanged,
#               bottom row = error rate and time at d = 27 versus t_dec
#   percentile  scripts/figure13_template.tex (copied from contents/exp.tex):
#               bottom row = d-sweep under the 99.9th-percentile latency
# Only the data tables come from the results file. Page width and fonts follow
# the paper (acmart acmsmall, Libertine).
#
# Usage: scripts/plot_figure13.sh [--semantics v1|v2|both] [--row2 latency|percentile]
#                                 [--results DIR] [--tex-only]
#   Reads  DIR/<semantics>/exp_tables.tex  (default DIR: results/figure14)
#   Writes DIR/<semantics>/exp_figure.tex  (figure environment for contents/exp.tex)
#          DIR/<semantics>/figure13.pdf, figure13.png, figure13.tex (not with --tex-only)
# Needs pdflatex (TeX Live with pgfplots) unless --tex-only; pdfcrop and pdftoppm are optional.
set -euo pipefail

SEMANTICS="v2"
ROW2="latency"
RESULTS=""
TEX_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --semantics) SEMANTICS="$2"; shift 2 ;;
    --row2) ROW2="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --tex-only) TEX_ONLY=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS="${RESULTS:-$PKG_DIR/results/figure14}"

case "$ROW2" in
  latency)
    TEMPLATE="$SCRIPT_DIR/figure13_template_latency.tex"
    CAPTION='    \caption{Resource estimation for a logical \(T\) gate followed by 36 logical
    Hadamard gates, comparing the naive and auto-corrected implementations of
    Figs.~\ref{fig:logicalT} and~\ref{fig:logicalT-auto}.
    Top: error rate, space and time across code distances under the mean
    decoding latency, adapted from Fig.~2(b) of~\cite{delfosse2023choosedecoderfaulttolerantquantum}.
    Bottom: error rate and time at \(d = 27\) as functions of the decoding latency.}'
    ;;
  percentile)
    TEMPLATE="$SCRIPT_DIR/figure13_template.tex"
    CAPTION='    \caption{Resource estimation for a logical \(T\) gate followed by 36 logical
    Hadamard gates across different code distances, comparing the naive and
    auto-corrected implementations shown in Figs.~\ref{fig:logicalT}
    and~\ref{fig:logicalT-auto}, respectively.
    The two decoding-latency plots in the first column are adapted from Fig.~2(b) of~\cite{delfosse2023choosedecoderfaulttolerantquantum}.}'
    ;;
  *) echo "--row2 must be latency or percentile" >&2; exit 2 ;;
esac

# Optional custom template and caption suppression for measured datasets.
TEMPLATE="${FIGURE14_TEMPLATE:-$TEMPLATE}"
if [[ "${FIGURE14_OMIT_CAPTION:-0}" == 1 ]]; then CAPTION=""; fi

case "$SEMANTICS" in
  v1|v2) SEMANTICS_LIST=("$SEMANTICS") ;;
  both) SEMANTICS_LIST=(v1 v2) ;;
  *) echo "--semantics must be v1, v2 or both" >&2; exit 2 ;;
esac

if [[ $TEX_ONLY -eq 0 ]]; then
  command -v pdflatex >/dev/null || { echo "pdflatex not found; install TeX Live with pgfplots" >&2; exit 1; }
fi

for sem in "${SEMANTICS_LIST[@]}"; do
  data="$RESULTS/$sem/exp_tables.tex"
  if [[ ! -f "$data" ]]; then
    echo "missing $data; run scripts/run_figure14.sh --semantics $sem --row2 $ROW2 first" >&2
    exit 1
  fi

  # Drop-in replacement for contents/exp.tex: the paper's figure wrapper, the
  # tables, the plotting code (without the template's header comment), caption, label.
  {
    printf '%s\n' '\begin{figure}[t]' '    \centering' '    \vspace{-1em}' ''
    cat "$data"
    echo
    awk 'body || !/^%/ { body = 1; print }' "$TEMPLATE"
    echo
    printf '%s\n' "$CAPTION" '    \label{fig:resource-result}' '  \end{figure}'
  } > "$RESULTS/$sem/exp_figure.tex"
  echo "wrote $RESULTS/$sem/exp_figure.tex"
  [[ $TEX_ONLY -eq 1 ]] && continue

  build="$(mktemp -d "${TMPDIR:-/tmp}/figure13.XXXXXX")"
  {
    cat <<'EOF'
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
% Text width of acmart acmsmall, so 0.27\textwidth matches the paper.
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=396pt,margin=0.5in]{geometry}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\centering
EOF
    cat "$data"
    echo
    cat "$TEMPLATE"
    echo
    echo '\end{document}'
  } > "$build/figure13.tex"

  if ! (cd "$build" && pdflatex -interaction=nonstopmode -halt-on-error figure13.tex > build.log 2>&1); then
    echo "pdflatex failed; log: $build/build.log" >&2
    tail -20 "$build/build.log" >&2
    exit 1
  fi
  pdf="$build/figure13.pdf"
  if command -v pdfcrop >/dev/null; then
    pdfcrop --margins 4 "$pdf" "$build/figure13-crop.pdf" > /dev/null
    pdf="$build/figure13-crop.pdf"
  fi
  cp "$pdf" "$RESULTS/$sem/figure13.pdf"
  cp "$build/figure13.tex" "$RESULTS/$sem/figure13.tex"
  if command -v pdftoppm >/dev/null; then
    pdftoppm -png -r 200 -singlefile "$RESULTS/$sem/figure13.pdf" "$RESULTS/$sem/figure13"
  fi
  rm -rf "$build"
  echo "wrote $RESULTS/$sem/figure13.pdf"
done
