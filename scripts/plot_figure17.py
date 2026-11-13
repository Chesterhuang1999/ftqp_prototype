#!/usr/bin/env python3
"""Fig. 17 (15-to-1: ours vs Litinski) and the layout of Figs. 17/18.

Writes a drop-in replacement for contents/compare.tex of arXiv-2608.04573:

* Fig. 17 (left minipage) is regenerated from results/distillation/
  distill_compare.csv (scripts/run_distill_compare.sh): space, time and output
  error for the chosen distances. Our error includes storage errors (the
  default of the distillation estimator); Litinski's comes from his model
  (scripts/litinski15to1.py).
* Fig. 18 (right minipage) keeps its code and data; only its widths change so
  that all three panels fit next to Fig. 17.

With --render, the result is compiled in the paper's own class (acmart, the
paper's newcommands.tex) and the width of each picture is checked against its
minipage.

Usage:
  scripts/plot_figure17.py --paper ../arXiv-2608.04573v1 [--distances 5,7,9,11,21,61]
                           [--output results/distillation/compare_new.tex] [--render]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PACKAGE_DIR / "results" / "distillation" / "distill_compare.csv"
DEFAULT_OUT = PACKAGE_DIR / "results" / "distillation" / "compare_new.tex"

# Layout (fractions of \textwidth). Left + gap + right must stay below 1.
LEFT, RIGHT, GAP = 0.48, 0.49, "0.5em"

LEFT_FIGURE = r"""\begin{minipage}{%(left)s\textwidth}

\centering
    \begin{tikzpicture}
    \begin{groupplot}[
        group style={
            group size=3 by 1,
            horizontal sep=0.66cm,
        },
        height=0.25*396pt,
        x=0.24cm,
        enlarge x limits={abs=0.12cm},
        ybar,
        log origin=infty,
        area legend,
        symbolic x coords={%(coords)s},
        xtick=data,
        xticklabels={%(ticklabels)s},
        x tick label style={font=\tiny, yshift=0.5em},
        xtick style={draw=none},
        xlabel={Code distance},
        grid=both,
        grid style={dashed,gray!30},
        y tick label style={font=\tiny, xshift=0.3em},
        label style={font=\tiny},
        legend style={
            at={(-0.85,-0.40)},
            anchor=north,
            legend columns=2,
            draw=none,
            font=\scriptsize,
            fill=white,
            fill opacity=0.8,
            text opacity=1
        },
        title style={font=\scriptsize},
        xlabel style={yshift=0.5em},
        ylabel style={yshift=-0.5em},
    ]

    \nextgroupplot[
        title={Space cost},
        ylabel={\# physical qubits},
        ymode=log,
        log basis y=10,
        ymin=100,
    ]
%(space)s
    \nextgroupplot[
        title={Time cost},
        ylabel={\(\#\mathrm{SE}\) cycles},
        ymode=log,
        log basis y=10,
        ymin=1,
        ymax=4000,
    ]
%(time)s
    \nextgroupplot[
        title={Output error \(\epsilon_{\mathrm{acc}}\)},
        ymode=log,
        log basis y=10,
        ymin=1e-12,
        ymax=1e-4,
        ytick={1e-12,1e-9,1e-6},
    ]
%(error)s
    \legend{\mbox{Our framework\quad}, Litinski~\cite{Litinski2019magicstate}}
    \end{groupplot}
    \end{tikzpicture}
  \caption{Comparison of space and time costs at different code distances for the program in Fig.~\ref{fig:distill-code}.}
  \label{fig:distill-compare}
\end{minipage}"""

BARS = r"""    \addplot+[fill=blue!50, draw=blue!80, /pgf/bar width=2.4pt, bar shift=-1.3pt] coordinates {
        %(ours)s
    };
    \addplot+[fill=orange!55, draw=orange!90, /pgf/bar width=2.4pt, bar shift=1.3pt] coordinates {
        %(lit)s
    };
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=Path, required=True, help="paper folder (contains contents/compare.tex)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--distances", default="5,7,9,11,21,61")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--render", action="store_true", help="compile with the paper's class and check widths")
    args = parser.parse_args()

    distances = [int(d) for d in args.distances.split(",")]
    rows = {int(r["d"]): r for r in csv.DictReader(args.csv.open(encoding="utf-8"))}
    missing = [d for d in distances if d not in rows]
    if missing:
        raise SystemExit(f"{args.csv} has no rows for d = {missing}; run scripts/run_distill_compare.sh")

    def bars(ours: str, lit: str, fmt) -> str:
        return BARS % {
            "ours": " ".join(f"(d{d},{fmt(rows[d][ours])})" for d in distances),
            "lit": " ".join(f"(d{d},{fmt(rows[d][lit])})" for d in distances),
        }

    integer = lambda v: f"{round(float(v))}"  # noqa: E731
    sci = lambda v: f"{float(v):.4e}"  # noqa: E731
    left = LEFT_FIGURE % {
        "left": f"{LEFT:.2f}",
        "coords": ",".join(f"d{d}" for d in distances),
        "ticklabels": ",".join(f"{{${d}$}}" for d in distances),
        "space": bars("ours_qubits", "lit_qubits", integer),
        "time": bars("ours_cycles_attempt", "lit_cycles_attempt", integer),
        "error": bars("ours_eps_acc", "lit_p_out", sci),
    }

    source = (args.paper / "contents" / "compare.tex").read_text(encoding="utf-8")
    start = source.index(r"\begin{minipage}{")
    end = source.index(r"\end{minipage}", start) + len(r"\end{minipage}")
    text = source[:start] + left + source[end:]
    # Fig. 18: narrower panels and gaps, so all three fit in its minipage.
    text = re.sub(r"\\hspace\{1em\}", lambda _: rf"\hspace{{{GAP}}}", text, count=1)
    text = re.sub(r"\\begin\{minipage\}\{0\.\d+\\textwidth\}\n  \\centering\n\\begin\{tikzpicture\}",
                  lambda _: rf"\begin{{minipage}}{{{RIGHT:.2f}\textwidth}}" + "\n  \\centering\n\\begin{tikzpicture}",
                  text, count=1)
    text = text.replace("    horizontal sep=2.0em,\n  },\n  width=0.25*396pt,",
                        "    horizontal sep=1.55em,\n  },\n  width=0.205*396pt,")
    # The adder's time axis reaches 800 ms: fewer ticks so the labels do not touch.
    text = text.replace("  title={Ripple-Carry Adder},", "  title={Ripple-Carry Adder},\n  xtick={0,400,800},", 1)
    # Keep Fig. 18's legend clear of its caption.
    text = text.replace("\\end{tikzpicture}\n\\vspace{-1.8em}", "\\end{tikzpicture}\n\\vspace{-1.1em}", 1)
    # Its legend row was the widest part: \enspace gaps, no trailing space.
    legend_start = text.index(r"\text{\legendLine{qubitsHundred}")
    legend_end = text.index(r"\end{array}", legend_start)
    legend = text[legend_start:legend_end].replace(r"\qquad\\", r"\\").replace(r"\qquad", r"\enspace")
    text = text[:legend_start] + legend + text[legend_end:]
    header = (f"% Fig. 17 data generated by ftqp_prototype/scripts/plot_figure17.py from\n"
              f"% results/distillation/distill_compare.csv (d = {args.distances}).\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + text, encoding="utf-8")
    print(f"wrote {args.output}")
    return render(args.paper, args.output) if args.render else 0


HARNESS = r"""\documentclass[acmsmall, screen, nonacm]{acmart}
\title{Figures 17 and 18}
\input{newcommands}
\makeatletter
\tikzset{every picture/.append style={execute at end picture={%
  \pgfpointdiff{\pgfpointanchor{current bounding box}{west}}{\pgfpointanchor{current bounding box}{east}}%
  \typeout{PICTUREWIDTH \the\pgf@x}}}}
\makeatother
\AtBeginDocument{\let\origcaption\caption
  \renewcommand{\caption}{\typeout{CAPTIONMARK}\origcaption}}
\begin{document}
\typeout{TEXTWIDTH \the\textwidth}
\input{FIGURE}
\end{document}
"""


def render(paper: Path, figure: Path) -> int:
    if shutil.which("pdflatex") is None:
        print("pdflatex not found: skipping the render", file=sys.stderr)
        return 0
    with tempfile.TemporaryDirectory(prefix="figure17-") as tmp:
        tex = Path(tmp) / "figure17.tex"
        tex.write_text(HARNESS.replace("FIGURE", str(figure.resolve())), encoding="utf-8")
        env = dict(os.environ, TEXINPUTS=f".:{paper.resolve()}//:")
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=tmp, env=env,
                             capture_output=True, text=True)
        log = (Path(tmp) / "figure17.log").read_text(errors="replace")
        if run.returncode != 0:
            print("\n".join(l for l in log.splitlines() if l.startswith("!"))[:2000], file=sys.stderr)
            return 1
        textwidth = float(re.search(r"TEXTWIDTH ([0-9.]+)pt", log).group(1))
        # Widest picture before each caption (legend images are small pictures too).
        widths, current = [], 0.0
        for kind, value in re.findall(r"(PICTUREWIDTH|CAPTIONMARK) ?([0-9.]*)", log):
            if kind == "CAPTIONMARK":
                widths.append(current)
                current = 0.0
            else:
                current = max(current, float(value.rstrip(".")))
        boxes = [LEFT * textwidth, RIGHT * textwidth]
        for name, w, box in zip(("Fig. 17", "Fig. 18"), widths, boxes):
            status = "fits" if w <= box + 0.5 else f"TOO WIDE by {w - box:.1f}pt"
            print(f"{name}: picture {w:.1f}pt, minipage {box:.1f}pt -> {status}")
        overfull = [l for l in log.splitlines() if l.startswith("Overfull")]
        if len(widths) != 2:
            print(f"warning: measured {len(widths)} figures, expected 2", file=sys.stderr)
        print("overfull boxes:", overfull or "none")
        pdf = figure.with_suffix(".pdf")
        shutil.copy(Path(tmp) / "figure17.pdf", pdf)
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "160", "-f", "1", "-l", "1", "-singlefile",
                            str(pdf), str(pdf.with_suffix(""))], check=False)
        print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
