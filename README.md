# FTQP Prototype

This folder contains a Python prototype for the fault-tolerant quantum
programming language and resource-estimation semantics described in
`theory/main.pdf`, `theory/3-language.tex`, `theory/4-qecc.tex`, and
`theory/5-re.tex`.

The implementation uses only the Python standard library. Graphviz is optional
and is only needed if you want to turn DOT output into SVG/PNG/PDF images.

## Implemented Components

- Parser and AST construction for `.ft` files.
- Hardware declarations with `p1`, `p2`, `pm`, `t1`, `t2`, and `tm`.
- QEC declarations explicitly bound to hardware with
  `qecc <name> using <hardware>`.
- QEC fields `SE`, `PREP`, `OPL`, `OPF`, `tdec`, `pL`, and `naux`.
- Program constructs: scoped logical allocation, `prepare`/`reset`,
  primitive operations, sequencing, parallel composition, measurement
  conditionals, unary `map` operations, and subcircuit calls.
- Static checks for declaration dependencies, QEC/hardware binding, primitive
  support and arity, single-code-block primitive constraints, joint Pauli arity
  and distinct targets, disjoint parallel footprints, call arity, and acyclic
  call graphs.
- Symbolic compilation into a dynamic-circuit representation.
- ASCII and Graphviz DOT circuit timeline visualization with one row per code
  block, explicit lifetimes, measurement annotations, and dashed control
  dependency arrows in DOT output.
- Big-step resource semantics with time, auxiliary space, and evolving error
  context, in two versions (see "Error Semantics: v1 and v2" below):
  - `resource/estimator_v2.py` (default): source-aware rules of the revised §5
    (arXiv-2608.04573 §5.1–5.4), with program output error and distillation
    detection (§5.5).
  - `resource/estimator.py` (`--semantics v1`): scalar rules of the first paper.

## Quick Start

Run from the repository root:

```bash
cd /home/subonan/ftqp
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft
```

This parses and validates the `.ft` file, prints the compiled circuit timeline,
and prints the resource estimate.

If Python cannot find the package, either run from `/home/subonan/ftqp` or set
`PYTHONPATH`:

```bash
PYTHONPATH=/home/subonan/ftqp \
python -m ftqp_prototype /home/subonan/ftqp/ftqp_prototype/examples/cnot.ft
```

## CLI Reference

```bash
python -m ftqp_prototype [OPTIONS] file
```

Options:

```text
-h, --help       Show command help.
--dot            Render Graphviz DOT instead of ASCII.
--no-circuit     Suppress compiled circuit visualization.
--no-estimate    Suppress resource estimate.
--trace          Print the resource/error-context trace.
--tree           Show the structural compiler tree instead of the circuit timeline.
--check-only     Parse and type-check without compiling or estimating.
--semantics v1|v2|distill  Error analysis (default v2); distill = §5.5
                 estimator for distillation programs. Time and space are identical.
--outputs a,b    v2: program-level registers whose joint error is reported.
--no-distill     v2: do not detect distillation protocols.
--no-storage     distillation (v2 plug-in and --semantics distill): leave out
                 storage errors of the outputs and checks (plain §5.5 recursion).
--no-compress    v2: slow reference mode (one id per fault event), for testing.
```

## Common Commands

Default output, including circuit visualization and resource estimate:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft
```

Only show the compiled circuit timeline:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-estimate
```

Only show the resource estimate:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-circuit
```

Show the resource estimate with the full error-context trace:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-circuit --trace
```

Show both the circuit visualization and the trace:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --trace
```

Show the structural compiler tree instead of the timeline:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --tree --no-estimate
```

The tree view is mainly for debugging compiler expansion. The default timeline
view is the intended circuit visualization.

## VS Code Syntax Highlighting

This repository includes a local VS Code extension in `vscode-ftqp`.
To enable syntax highlighting for `.ft` files during development:

```bash
mkdir -p ~/.vscode/extensions
ln -s /home/subonan/ftqp/vscode-ftqp ~/.vscode/extensions/ftqp-syntax
```

Then reload VS Code. The extension registers the `FTQP` language mode and
highlights declarations, QEC fields, typed subcircuit parameters, measurements,
Pauli operators, comments, strings, and numbers.

## Visualization

The default circuit visualization is an ASCII timeline:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-estimate
```

Rows are code blocks. Columns are scheduled compiled stages. A block row is
blank before allocation and after release.

Timeline legend:

```text
A  allocation
P  prepare/reset
O  primitive operation
M  measured block
C  decode+controlled operation
R  release
```

The step list below the timeline includes details such as which blocks are
measured and which blocks are controlled by measurement outcomes.

## Graphviz DOT And SVG/PNG/PDF Output

Generate pure DOT for the compiled circuit timeline:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --dot > circuit.dot
```

Render the DOT file with Graphviz:

```bash
dot -Tsvg circuit.dot -o circuit.svg
dot -Tpng circuit.dot -o circuit.png
dot -Tpdf circuit.dot -o circuit.pdf
```

DOT output has one row per live code block, no placeholder wires outside a
block lifetime, and dashed red arrows for measurement-outcome control
dependencies.

Generate DOT for the structural compiler tree:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --tree --dot > tree.dot
dot -Tsvg tree.dot -o tree.svg
```

When `--dot` is used, the program emits pure DOT on stdout with no text
header, so it can be redirected directly to Graphviz. If resource estimation is
enabled, the estimate is printed to stderr instead of stdout.

## Resource Estimation And Trace Output

Print the summary estimate:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-circuit
```

Print the full trace:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-circuit --trace
```

The trace table contains:

```text
#        evaluation step number
event    statement or resource-semantics event
time     absolute time cost for that event
#SE      corresponding time cost in syndrome-extraction cycles
space    auxiliary physical qubits for that event
live     number of live code blocks in the error context
max eps  largest current error probability among live blocks
context  compact summary of grouped live blocks and probabilities
```

Generated registers introduced during subcircuit expansion are displayed as
`name@call<line>:<column>` instead of the raw internal rename marker.

## Error Semantics: v1 and v2

Both versions compute the same time and space. They differ in how errors are
tracked:

- **v1** keeps one error probability ε per code block.
- **v2** (default) keeps, per code block, the set of independent *fault sources*
  that may have disturbed it, with ε = 1 − Π(1 − p(s)). A fault shared by several
  blocks, e.g. through a joint measurement, is then counted once when those blocks
  interact again.

v2 also:

- uses `1 − (1 − pL)^n` for n syndrome-extraction rounds instead of `n·pL`;
- counts idle rounds from the end of a block's last operation, without rounding
  up again at every step;
- propagates a measurement's snapshot to the blocks a controlled branch touches,
  with call bodies unfolded. A block reset inside the branch also keeps the
  sources it would have had if the branch were skipped.

**Program outputs.** v2 prints the error of the output registers at release, and
their joint error with shared sources counted once. Registers are classified in
two levels:

1. `anc` is a **reserved register name** for ancillas. It is never an output.
2. Any other register allocated at program level is an output if it is released
   only after all circuit operations have finished. A final measurement on it
   does not change that, since the readout's error counts. Registers released
   mid-program, or allocated inside conditionals, calls or loops, are ancillas.

Override this with `--outputs a,b`.

**Distillation.** A scope that ends with check measurements on its own ancillas,
such as `(if M(X)[anc[0]]; then skip) || …` right before `release(anc)`, is
analyzed as a rotation-form distillation protocol (§5.5). The estimate reports
the acceptance probability and accepted-output error, and the output block
continues with that error. For `examples/distillation/15to1-d11.ft` this gives
A = 0.999 and ε_acc ≈ 4.57e-11 (1.04e-11 without storage errors).

A protocol with several outputs, such as the Bravyi–Haah (3k+8)-to-k family in
`examples/distillation/bh/` (generated by `scripts/bravyi_haah.py`), reports two
errors: ε_acc is the probability that at least one output is wrong, and "error of
one output" is the marginal error of a single output. The outputs fail together,
so in v2 they share fault sources: their joint output error is ε_acc, and each
output block carries its own marginal error (as a bound, equal up to O(ε_acc²)).
For `bh4.ft` (20-to-4): A = 0.998668, ε_acc ≈ 9.78e-8, one output ≈ 5.78e-8.

Storage errors (on by default, `--no-storage` turns them off) are faults on the
outputs and checks while they wait between rotations. They enter the recursion
as extra events from the protocol's timeline (`resource/distill.py`, module
docstring): a block stored for n rounds has an error with 1 - (1 - p_L)^n, half X
and half Z; Z errors flip the block's bit; X errors are moved to the nearer end
of the protocol, and each rotation they cross flips with probability 1/2. On
Litinski's 15-to-1 schedule this matches his density-matrix model within 1e-5
(`scripts/distill_storage.py validate`). At p = 1e-4 it matters for d ≤ 11:

| d | 3 | 5 | 7 | 9 | 11 | ≥ 21 |
|---|---|---|---|---|---|---|
| ε_acc, storage on | 1.04e-3 | 1.66e-5 | 2.28e-7 | 2.92e-9 | 4.57e-11 | 1.04e-11 |
| ε_acc, `--no-storage` | 1.9e-11 | 1.05e-11 | 1.04e-11 | 1.04e-11 | 1.04e-11 | 1.04e-11 |

For distillation programs themselves, use the standalone estimator
(`resource/estimator_distill.py`):

```bash
python -m ftqp_prototype ftqp_prototype/examples/distillation/15to1-d11.ft --no-circuit --semantics distill
```

Time and space follow the normal rules; only the error estimate is §5.5. The
report lists:

- the protocol read from the code: rotation axes P_j, checks and outputs;
- per-rotation q_F, q_m, q_j (and q_j with the rotation ancilla's storage) and
  the flip patterns l_j, h_j;
- the acceptance A, expected attempts 1/A, and the accepted-output error, with
  the values without storage errors for comparison;
- for comparison, the §5.4 disturbance risk without postselection.

**`repeat … until`.** It keeps v1's rule in v2 for now (marked provisional in the
trace).

Design notes and the correspondence to the paper: `ESTIMATOR_V2_DESIGN.md`.

## Examples

For compact constant loading, `map X[q0, q1, ...]` means a sequential
application of the unary `X` primitive to every listed target. Resource
estimation charges every mapped gate separately; `map` is source compression,
not a new logical primitive.

`xor "HEX_MASK" into q[start:width]` is the compact equivalent for loading or
changing a classical basis-state constant. It applies one sequential `X`
primitive for every set bit in the mask.

Available example programs:

```text
ftqp_prototype/examples/15to1.ft
ftqp_prototype/examples/cnot.ft
ftqp_prototype/examples/comments.ft
ftqp_prototype/examples/invalid_parallel.ft
ftqp_prototype/examples/joint-pauli.ft
ftqp_prototype/examples/paper_style.ft
ftqp_prototype/examples/prepare.ft
```

Run the logical CNOT example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --trace
```

Run the joint Pauli measurement example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/joint-pauli.ft --trace
```

Run the preparation/error-refresh example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/prepare.ft --trace
```

The generated `eaa.ft` example contains the complete hierarchical `n=64`
extended-Euclidean modular-inversion circuit, lowered to `X`, `CX`, and `CCX`
primitives. Validate it without expanding its 10,698,654 primitive operations
into a visualization or resource trace:

```bash
python -m ftqp_prototype ftqp_prototype/examples/eaa.ft --check-only
```

Regenerate it from a checkout of the upstream repository with Qiskit available:

```bash
python ftqp_prototype/examples/generate_eaa.py \
  --upstream /path/to/upstream/eea_circuit.py
```

### 200-bit Shor factoring

`examples/shor200.ft` is a complete coherent order-finding circuit for the
200-bit semiprime

```text
903902649900139609986213308567970996774518321965199955568201
```

with base `a=2`. It uses 400 coherent phase-estimation qubits, clean-ancilla
Cuccaro arithmetic, controlled modular multiplication, a cutoff-20 inverse
QFT, and terminal measurements. Arithmetic is lowered to `X`, `CX`, and
`CCX`; only `H` and `CP2` through `CP20` remain as QFT primitives.

Regenerate and run the exhaustive small-width arithmetic self-test:

```bash
source ~/Quantum/bin/activate
python ftqp_prototype/examples/generate_shor200.py
```

Validate or estimate the generated circuit:

```bash
source ~/Quantum/bin/activate
python -m ftqp_prototype ftqp_prototype/examples/shor200.ft --check-only
python -m ftqp_prototype ftqp_prototype/examples/shor200.ft --no-circuit
```

The generated recursive primitive totals are approximately 1.030 billion
`CCX`, 773 million `CX`, 210 million `X`, and 7,410 controlled phase gates.
With the example distance-31 surface-code model, the current estimator reports
2,336,032 peak physical qubits and 3.94083e11 syndrome-extraction cycles
(about 6.48 days at 1.42 microseconds per cycle).
This is a conservative serial schedule with one reusable magic-state factory:
`CCX` is charged as seven T injections, each `CPk` as a 60-T synthesized
rotation, and routing/layout overhead is not yet included.

The `.ft` file implements the quantum order-finding part. Feed a measured
400-bit phase-register value into the companion continued-fraction and GCD
post-processor:

```bash
source ~/Quantum/bin/activate
python ftqp_prototype/examples/shor_postprocess.py MEASURED_INTEGER
```

This is a clean-ancilla, gate-transparent implementation, not the paper's
more intricate `2n+2` dirty-ancilla construction. Its larger logical width
keeps every modular-arithmetic operation directly testable and gives a
defensible surface-code estimate.

Run the intentionally invalid 15-to-1 rotation-style example. This should fail
because bare `Codeblock` formals with `k > 1` must be addressed explicitly as
logical qubits, such as `q2[0]`:

```bash
python -m ftqp_prototype ftqp_prototype/examples/15to1.ft
```

Run the comments example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/comments.ft
```

Run the paper-style allocation/release example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/paper_style.ft
```

Run the intentionally invalid parallel example:

```bash
python -m ftqp_prototype ftqp_prototype/examples/invalid_parallel.ft
```

This should fail with a diagnostic like:

```text
parallel branches touch overlapping code blocks: q[0]
```

## Concrete Syntax

Scoped allocation can use braces:

```ft
logical q[2] := new surface_code {
  prepare(q[0]);
  X[q[0]]
}
```

The paper-style release form is also accepted:

```ft
logical q := new tiny;
prepare(q);
X[q];
release(q)
```

Hardware declarations:

```ft
hardware superconducting = {
  t1: 1e-8,
  t2: 1e-7,
  tm: 1e-6,
  p1: 1e-3,
  p2: 2e-3,
  pm: 3e-3
};
```

QEC declarations use the compact JSON-like form from the theory:

```ft
qecc surface_code using superconducting = {
  [[n,k,d]]: [25, 1, 5],
  SE: { "#ancilla": "n-1", tSE: "6t1+tm" },
  PREP: { time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" },
  OPL: {
    X: { arity: 1, time: "t1+d*tSE", "#SE": "d", "#ancilla": "0" }
  },
  OPF: {},
  tdec: "10*tSE",
  pL: "0.1*(max(p1,p2,pm)/0.0057)^((d+1)/2)",
  naux: "d*d"
};
```

`naux` specifies the auxiliary graph surgery qubits used by joint Pauli
measurement resource estimation. The parser accepts both `naux` and `n_aux`.

Pauli measurements use strings or bare Pauli words:

```ft
if M(ZZ)[q[0], q[1]]; skip then X[q[1]];
if M("XZX")[a, b, c]; skip then Z[a];
```

Pauli operation statements can use a Pauli word over a matching list of target
qubits. Each component is checked as a single-qubit primitive in the target
qubit's QEC scheme, and the statement is interpreted as parallel composition:

```ft
XZ[q[0], q[1]];  // equivalent to X[q[0]] || Z[q[1]]
```

Subcircuit parameters may be typed as `Pauli` or `Codeblock`. For fully typed
subcircuit signatures, call arguments are matched by type rather than only by
position. Pauli formals can be concatenated with literal Pauli factors inside
measurements. A `Codeblock` formal can be indexed as `q[i]` to address logical
qubit `i` inside the actual code block:

```ft
def rotation(q: Codeblock, P: Pauli) {
  logical anc := new surface_code;
  if M(PZ)[q[1], anc]; then skip;
  release(anc)
}

call rotation(Z, q);
```

Line comments start with `//` and run to the end of the line:

```ft
prepare(q);
// X[q];
skip
```

## Validation Behavior

The checker reports source locations and specific errors, including:

- Undeclared hardware or QEC scheme.
- QEC scheme missing hardware binding.
- QEC scheme missing required `naux`.
- Unsupported primitive operation.
- Operation or Pauli measurement arity mismatch.
- Duplicate operation targets.
- Primitive operations crossing code-block boundaries.
- Parallel branches touching overlapping code blocks.
- Calls to undeclared subcircuits.
- Subcircuit call arity mismatch.
- Recursive subcircuit cycles.
- Subcircuit bodies that use a register that is neither a parameter nor
  allocated inside the subcircuit (bodies must be closed, so a call's footprint
  is determined by its arguments).

## Paper Experiments

Scripts in `scripts/` rerun the logical-T experiments of arXiv-2608.04573. Each
takes `--semantics v1|v2|both` (default v2) and writes to `results/`. With v1
they reproduce the numbers of the first paper version.

- `scripts/run_figure14.sh` runs Fig. 13 of the current PDF (Fig. 14 in the
  first version): logical T + 36 H, naive vs auto-corrected, d = 23–31 with
  hybrid even-d points, mean and 99.9th-percentile decoding. It writes the
  pgfplots tables and, if `pdflatex` is installed, `figure13.pdf`/`.png` via
  `scripts/plot_figure13.sh` (the paper's plotting code,
  `scripts/figure13_template.tex`).
- `scripts/run_heteroEC.sh` runs Table 1 ((d_data, d_anc) sweep of the naive T
  gate) and Table 3 ((d_q, d_anc[0], d_anc[1]) sweep of the auto-corrected T
  gate), then picks the lowest error under each space budget. It prints Markdown
  plus LaTeX rows in the paper's format. Helper: `scripts/heteroec.py`.
- `scripts/run_distill_compare.sh` runs the 15-to-1 programs
  (`examples/distillation/15to1-d*.ft`) with the distillation estimator and
  compares them with Litinski's one-level 15-to-1 model at p = 1e-4,
  d_X = d_Z = d_m = d: qubits, cycles, acceptance and output error.
  `scripts/litinski15to1.py` reimplements his error model (5-qubit density
  matrix with faulty-T, storage and lattice-surgery errors; source
  github.com/litinski/magicstates, Apache-2.0). `litinski15to1.py validate`
  checks it against Table 1 of his paper, and `scripts/distill_storage.py
  validate` checks our storage-error recursion against his density matrix.
  Distances 3–11 and 21–61 (`15to1-d3.ft` … `15to1-d61.ft`). Writes
  `results/distillation/`.
- `scripts/plot_figure17.py --paper ../arXiv-2608.04573v1 --render` writes a
  new `contents/compare_new.tex` (`results/distillation/compare_new.tex`): Fig. 17
  regenerated from the comparison CSV (space, time and output error at
  d = 5, 7, 9, 11, 21, 61 by default), and Fig. 18 narrowed so both fit side by side.
  `--render` compiles it in the paper's class and checks each figure's width
  against its minipage.

## Development Checks

Run the estimator v2 test suite. It checks time/space parity with v1 on every
example, hand-computed error values for each rule, output selection,
distillation, and that the compressed mode matches the reference mode:

```bash
python3 ftqp_prototype/tests/test_estimator_v2.py -v
```

Compile the Python package:

```bash
python -m compileall ftqp_prototype
```

Run all non-invalid examples quickly:

```bash
python -m ftqp_prototype ftqp_prototype/examples/cnot.ft --no-circuit
python -m ftqp_prototype ftqp_prototype/examples/joint-pauli.ft --no-circuit
python -m ftqp_prototype ftqp_prototype/examples/prepare.ft --no-circuit
python -m ftqp_prototype ftqp_prototype/examples/comments.ft --no-circuit
python -m ftqp_prototype ftqp_prototype/examples/paper_style.ft --no-circuit
```
