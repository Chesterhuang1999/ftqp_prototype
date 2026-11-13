#!/usr/bin/env python3
"""Bravyi-Haah (3k+8)-to-k distillation: program generator and analytic baselines.

Draft for the RQ2 check (Sec. 6.2). The protocol is in rotation form, as in
examples/distillation/15to1-d*.ft: k output qubits and 3 check qubits start in
|+>, one pi/8 rotation is applied per column of the (k+3) x (3k+8)
triorthogonal matrix, and the 3 check qubits are measured in X.

Columns (outputs | checks), for even k:
  4 columns   0...0 | 100, 010, 001, 111
  4 columns   1...1 | 100, 010, 001, 111
  3 per output i:  e_i | 110, 101, 011
For k = 4 this is the 20-to-4 protocol of Litinski's twolevel20to4.py up to
column order. For every even k it is G(k) of Bravyi-Haah (arXiv:1209.2426,
Eqs. 24-26) up to column order and a change of basis of the 3 check rows, so
the same code; `bh_compare.py matrix` checks this, including the weight
enumerator 1 + x^8 + 6x^(4+2k) of the check rows.

  python bravyi_haah.py                 # check the matrices, print baselines, write bh<k>.ft here
  python -m ftqp_prototype bh4.ft --no-circuit --semantics distill

Baselines printed, with q the probability that one rotation is faulty
(q = 2 pF / 3 in the estimator):
  exact      distribution over (output flips, check flips), independent faults
  leading    acceptance 1-(3k+8)q; one output (3k+1)q^2; any output (4+3k(k-1)/2)q^2
  brute      full enumeration of the 2^(3k+8) fault patterns (k <= 4)
"""
import itertools, sys
from collections import defaultdict
def columns(k):
    """(outputs bits, check bits) per column of the (k+3) x (3k+8) matrix."""
    cols=[]
    for chk in ((1,0,0),(0,1,0),(0,0,1),(1,1,1)):
        cols.append(((0,)*k, chk))
    for chk in ((1,0,0),(0,1,0),(0,0,1),(1,1,1)):
        cols.append(((1,)*k, chk))
    for i in range(k):
        for chk in ((1,1,0),(1,0,1),(0,1,1)):
            cols.append((tuple(int(j==i) for j in range(k)), chk))
    return cols
def tri(G):
    m=len(G); n=len(G[0])
    return all(sum(G[a][j]&G[b][j] for j in range(n))%2==0 for a,b in itertools.combinations(range(m),2)) and \
           all(sum(G[a][j]&G[b][j]&G[c][j] for j in range(n))%2==0 for a,b,c in itertools.combinations(range(m),3))
def exact(cols,k,q):
    mu={(0,0):1.0}
    for o,c in cols:
        ob=sum(b<<i for i,b in enumerate(o)); cb=sum(b<<i for i,b in enumerate(c))
        nu=defaultdict(float)
        for (a,b),p in mu.items():
            nu[(a,b)]+=p*(1-q); nu[(a^ob,b^cb)]+=p*q
        mu=nu
    A=sum(p for (a,b),p in mu.items() if b==0)
    return A, sum(p for (a,b),p in mu.items() if b==0 and a)/A, sum(p for (a,b),p in mu.items() if b==0 and a&1)/A
def brute(cols, k, q):
    """Enumerate every fault pattern; independent of the recursion in exact()."""
    n = len(cols)
    ob = [sum(b << i for i, b in enumerate(o)) for o, c in cols]
    cb = [sum(b << i for i, b in enumerate(c)) for o, c in cols]
    acc = anyo = per = 0.0
    for mask in range(1 << n):
        a = b = 0
        w = 0
        m = mask
        j = 0
        while m:
            if m & 1:
                a ^= ob[j]; b ^= cb[j]; w += 1
            m >>= 1; j += 1
        if b:
            continue
        pr = q ** w * (1 - q) ** (n - w)
        acc += pr
        if a:
            anyo += pr
        if a & 1:
            per += pr
    return acc, anyo / acc, per / acc

def ft(k,d=21,p="1e-4"):
    cols=columns(k)
    ws=sorted({sum(o)+sum(c) for o,c in cols})
    out=[f'''hardware superconducting = {{
  t1: 1e-8, t2: 1e-7, tm: 1e-6, p1: {p}, p2: {p}, pm: {p}
}};

qecc surface_code using superconducting = {{
  [[n,k,d]]: [{d*d}, 1, {d}],
  SE: {{ "#ancilla": "n-1", tSE: "0.000001" }},
  PREP: {{ time: "d*tSE", "#SE": "d", "#ancilla": "0" }},
  OPL: {{
    Z: {{ arity: 1, time: "d*tSE", "#SE": "d", "#ancilla": "0" }},
    X: {{ arity: 1, time: "d*tSE", "#SE": "d", "#ancilla": "0" }},
    H: {{ arity: 1, time: "d*tSE", "#SE": "d", "#ancilla": "0" }}
  }},
  OPF: {{
    faultyT: {{ arity: 1, time: "2t1+2tm", pF: "max(p1, p2, pm)", "#ancilla": "0" }}
  }},
  tdec: "0",
  pL: "0.1*(max(p1,p2,pm)/0.01)^((d+1)/2)",
  naux: "2*d*d"
}};
''']
    for w in ws:
        ps=", ".join(f"q{i}: Codeblock" for i in range(w)); qs=", ".join(f"q{i}" for i in range(w))
        out.append(f"def rotation{w}(P: Pauli, {ps}) {{\n  logical anc := new surface_code;\n  if M(PZ)[{qs}, anc]; then X[anc];\n  if M(faultyT)[anc]; then P[{qs}];\n  release(anc);\n}}\n")
    outs=[f"q{i}" for i in range(k)]
    body=[f"def distill_circuit({', '.join(o+': Codeblock' for o in outs)}) {{",
          "  logical anc[3] := new surface_code;",
          "  prepare(anc[0]) || prepare(anc[1]) || prepare(anc[2]);",
          "  "+" || ".join([f"H[{o}]" for o in outs]+[f"H[anc[{i}]]" for i in range(3)])+";"]
    for o,c in cols:
        tg=[outs[i] for i in range(k) if o[i]]+[f"anc[{i}]" for i in range(3) if c[i]]
        body.append(f"  call rotation{len(tg)}({'Z'*len(tg)}, {', '.join(tg)});")
    body+=["  (if M(X)[anc[0]]; then skip) || (if M(X)[anc[1]]; then skip) || (if M(X)[anc[2]]; then skip);","  release(anc);","}\n"]
    out.append("\n".join(body))
    prog=["program {"]+[f"    logical {o} := new surface_code;" for o in outs]+[f"    call distill_circuit({', '.join(outs)});"]+[f"    release({o});" for o in reversed(outs)]+["}"]
    out.append("\n".join(prog)+"\n")
    return "\n".join(out)
if __name__=="__main__":
    for k in (2,4,6,8):
        cols=columns(k); G=[[ (o+c)[r] for o,c in cols] for r in range(k+3)]
        q=2e-4/3
        A,anyo,per=exact(cols,k,q)
        print(f"k={k} n={len(cols)} tri={tri(G)} A={A:.6f} (1-(3k+8)q={1-(3*k+8)*q:.6f}) per={per:.4e} ({per/q/q:.3f} q^2; 3k+1={3*k+1}) any={anyo:.4e} ({anyo/q/q:.3f} q^2; 4+3k(k-1)/2={4+3*k*(k-1)//2})")
        if k <= 4:
            b = brute(cols, k, q)
            print(f"      brute force: A={b[0]:.6f} per={b[2]:.4e} any={b[1]:.4e}")
        open(f"bh{k}.ft","w").write(ft(k))
