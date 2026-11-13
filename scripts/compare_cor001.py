"""Compare old (code+spec as in the paper) vs COR-001 fix (new r_M, t1 absorbed) on every .ft under the given dirs."""
import re, sys, csv, subprocess, tempfile, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OLD = str(Path.home()/"scratch/pkg_old"); NEW = str(Path.home()/"mnt")
def absorb(text):
    return re.sub(r'"t1\s*\+\s*(\d+\s*\*\s*)?d\s*\*\s*tSE"', lambda m: '"' + (m.group(1) or '') + 'd*tSE"', text)
def run(pkg, path):
    out = subprocess.run([sys.executable, "-m", "ftqp_prototype", str(path), "--no-circuit"], cwd="/tmp",
                         env={**os.environ, "PYTHONPATH": pkg}, capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out, re.M) or [None, ""])[1]
    return g(r"^time \(#SE cycles\)\s+(\S+)"), g(r"^space\s+(\d+)"), g(r"^joint output error\s+(\S+)")
tmp = Path(tempfile.mkdtemp())
def job(f):
    new = tmp / (str(abs(hash(str(f)))) + ".ft"); new.write_text(absorb(f.read_text()))
    return [str(f)] + list(run(OLD, f)) + list(run(NEW, new))
files = [f.resolve() for d in sys.argv[2:] for f in sorted(Path(d).rglob("*.ft"))]
with ThreadPoolExecutor(4) as ex: rows = list(ex.map(job, files))
with open(sys.argv[1], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file","old_cycles","old_space","old_err","new_cycles","new_space","new_err"]); w.writerows(rows)
print(len(rows), "programs")
