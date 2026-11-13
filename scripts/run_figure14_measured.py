#!/usr/bin/env python3
"""Recompute Figure 14 using measured PyMatching latencies, retaining original inputs."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

PKG = Path(__file__).resolve().parents[1]
EXP = PKG / 'examples/logicalT-exp'
MEASURED = {
    'mean': {23: 96.261587876, 25: 126.581714302, 27: 165.811974528, 29: 219.424818992, 31: 281.798007265},
    '99.9': {23: 170.842, 25: 228.932, 27: 326.003, 29: 420.834, 31: 542.596},
}
OLD = {'mean': dict(zip([23,25,27,29,31],[200,280,390,520,680])),
       '99.9': dict(zip([23,25,27,29,31],[0,0,255,470,1760]))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--units',choices=['us','cycles'],default='cycles',help='At tSE=1 us, either display unit has identical numeric values')
    ap.add_argument('--semantics',choices=['v1','v2'],default='v2')
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    out=(a.out or PKG/'results'/('figure14_measured_1us_'+a.units)/a.semantics).resolve()
    out.mkdir(parents=True,exist_ok=True)
    inputs=out/'inputs'
    rounded={ds:{d:math.floor(v+.5) for d,v in vals.items()} for ds,vals in MEASURED.items()}
    spec=importlib.util.spec_from_file_location('figure14_collector',EXP/'collect_results.py')
    collector=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=collector
    spec.loader.exec_module(collector)
    collector.SEMANTICS=a.semantics
    collector.SCRIPT_DIR=inputs
    sections=[f'% Measured PyMatching 2.4.0; latency column unit: {a.units}; error semantics: {a.semantics}']
    all_results=[]
    changed=0
    for ds,label,suffix in collector.DATASETS:
        folder=inputs/ds
        folder.mkdir(parents=True,exist_ok=True)
        for source in sorted((EXP/ds).glob('*.ft')):
            original=source.read_text()
            def replace(m):
                d=int(m.group(2)); old=int(m.group(3))
                assert old==OLD[ds][d],(source,d,old)
                value=rounded[ds][d]
                literal=f'{value/1000000:.6f}' if a.units=='us' else f'{value}*tSE'
                return m.group(1)+literal+'"'
            text,count=re.subn(r'(qecc\s+surface_code_(\d+)\b.*?tdec:\s*")(\d+)\*tSE"',replace,original,flags=re.S)
            assert count==(2 if source.name.startswith('hybrid') else 1),(source,count)
            text,nse=re.subn(r'tSE:\s*"2t1\+4t2\+tm"', 'tSE: "0.000001"', text)
            assert nse==count,(source,nse,count)
            (folder/source.name).write_text(text)
            changed+=count
        # Only the latency display column is in microseconds. The estimator reads
        # each generated program's actual duration and recomputes error/time/space.
        def display_latency(path):
            original=path.read_text()
            d=int(re.search(r'program\s*\{.*?logical\s+q\s*:=\s*new\s+surface_code_(\d+)',original,re.S).group(1))
            return rounded[ds][d]
        collector.parse_dec_cycles=display_latency
        results=collector.collect_dataset(PKG.parent,folder)
        assert len(results)==18
        all_results.extend(results.values())
        sections.append(collector.render_tables(results,label,suffix))
    tables='\n\n'.join(sections)+'\n'
    (out/'exp_tables.tex').write_text(tables)
    template=(PKG/'scripts/figure13_template.tex').read_text()
    if a.units=='us':
        old=r'ylabel={\(\#\mathrm{SE}\) cycles},'
        assert template.count(old)==2
        template=template.replace(old,r'ylabel={Decoding time ($\mu$s)},')
    # Retain the original visual style, fitting the time axes to the new data.
    max_time=max(r.time_cycles for r in all_results)
    step=math.ceil(max_time/4/250)*250
    upper=step*4
    template=template.replace('ymax=11000',f'ymax={upper}')
    template=template.replace('ytick={0,2500,5000,7500,10000}', 'ytick={'+','.join(str(step*i) for i in range(5))+'}')
    template=template.replace('cs:-0.13,0.96', 'cs:-0.13,1.04').replace('cs:0.87,0.96', 'cs:0.87,1.04')
    template=template.replace('percentile decoding time}', 'percentile}')
    # Retain the two aligned four-panel rows. Export a figure body without the
    # old paper-source caption, since these data come from the new measurement.
    (out/'exp_figure.tex').write_text('\\begin{figure}[t]\n\\centering\n'+tables+template+'\n\\end{figure}\n')
    preamble=r'''\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=396pt,margin=0.5in]{geometry}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\centering
'''
    (out/'figure14.tex').write_text(preamble+tables+template+'\n\\end{document}\n')
    metadata={'measured_us':MEASURED,'rounded':rounded,'input_units':a.units,'semantics':a.semantics,'source':'192.168.20.20:/pub/data/huangqf/pymatching_fig2_20261111/results/validation.json','shots_per_distance':1000000,'pymatching':'2.4.0','stim':'1.15.0','stim_noise_parameters':0.001,'ftqp_hardware':{'t1_s':1e-8,'t2_s':1e-7,'tm_s':1e-6,'tSE_us':1.0,'p1':.001,'p2':.002,'pm':.003},'SE_cycle_convention':'1 microsecond, explicitly overrides the gate-duration sum per user request','program_count':36,'qecc_latency_replacements':changed,'time_axis_upper':upper}
    (out/'latencies.json').write_text(json.dumps(metadata,indent=2)+'\n')
    build=out/'build'; build.mkdir(exist_ok=True)
    with (build/'render.log').open('w') as log:
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),str(out/'figure14.tex')],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['pdfcrop','--margins','4',str(build/'figure14.pdf'),str(out/'figure14.pdf')],check=True)
    subprocess.run(['pdftoppm','-png','-r','200','-singlefile',str(out/'figure14.pdf'),str(out/'figure14')],check=True)
    print(json.dumps({'out':str(out),'programs':36,'replacements':changed,'rounded':rounded},indent=2))

if __name__=='__main__':
    main()
