#!/usr/bin/env python3
"""Recompute percentile distance sweep plus latency and Hadamard sweeps."""
from pathlib import Path
import sys, re, json, csv, subprocess, os, shutil, hashlib, argparse
PKG=Path(__file__).resolve().parents[1]
SOURCE=PKG/'results/figure14_p003_1us'
OUT=PKG/'results/figure13_p003_sweeps'
sys.path.insert(0,str(PKG/'examples/logicalT-exp'))
import collect_results as cr
cr.SEMANTICS='v2'; cr.RAW_DIR=OUT/'v2/raw'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    args=parser.parse_args()
    implementations=['naive','autocorrected']
    out=OUT/'v2';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'configuration.json').read_text())
    mean=cfg['latency_cycles']['mean']['27']; pct=cfg['latency_cycles']['99.9']['27']
    lat=[0,mean/2,mean,pct,2*pct]; ks=[0,36,54,72,90]
    for ds in ['99.9','latency','clifford']:(OUT/'inputs'/ds).mkdir(parents=True,exist_ok=True)
    for ds in ['latency','clifford']:
        folder=OUT/'inputs'/ds
        for old in folder.glob('*.ft'):
            assert re.fullmatch(r'(naive|autocorrected)(_hybrid)?_[0-9.]+\.ft',old.name)
            for suffix in ['.stdout.txt','.stderr.txt']:
                (cr.RAW_DIR/ds/(old.stem+suffix)).unlink(missing_ok=True)
            old.unlink()
    for p in (SOURCE/'inputs/99.9').glob('*.ft'):shutil.copy2(p,OUT/'inputs/99.9'/p.name)
    top=cr.collect_dataset(PKG.parent,OUT/'inputs/99.9');assert len(top)==18
    tables=cr.render_tables(top,'99.9th percentile','percentile')+'\n'
    results={};rows=[]
    for sweep,values in [('latency',lat),('clifford',ks)]:
        for x in values:
            t=x if sweep=='latency' else pct
            k=36 if sweep=='latency' else int(x)
            scenario=lat.index(x) if sweep=='latency' else 3
            def delay(distance):
                m=cfg['latency_cycles']['mean'][str(distance)];q=cfg['latency_cycles']['99.9'][str(distance)]
                return [0,m/2,m,q,2*q][scenario]
            for impl in implementations:
                source_name=f'd=27_{impl}.ft'
                text=(SOURCE/'inputs/99.9'/source_name).read_text()
                def replace_delay(m):return m.group(1)+f'{delay(int(m.group(2))):g}*tSE\"'
                text,n=re.subn(r'(qecc\s+surface_code_(\d+)\b.*?tdec:\s*")[^"]+"',replace_delay,text,flags=re.S);assert n==1
                # Match only the consecutive data-register H gates, never the ancilla H.
                text,n=re.subn(r'(?:H\[q\];\s*){36}',('H[q];'*k if k else 'skip;')+'\n',text);assert n==1
                assert len(re.findall(r'H\[q\];',text))==k
                path=OUT/'inputs'/sweep/f'{impl}_{x:g}.ft';path.write_text(text)
                stdout=cr.run_ftqp(PKG.parent,path)
                err=cr.parse_final_q_epsilon(stdout)*100
                time=float(cr.FTQP_SE_CYCLES_RE.search(stdout).group(1))
                space=int(cr.FTQP_SPACE_RE.search(stdout).group(1))
                assert time>0 and 0<=err<=100
                results[sweep,x,impl]=(err,time,space)
                rows.append(dict(sweep=sweep,x=x,implementation=impl,d_data=27,d_anc=27,tdec_cycles=t,tdec_data_cycles=delay(27),tdec_anc_cycles=delay(27),scenario=['zero','half_mean','mean','p999','twice_p999'][scenario],k=k,error_percent=err,time_cycles=time,space_qubits=space))
        lines=[r'\pgfplotstableread{','x errA timeA errB timeB']
        for x in values:
            cells=[f'{x:g}']
            for impl in implementations:
                r=results[sweep,x,impl];cells.extend([f'{r[0]:.8f}',f'{r[1]:.8f}'])
            lines.append(' '.join(cells))
        tables+='\n'.join(lines)+'\n}\\plot'+sweep+'\n'
    for impl in implementations:
        assert results['latency',pct,impl]==results['clifford',36,impl]
        old=top[27,'A' if impl=='naive' else 'B','odd']
        new=results['latency',pct,impl]
        assert abs(old.err_percent-new[0])<1e-9 and old.time_cycles==round(new[1])
    with (out/'sweeps.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'exp_tables.tex').write_text(tables)
    (OUT/'configuration.json').write_text(json.dumps({'semantics':'v2','p':.003,'tSE_us':1,'d_sweeps':27,'mean_cycles':mean,'percentile_cycles':pct,'latency_values':lat,'latency_k':36,'hadamard_values':ks,'hadamard_latency':pct,'latency_control':'Fixed distance d=27 for all blocks in the bottom row','case_count':38,'validation':'all passed; H counts and shared-point agreement verified','inputs':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'inputs').rglob('*.ft')}},indent=2))
    # Reuse the prior figure's percentile row and visual style, not old sweep results.
    source=(SOURCE/'template.tex').read_text()
    pictures=re.findall(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',source,re.S);assert len(pictures)==2
    top_tex=pictures[1].replace('at={(1.3,-0.45)}','at={(1.3,1.30)}').replace('anchor=north,','anchor=south,')
    bottom=r'''
\begin{tikzpicture}
\pgfplotsset{oddA/.style={blue,solid,mark=*},oddB/.style={red,solid,mark=square*},evenA/.style={blue,opacity=0.2,mark=*,mark size=1.2pt},evenB/.style={red,opacity=0.2,mark=square*,mark size=1.2pt}}
\begin{groupplot}[
 group style={group name=sweeps,group size=4 by 1,horizontal sep=3.3em},
 width=0.27\textwidth,height=0.26\textwidth,
 grid=both,ymin=0,
 tick label style={font=\scriptsize},yticklabel style={text width=1.8em,align=right},
 label style={font=\scriptsize},title style={font=\scriptsize},
 xlabel style={align=center}, ylabel style={yshift=-0.5em},
 legend style={font=\scriptsize,at={(1.3,-0.65)},anchor=north,legend columns=4},
]
'''
    for idx,(sweep,metric,title) in enumerate([('latency','time','Time cost'),('latency','err','Error rate'),('clifford','time','Time cost'),('clifford','err','Error rate')]):
        condition = r'd=27,k=36' if sweep=='latency' else r'd=27,t_{\mathrm{dec}}=839'
        opts=[f'title={{{title}\\\\${condition}$}}', 'title style={align=center}']
        if sweep=='latency':
            opts += [r'xlabel={Latency\\($\#\mathrm{SE}$ cycles)}', 'xtick={'+','.join(f'{x:g}' for x in lat)+'}', 'xticklabels={0,215,431,839,1678}',r'xticklabel style={rotate=45,anchor=north east,/pgf/number format/1000 sep={}}']
        else:
            opts += [r'xlabel={Hadamard count $k$}',r'xtick={0,36,54,72,90}']
        if metric=='time':
            opts += [r'ylabel={Total $\#\mathrm{SE}$ cycles}',r'scaled y ticks=false',r'yticklabel={\pgfmathparse{\tick/1000}\pgfmathprintnumber[fixed,precision=1]{\pgfmathresult}}',r'extra description/.code={\node[font=\scriptsize,anchor=south west] at (axis description cs:-0.13,1.04) {$\times10^3$};}']
        else:opts += [r'ylabel={Error rate (\%)}']
        bottom+='\\nextgroupplot['+','.join(opts)+']\n'
        for variant,style in [('A','oddA'),('B','oddB')]:
            bottom+='\\addplot+['+style+'] table[x=x,y='+metric+variant+'] {\\plot'+sweep+'};\n'
    bottom+=r'\end{groupplot}\end{tikzpicture}'
    (OUT/'template.tex').write_text(top_tex+'\n\\par\\vspace{1.5em}\n'+bottom+'\n')
    env=os.environ.copy();env.update(FIGURE14_TEMPLATE=str(OUT/'template.tex'),FIGURE14_OMIT_CAPTION='1')
    subprocess.run(['bash',str(PKG/'scripts/plot_figure13.sh'),'--semantics','v2','--row2','percentile','--results',str(OUT)],env=env,check=True)
    runtime=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    subprocess.run([str(runtime),str(PKG/'scripts/crop_plot_pdf.py'),str(out/'figure13.pdf')],check=True)
    print(json.dumps({'cases':38,'latencies':lat,'k':ks,'output':str(out)},indent=2))
if __name__=='__main__':main()
