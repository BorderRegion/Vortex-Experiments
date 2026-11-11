import json
import subprocess
import sys
import time
from src.common import ROOT,append_jsonl


def main():
    pid=int((ROOT/'results/raw/final_mechanism_runner.pid').read_text())
    while True:
        f=ROOT/'results/raw/final_mechanism_jobs.jsonl'
        jobs=[json.loads(s) for s in f.read_text().splitlines()] if f.exists() else []
        if len(jobs)==3 and all(j['returncode']==0 for j in jobs):break
        try:
            cmd=open(f'/proc/{pid}/cmdline','rb').read();state=open(f'/proc/{pid}/stat').read().split()[2]
            alive=b'experiments.run_final_mechanism' in cmd and state not in ['Z','X']
        except FileNotFoundError:alive=False
        if not alive:raise SystemExit('Final original mechanism runner stopped before completion')
        time.sleep(20)
    cfg=json.loads((ROOT/'configs/p2b.json').read_text())
    for method in cfg['methods']:
        name=f'p2b_associative_{method}_s2026'
        command=[sys.executable,'-u','-m','experiments.p2b','--method',method,'--steps',str(cfg['additional_steps'])]
        start=time.time();print('START '+name,flush=True)
        with (ROOT/f'results/raw/{name}.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        append_jsonl('results/raw/p2b_jobs.jsonl',dict(name=name,command=command,start=start,end=time.time(),returncode=result.returncode))
        if result.returncode:raise SystemExit(result.returncode)
        print('COMPLETE '+name,flush=True)
    jobs=[('p1_real_associative_extended',[sys.executable,'-u','-m','experiments.p1','--name','p1_real_associative_extended','--real','results/raw/p2b_associative_vortex_s2026_qk.pt']),
          ('backend_dispatch',[sys.executable,'-u','-m','experiments.verify_backend'])]
    for name,command in jobs:
        with (ROOT/f'results/raw/{name}.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
        print('COMPLETE '+name,flush=True)
    print('ALL GPU EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
