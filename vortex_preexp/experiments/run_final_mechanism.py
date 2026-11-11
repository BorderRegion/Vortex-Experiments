"""Wait for a confirmed training runner, then run final corrected + trained-QK P1."""
import json
import os
import subprocess
import sys
import time
from src.common import ROOT,append_jsonl


def main():
    pid=int((ROOT/'results/raw/p2_runner.pid').read_text())
    while True:
        path=ROOT/'results/raw/p2_jobs.jsonl'
        jobs=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        if len(jobs)==6 and all(x['returncode']==0 for x in jobs):break
        try:
            cmd=open(f'/proc/{pid}/cmdline','rb').read()
            state=open(f'/proc/{pid}/stat').read().split()[2]
            alive=b'experiments.run_p2' in cmd and state not in ['Z','X']
        except FileNotFoundError:alive=False
        if not alive:raise SystemExit('P2 runner stopped before all six jobs completed; inspect its logs.')
        time.sleep(20)
    for name,extra in [('p1',[]),('p1_real_relative',['--real','results/raw/p2_relative_vortex_s2026_qk.pt']),
                       ('p1_real_associative',['--real','results/raw/p2_associative_vortex_s2026_qk.pt'])]:
        command=[sys.executable,'-u','-m','experiments.p1','--name',name]+extra
        with (ROOT/f'results/raw/{name}.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        append_jsonl('results/raw/final_mechanism_jobs.jsonl',dict(name=name,command=command,returncode=result.returncode))
        if result.returncode:raise SystemExit(result.returncode)
        print(name+' complete',flush=True)


if __name__=='__main__':main()
