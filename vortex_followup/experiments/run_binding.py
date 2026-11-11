"""Run the separately frozen binding diagnostic after prior GPU work ends."""
import json
import subprocess
import sys
import time
from src.common import ROOT, write_json
from .continue_stage import gate


def run(method):
    with (ROOT/f'results/raw/binding_{method}.log').open('x') as f:
        subprocess.run([sys.executable,'-u','-m','experiments.binding','--method',method],cwd=ROOT,
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    print('binding '+method+' COMPLETE',flush=True)


def main():
    deadline=time.monotonic()+2400
    while not (ROOT/'results/raw/query_swap.json').exists():
        if time.monotonic()>deadline:
            raise RuntimeError('Prior GPU stages did not end in time.')
        time.sleep(30)
    run('rope_safe')
    result=gate('binding_rope_safe_s2026')
    write_json('results/raw/binding_gate.json',result)
    if result['passed']:
        for method in ('vortex','ring_plain'):
            run(method)
    write_json('results/raw/binding_stage_complete.json',dict(complete=True,gate=result,matched_comparisons_run=result['passed']))
    print('BINDING STAGE COMPLETE',flush=True)


if __name__=='__main__':
    main()
