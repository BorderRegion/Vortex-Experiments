"""Single evidence-guided alignment trial after the first binding diagnosis."""
import subprocess
import sys
from src.common import ROOT, write_json
from .continue_stage import gate


def run(method):
    with (ROOT/f'results/raw/alignment_{method}.log').open('x') as f:
        subprocess.run([sys.executable,'-u','-m','experiments.binding','--method',method,'--preserve-query'],
                       cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('alignment '+method+' COMPLETE',flush=True)


def main():
    assert (ROOT/'results/raw/binding_probe.json').exists()
    assert (ROOT/'results/raw/full_gradient.json').exists()
    run('rope_safe')
    result=gate('binding_rope_safe_aligned_s2026')
    write_json('results/raw/alignment_gate.json',result)
    if result['passed']:
        for method in ('vortex','ring_plain'):
            run(method)
    write_json('results/raw/alignment_stage_complete.json',dict(complete=True,gate=result,matched_comparisons_run=result['passed']))
    print('ALIGNMENT STAGE COMPLETE',flush=True)


if __name__=='__main__':
    main()
