"""One frozen curriculum fallback and matched comparisons, only after stage1 ends."""
import json
import subprocess
import sys
import time
from src.common import ROOT, write_json


def gate(name):
    data = json.loads((ROOT/f'results/raw/{name}.json').read_text())
    rows = [r for r in data['bf16'] if r['length'] in (256, 512) and r['distance'] in (8, 32, 128)]
    mean = sum(r['accuracy'] for r in rows)/len(rows)
    minimum = min(r['accuracy'] for r in rows)
    return dict(mean=mean, minimum=minimum, passed=mean >= .9 and minimum >= .8)


def run(args, name):
    with (ROOT/f'results/raw/{name}.log').open('x') as log:
        subprocess.run([sys.executable, '-u', '-m', *args], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print(name+' COMPLETE', flush=True)


def main():
    # Never overlap GPU experiments. This wait imports no MUSA and allocates no device tensors.
    deadline = time.monotonic()+1800
    while not (ROOT/'results/raw/stage1_complete.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Stage1 did not complete in 30 minutes; do not start dependent runs.')
        time.sleep(30)
    result = gate('supervision_rope_safe_dense_s2026')
    write_json('results/raw/flat_gate.json', result)
    schedule = 'flat'
    if not result['passed']:
        run(['experiments.supervision', '--method', 'rope_safe', '--loss', 'dense', '--schedule', 'curriculum'], 'rope_curriculum')
        result = gate('supervision_rope_safe_dense_curriculum_s2026')
        write_json('results/raw/curriculum_gate.json', result)
        schedule = 'curriculum'
    if result['passed']:
        for method in ('vortex', 'ring_plain'):
            run(['experiments.supervision', '--method', method, '--loss', 'dense', '--schedule', schedule], f'{method}_{schedule}')
    run(['experiments.ring_groups'], 'ring_groups')
    write_json('results/raw/stage2_complete.json', dict(complete=True, schedule=schedule,
               rope_gate=result, matched_comparisons_run=result['passed']))
    print('FOLLOWUP TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
