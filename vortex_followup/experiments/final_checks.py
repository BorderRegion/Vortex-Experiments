"""Finish sequential evidence collection once all scheduled training has ended."""
import subprocess
import sys
import time
from src.common import ROOT


def main():
    deadline = time.monotonic()+2400
    while not (ROOT/'results/raw/stage2_complete.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Dependent training did not finish; do not overlap GPU jobs.')
        time.sleep(30)
    for module, name in [('experiments.ring_groups', 'ring_groups'), ('experiments.query_swap', 'query_swap')]:
        if not (ROOT/f'results/raw/{name}.json').exists():
            with (ROOT/f'results/raw/{name}.log').open('x') as log:
                subprocess.run([sys.executable, '-u', '-m', module], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    print('FINAL GPU CHECKS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
