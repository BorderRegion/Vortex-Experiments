"""Sequential bounded diagnostic runs; never overwrite the completed preexperiment."""
import subprocess
import sys
from src.common import ROOT, write_json


def main():
    jobs = [(['experiments.check_supervision'], 'oracle'),
            (['experiments.diagnose'], 'diagnosis'),
            (['experiments.supervision', '--method', 'rope_safe', '--loss', 'dense'], 'rope_dense'),
            (['experiments.supervision', '--method', 'rope_safe', '--loss', 'sparse'], 'rope_sparse')]
    for args, name in jobs:
        with (ROOT/f'results/raw/{name}.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', *args], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(name+' COMPLETE', flush=True)
    write_json('results/raw/stage1_complete.json', dict(complete=True))


if __name__ == '__main__':
    main()
