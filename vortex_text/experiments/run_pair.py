"""Sequential single-GPU jobs with durable status and immediate failure propagation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot','train'], required=True)
    ap.add_argument('--steps', type=int, default=6104)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    jobs = []
    suffix = '' if args.seed == 2026 else f'_s{args.seed}'
    for method in ['rope_safe', 'vortex']:
        jobs.append((f'{args.mode}_{method}{suffix}', [sys.executable, '-u', '-m', 'experiments.text',
            '--mode', args.mode, '--method', method, '--steps', str(args.steps), '--seed', str(args.seed)]))
    if args.mode == 'train':
        jobs.append(('final_evaluation'+suffix, [sys.executable, '-u', '-m', 'experiments.evaluate', '--steps', str(args.steps), '--seed', str(args.seed)]))
    status = dict(mode=args.mode, seed=args.seed, steps=args.steps, started_unix=time.time(), jobs=[])
    path = ROOT / f'results/raw/{args.mode}{suffix}_runner.json'
    assert not path.exists(), f'Refuse to overwrite prior runner: {path}'
    for name, command in jobs:
        record = dict(name=name, command=command, started_unix=time.time())
        status['jobs'].append(record)
        path.write_text(json.dumps(status, indent=2)+'\n')
        with (ROOT / f'results/raw/{name}.log').open('x') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=code, ended_unix=time.time())
        path.write_text(json.dumps(status, indent=2)+'\n')
        if code:
            raise SystemExit(code)
    status['completed_unix'] = time.time()
    path.write_text(json.dumps(status, indent=2)+'\n')


if __name__ == '__main__':
    main()
