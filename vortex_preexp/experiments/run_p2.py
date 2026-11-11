"""Sequential jobs: never oversubscribe the single card or silently retry runs."""
import json
import subprocess
import sys
import time
from src.common import ROOT,append_jsonl


def main():
    assert (ROOT/'results/raw/p0_pilot.json').exists()
    assert json.loads((ROOT/'results/raw/p0_math.json').read_text())['all_integer_pass']
    config=json.loads((ROOT/'configs/p2.json').read_text())
    for task in config['tasks']:
        for method in config['methods']:
            name=f'p2_{task}_{method}_s{config["seed"]}'
            command=[sys.executable,'-u','-m','experiments.p2','--task',task,'--method',method,
                '--steps',str(config['steps']),'--batch',str(config['batch']),'--seed',str(config['seed']),
                '--eval-samples',str(config['eval_samples']),'--lr',str(config['lr'])]
            start=time.time()
            print('START',name,flush=True)
            with (ROOT/f'results/raw/{name}.log').open('w') as logfile:
                result=subprocess.run(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT)
            append_jsonl('results/raw/p2_jobs.jsonl',dict(name=name,command=command,start=start,end=time.time(),returncode=result.returncode))
            if result.returncode:raise SystemExit(result.returncode)
            print('COMPLETE',name,flush=True)


if __name__=='__main__':main()
