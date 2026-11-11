import json
import subprocess
import sys
import time
from src.common import ROOT,append_jsonl,write_json


def main():
    pid=int((ROOT/'results/raw/gradient_after_runner.pid').read_text())
    while True:
        if 'GRADIENT CONFIRMATION COMPLETE' in (ROOT/'results/raw/gradient_after_runner.log').read_text():break
        try:
            cmd=open(f'/proc/{pid}/cmdline','rb').read();state=open(f'/proc/{pid}/stat').read().split()[2]
            alive=b'experiments.run_gradient_after' in cmd and state not in ['Z','X']
        except FileNotFoundError:alive=False
        if not alive:raise SystemExit('Gradient check did not complete; investigate before training.')
        time.sleep(20)
    assert json.loads((ROOT/'results/raw/sparse_gradient_check.json').read_text())['passed']
    cfg=json.loads((ROOT/'configs/p2c.json').read_text())
    def run(method):
        name=f'p2c_associative_{method}_s2026'
        command=[sys.executable,'-u','-m','experiments.p2','--task','associative','--method',method,
                 '--layers','4','--steps',str(cfg['steps']),'--batch',str(cfg['batch'])]
        start=time.time();print('START '+name,flush=True)
        with (ROOT/f'results/raw/{name}.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        append_jsonl('results/raw/p2c_jobs.jsonl',dict(name=name,command=command,start=start,end=time.time(),returncode=result.returncode))
        if result.returncode:raise SystemExit(result.returncode)
        print('COMPLETE '+name,flush=True)
        return json.loads((ROOT/f'results/raw/{name}.json').read_text())
    data=run('rope_safe')
    rs=[r for r in data['bf16'] if r['length'] in [256,512] and r['distance'] in [8,32,128]]
    avg=sum(r['accuracy'] for r in rs)/len(rs);minimum=min(r['accuracy'] for r in rs)
    passed=avg>=.9 and minimum>=.8
    write_json('results/raw/p2c_gate.json',dict(passed=passed,mean_accuracy=avg,minimum_accuracy=minimum,
        positive_control_conditions=rs,config=cfg,reason='Four-layer positive control calibrated' if passed else 'Four-layer positive control did not learn within the fixed pilot budget; associative RQ2 remains unresolved.'))
    if passed:
        run('vortex');run('ring_plain')
        command=[sys.executable,'-u','-m','experiments.p1','--real','results/raw/p2c_associative_vortex_s2026_qk.pt','--name','p1_real_associative_fourlayer']
        with (ROOT/'results/raw/p1_real_associative_fourlayer.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    print('P2C CALIBRATION COMPLETE',flush=True)


if __name__=='__main__':main()
