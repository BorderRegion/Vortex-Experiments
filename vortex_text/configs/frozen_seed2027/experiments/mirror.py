"""Local durable mirror until the remote pair finishes or the expiry reserve begins."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/root/autodl-tmp/vortex_text/results/'
TRANSPORT = ('/usr/bin/sshpass -f /tmp/wcc_ssh_password /usr/bin/ssh -p 14139 '
             '-o ControlMaster=auto -o ControlPath=/tmp/wcc_control_14139 -o ControlPersist=600 '
             '-o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=3')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    suffix='' if args.seed==2026 else f'_s{args.seed}'
    while time.time() < 1794406138:
        result = subprocess.run(['rsync','-az','--partial','--exclude=__pycache__/',
            '--exclude=*.tmp',
            '-e',TRANSPORT,'root@connect.sha1.seetacloud.com:'+REMOTE,str(ROOT/'results')+'/'],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
        state=dict(updated_unix=time.time(),rsync_exit=result.returncode,output=result.stdout[-2000:])
        runner=ROOT/f'results/raw/train{suffix}_runner.json'
        if runner.exists():
            try:
                data=json.loads(runner.read_text())
                state['runner']=data
                completed='completed_unix' in data
                failed=any(j.get('exit_code',0)!=0 for j in data['jobs'])
            except json.JSONDecodeError:
                completed=failed=False
        else:
            completed=failed=False
        (ROOT/f'reports/MIRROR_STATUS{suffix}.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
        if completed and result.returncode==0:
            verified=subprocess.call([sys.executable,'-u','-m','experiments.backup'],cwd=ROOT)
            if verified:
                raise SystemExit(verified)
            reported=subprocess.call([sys.executable,'-m','experiments.report','--seed',str(args.seed)],cwd=ROOT)
            if reported:
                raise SystemExit(reported)
            if args.seed==2027:
                raise SystemExit(subprocess.call([sys.executable,'-m','experiments.summarize_seeds'],cwd=ROOT))
            return
        if failed:
            raise SystemExit('Remote experiment failed; evidence mirrored locally')
        time.sleep(60)
    raise SystemExit('Expiry reserve reached; preserve the mirrored completed stages')


if __name__=='__main__':
    main()
