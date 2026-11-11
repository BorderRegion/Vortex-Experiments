"""Pull research artifacts to this host and verify remote/local SHA256 values."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from experiments.mirror import TRANSPORT

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/root/autodl-tmp/vortex_text/'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8 << 20),b''):
            h.update(block)
    return h.hexdigest()


def backup(immutable_only=False):
    started=time.time()
    result=subprocess.run(['rsync','-az','--partial','--exclude=__pycache__/',
        '--exclude=*.tmp','--exclude=reports/', '-e',TRANSPORT,
        'root@connect.sha1.seetacloud.com:'+REMOTE,str(ROOT)+'/'],
        capture_output=True,text=True,timeout=600)
    assert result.returncode==0,result.stderr
    # Avoid evolving log/metrics comparisons during training; final backup checks all.
    program='''from pathlib import Path
import hashlib,json
r=Path('/root/autodl-tmp/vortex_text')
immutable_only=IMMUTABLE_FLAG
out={}
for p in sorted(r.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.tmp': continue
 rel=str(p.relative_to(r))
 if rel.startswith('reports/'): continue
 if immutable_only and not (p.suffix=='.pt' or rel.startswith(('data/','src/','configs/','experiments/')) or rel=='README.md'): continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8 << 20),b''): h.update(b)
 out[rel]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(out))
'''.replace('IMMUTABLE_FLAG',repr(immutable_only))
    command="/usr/bin/python - <<'VORTEX_BACKUP_PY'\n"+program+"\nVORTEX_BACKUP_PY"
    hashed=subprocess.run(['/tmp/wcc_ssh',command],capture_output=True,text=True,timeout=600)
    assert hashed.returncode==0,hashed.stderr
    manifest=json.loads(hashed.stdout)
    differences=[]
    for name,expected in manifest.items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            differences.append(name)
    report=dict(started_unix=started,completed_unix=time.time(),local_directory=str(ROOT),
        remote_directory=REMOTE,immutable_only=immutable_only,files=len(manifest),
        bytes=sum(v['bytes'] for v in manifest.values()),mismatches=differences,passed=not differences)
    label='IMMUTABLE_BACKUP' if immutable_only else 'FINAL_BACKUP'
    (ROOT/f'reports/{label}_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/f'reports/{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert not differences,differences


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--immutable-only',action='store_true')
    ap.add_argument('--expiry-guard',action='store_true')
    args=ap.parse_args()
    if args.expiry_guard:
        # Training checks the 22:08:58 deadline each optimizer step. Final pull at
        # 22:10 leaves ~29 min for retries; this watchdog runs on the local host.
        while time.time()<1794406200:
            time.sleep(min(60,1794406200-time.time()))
    for attempt in range(3):
        try:
            backup(args.immutable_only)
            return
        except Exception as exc:
            print(type(exc).__name__,str(exc),flush=True)
            if attempt==2: raise
            time.sleep(20)


if __name__=='__main__':
    main()
