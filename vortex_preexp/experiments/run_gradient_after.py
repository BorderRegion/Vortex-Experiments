import subprocess
import sys
import time
from src.common import ROOT


def main():
    pid=int((ROOT/'results/raw/p2b_runner.pid').read_text())
    while True:
        if 'ALL GPU EXPERIMENTS COMPLETE' in (ROOT/'results/raw/p2b_runner.log').read_text():break
        try:
            cmd=open(f'/proc/{pid}/cmdline','rb').read();state=open(f'/proc/{pid}/stat').read().split()[2]
            alive=b'experiments.run_p2b' in cmd and state not in ['Z','X']
        except FileNotFoundError:alive=False
        if not alive:raise SystemExit('P2B runner stopped before its scheduled completion; inspect its logs.')
        time.sleep(20)
    with (ROOT/'results/raw/sparse_gradient_check.log').open('w') as f:
        result=subprocess.run([sys.executable,'-u','-m','experiments.gradient_check'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
    print('GRADIENT CONFIRMATION COMPLETE',flush=True)


if __name__=='__main__':main()
