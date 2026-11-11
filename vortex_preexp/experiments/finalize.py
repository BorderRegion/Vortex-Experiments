"""Rebuild the conditional disposition, readable report, figures and evidence audit."""
import json
import subprocess
import sys
from src.common import ROOT


def main():
    facts=json.loads((ROOT/'results/raw/final_disposition.json').read_text())
    commands=[
        ['experiments.p3','--decision',facts['p3_decision'],'--reason',facts['p3_reason']],
        ['experiments.summarize'],
        ['experiments.write_report','--decision',facts['decision'],'--strongest',facts['strongest'],'--fatal',facts['fatal']],
        ['experiments.audit'],
    ]
    for command in commands:
        subprocess.run([sys.executable,'-m',*command],cwd=ROOT,check=True)


if __name__=='__main__':main()
