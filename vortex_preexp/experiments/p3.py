"""Record the conditional P3 gate. Never replace a missing text experiment by synthetic NLL."""
import argparse
import json
from src.common import ROOT,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--decision',choices=['skip','run'],required=True)
    ap.add_argument('--reason',required=True)
    args=ap.parse_args()
    p2=list((ROOT/'results/raw').glob('p2_*_s2026.json'))
    real=list((ROOT/'results/raw').glob('p1_real*.json'))
    assert len(p2)==6 and len(real)>=2,'Make the gate only after original P2 and real-QK P1 finish.'
    extensions=list((ROOT/'results/raw').glob('p2b_*_s2026.json'))
    calibrated=list((ROOT/'results/raw').glob('p2c_*_s2026.json'))
    calibration_gate=ROOT/'results/raw/p2c_gate.json'
    assert len(extensions)==3 and calibration_gate.exists(),'Wait for the common continuation and positive-control calibration.'
    cg=json.loads(calibration_gate.read_text())
    assert len(calibrated)==(3 if cg['passed'] else 1),'Complete every conditionally required calibration run.'
    if args.decision=='run':
        raise SystemExit('Gate passed: implement and execute the specified 27M WikiText-103 experiment; no text training is claimed by this gate script.')
    write_json('results/raw/p3_gate.json',dict(executed=False,decision='skip',reason=args.reason,
        evidence_files=[str(p.relative_to(ROOT)) for p in p2+extensions+calibrated+real+[calibration_gate,ROOT/'results/raw/attention_replay.json']],
        text_nll=None,text_perplexity=None,warning='Synthetic-task NLL is not language-model validation NLL.'))


if __name__=='__main__':main()
