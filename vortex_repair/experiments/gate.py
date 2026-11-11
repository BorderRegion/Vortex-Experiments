"""A calibration gate controls matched follow-on comparisons, not conclusions."""
import json
from src.common import ROOT,write_json


def main():
    result=json.loads((ROOT/'results/raw/routing_value_rope_safe_s2026.json').read_text())
    rows=[r for r in result['bf16'] if r['length'] in (256,512)]
    assert len(rows)==6
    mean=sum(r['accuracy'] for r in rows)/6;minimum=min(r['accuracy'] for r in rows)
    passed=mean>=.90 and minimum>=.80
    out=dict(passed=passed,mean=mean,minimum=minimum,conditions=6,
             purpose='Proceed to matched Vortex/plain only when ordinary RoPE retrieval is usable.')
    write_json('results/raw/calibration_gate.json',out);print(json.dumps(out),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__': main()
