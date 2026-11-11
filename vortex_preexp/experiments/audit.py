"""Completion evidence checks for the specified conditional experiment."""
import hashlib
import json
import math
from pathlib import Path
from src.common import ROOT,write_json


def main():
    checks={};failures=[]
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:failures.append(name)
    def load(name):return json.loads((ROOT/'results/raw'/name).read_text())
    m=load('p0_math.json');env=load('p0_environment.json')
    check('P0 integer group/sign/dot-product invariants',m['all_integer_pass'])
    check('P0 exact matrix orthogonality and independent trace cross-check',load('p0_math_supplement.json')['passed'])
    check('P0 orthogonal spectral basis',m['orthogonality_max_abs']<1e-12)
    check('P0 decomposition and logits FP32 equivalence',m['decomposition_max_abs']<1e-4 and m['device_fp32_logit_max_abs']<1e-4)
    check('P0 full weight conversion equivalence',m['full_checkpoint_fp32_max_abs']<1e-4)
    check('actual MUSA device and BF16 backward',env['musa_available'] and env['checks']['torch.bfloat16']['backward_finite'] and 'musa' in env['checks']['torch.bfloat16']['device'])
    check('SDPA forward/backward verified',env['checks']['sdpa']['backward_finite'] and 'musa' in env['checks']['sdpa']['forward_device'])
    oracle=load('data_oracle.json')
    check('independent synthetic label and matched-control oracle',oracle['passed'] and oracle['paired_associative_queries_and_answers_identical'])
    dispatch=load('backend_dispatch.json')
    check('SDPA actual dispatch evidence',all('musa' in r['output_device'] and r['backward_finite'] and r['dispatch_operators'] for r in dispatch['rows']))
    check('CPU/MUSA sparse-supervision gradient agreement',load('sparse_gradient_check.json')['passed'])
    check('P0 correlation table complete',set(['0','1','2','11','58','1333','1334','1335','14674','25346'])<=set(load('p0_correlation.json')['selected']))
    files=sorted((ROOT/'results/raw').glob('p1*.json'))
    check('P1 synthetic + both trained QK distributions',len(files)>=3 and (ROOT/'results/raw/p1_real_relative.json').exists() and (ROOT/'results/raw/p1_real_associative.json').exists())
    for file in files:
        data=json.loads(file.read_text());comm=data['commutation']
        check(file.stem+' head INT4 exact commutation',all(r['dequant_max_abs']==0 and r['code_match_rate']==1 for r in comm if r['kind']=='int4' and r['group']==128))
        check(file.stem+' group16 crossing measured',any(r['dequant_max_abs']>0 and r['code_match_rate']<1 for r in comm if r['kind']=='int4' and r['group']==16))
        rs=[r for r in data['rows'] if r['kind']=='int4']
        check(file.stem+' all offsets/scopes/groups/methods',all(any(r['method']==method and r['offset']==offset and r['group']==group and r['scope']==scope for r in rs) for method in ['rope_safe','vortex_pre','vortex_post','spectral_rope'] for offset in [0,256,4096,65536,1000000] for group in [128,16] for scope in ['qk','k']))
    p2=[]
    for task in ['relative','associative']:
        variants=[]
        for method in ['rope_safe','vortex','ring_plain']:
            name=f'p2_{task}_{method}_s2026';data=load(name+'.json');variants.append(data);p2.append(data)
            curve=[json.loads(x) for x in (ROOT/f'results/raw/{name}_curve.jsonl').read_text().splitlines()]
            check(name+' complete equal token/step budget',data['train_tokens']==61440000 and data['optimizer_steps']==12000 and curve[-1]['step']==12000)
            check(name+' train/test seeds disjoint',data['dataset_train_seed']==12026 and '900000' in data['dataset_test_seed_formula'])
            check(name+' required evaluation lengths',{r['length'] for r in data['bf16']}=={256,512,1024,2048})
            check(name+' original and middle checkpoints',(ROOT/f'results/checkpoints/{name}_final.pt').exists() and (ROOT/f'results/checkpoints/{name}_mid.pt').exists())
            check(name+' PTQ every condition',all(any(q['length']==r['length'] and q['distance']==r['distance'] and q['group']==group and q['scope']==scope and q['method']==method for q in data['ptq']) for r in data['bf16'] for group in [128,16] for scope in ['qk','k']))
            if task=='associative':check(name+' matched collision controls',{1333,1334,1335}<={r['distance'] for r in data['bf16'] if r['length']==2048})
            if method=='vortex':
                check(name+' trained spectral FP32 equivalence',data['spectral_fp32_equivalence']['fp32_output_max_abs']<2e-4)
                check(name+' spectral PTQ complete',all(any(q['length']==r['length'] and q['distance']==r['distance'] and q['group']==group for q in data['spectral']) for r in data['bf16'] for group in [None,128,16]))
        check(task+' identical data stream',len({x['train_data_sha256'] for x in variants})==1)
        check(task+' identical parameter count',len({x['parameters'] for x in variants})==1)
    extensions=list((ROOT/'results/raw').glob('p2b_associative_*_s2026.json'))
    if (ROOT/'configs/p2b.json').exists():
        check('common associative continuation complete',len(extensions)==3)
        extended=[json.loads(p.read_text()) for p in extensions]
        check('common continuation same data stream',len({d['train_data_sha256'] for d in extended})==1)
        for d in extended:
            method=d['config']['method'];name=f'p2b_associative_{method}_s2026'
            check(name+' equal total/additional token budget',d['train_tokens']==307200000 and d['additional_tokens']==245760000 and d['optimizer_steps']==60000)
            check(name+' common fresh optimizer',d['config']['optimizer_reset'])
            check(name+' complete evaluation',{r['length'] for r in d['bf16']}=={256,512,1024,2048} and {1333,1334,1335}<={r['distance'] for r in d['bf16'] if r['length']==2048})
            check(name+' all PTQ conditions',all(any(q['length']==r['length'] and q['distance']==r['distance'] and q['group']==group and q['scope']==scope for q in d['ptq']) for r in d['bf16'] for group in [128,16] for scope in ['qk','k']))
            check(name+' checkpoints retained',all((ROOT/f'results/checkpoints/{name}_{suffix}.pt').exists() for suffix in ['mid','final']))
            if method=='vortex':check(name+' mapped spectral FP32 equivalence',d['spectral_fp32_equivalence']['relative_l2']<2e-5)
        check('continuation real QK mechanism tested',(ROOT/'results/raw/p1_real_associative_extended.json').exists())
    if (ROOT/'configs/p2c.json').exists():
        gate=load('p2c_gate.json')
        control=load('p2c_associative_rope_safe_s2026.json')
        conditions=[r for r in control['bf16'] if r['length'] in [256,512] and r['distance'] in [8,32,128]]
        avg=sum(r['accuracy'] for r in conditions)/len(conditions)
        minimum=min(r['accuracy'] for r in conditions)
        check('four-layer gate recomputed from six BF16 conditions',len(conditions)==6 and gate['mean_accuracy']==avg and gate['minimum_accuracy']==minimum and gate['passed']==(avg>=.9 and minimum>=.8))
        calibration=[json.loads(p.read_text()) for p in (ROOT/'results/raw').glob('p2c_associative_*_s2026.json')]
        expected={'rope_safe','vortex','ring_plain'} if gate['passed'] else {'rope_safe'}
        check('four-layer calibration respects frozen conditional scope',{d['config']['method'] for d in calibration}==expected)
        check('four-layer calibration same data stream and parameter count',len({d['train_data_sha256'] for d in calibration})==1 and len({d['parameters'] for d in calibration})==1)
        for d in calibration:
            name=f"p2c_associative_{d['config']['method']}_s2026"
            check(name+' fixed four-layer budget',d['train_tokens']==61440000 and d['optimizer_steps']==12000 and d['model_config']['layers']==4)
            check(name+' evaluation and checkpoints',{r['length'] for r in d['bf16']}=={256,512,1024,2048} and all((ROOT/f'results/checkpoints/{name}_{suffix}.pt').exists() for suffix in ['mid','final']))
            check(name+' all PTQ conditions',all(any(q['length']==r['length'] and q['distance']==r['distance'] and q['group']==g and q['scope']==scope for q in d['ptq']) for r in d['bf16'] for g in [128,16] for scope in ['qk','k']))
        if gate['passed']:
            check('four-layer real QK mechanism measured',(ROOT/'results/raw/p1_real_associative_fourlayer.json').exists())
            check('four-layer spectral mapping',next(d for d in calibration if d['config']['method']=='vortex')['spectral_fp32_equivalence']['fp32_output_max_abs']<2e-4)
    p4=load('p4.json')
    replay=load('attention_replay.json')
    check('actual causal attention replay spectral equivalence',all(r['relative_l2']<2e-5 for r in replay['fp32_equivalence']))
    check('actual causal replay learned relative and extended associative',{'p2_relative_vortex_s2026','p2b_associative_vortex_s2026'}<={r['source'] for r in replay['rows']})
    check('actual causal replay full shapes/groups/scopes',all(any(r['source']==source and r['length']==length and r['method']==method and r['group']==group and r['scope']==scope for r in replay['rows']) for source in {r['source'] for r in replay['rows']} for length in [256,512,2048] for method in ['rope_safe','vortex_pre','vortex_post','spectral_rope'] for group in [128,16] for scope in ['qk','k']))
    check('P4 full prefill/decode coverage',all(any(r['method']==method and r['sequence']==length and r['path']==path and len(r['wall_samples_ms'])>=30 and r['gpu_event_median_ms']>0 for r in p4['rows']) for method in ['rope_safe','vortex'] for length,path in [(128,'position_qk'),(512,'position_qk'),(2048,'position_qk'),(128,'prefill_attention'),(512,'prefill_attention'),(2048,'prefill_attention'),(512,'decode_attention'),(2048,'decode_attention')]))
    gate=load('p3_gate.json');check('P3 conditional disposition recorded',gate['decision']=='skip' and bool(gate['reason']) and not gate['executed'])
    check('P3 gate references calibration and extended mechanism',all(any(name in p for p in gate['evidence_files']) for name in ['p2c_gate.json','p2b_associative_vortex','p1_real_associative_extended']))
    ledger=[json.loads(x) for x in (ROOT/'results/raw/gpu_ledger.jsonl').read_text().splitlines()]
    check('all executed stages succeeded',all(x['success'] for x in ledger))
    seconds=sum(x['seconds'] for x in ledger)
    check('36h GPU budget respected',seconds<36*3600)
    check('no overlapping GPU stages',all(a['end_unix']<=b['start_unix'] for a,b in zip(sorted(ledger,key=lambda r:r['start_unix']),sorted(ledger,key=lambda r:r['start_unix'])[1:])))
    for name in ['PREEXP_REPORT.md','DECISION.md','RESULTS_TABLE.csv']:
        check('deliverable '+name,(ROOT/'reports'/name).exists() and (ROOT/'reports'/name).stat().st_size>100)
    report=(ROOT/'reports/PREEXP_REPORT.md').read_text()
    required=['研究问题','方法与基线','实验设置','核心结果表与曲线','是否证明了什么','未证明什么','下一步建议']
    positions=[report.find('## '+s) for s in required]
    check('required report structure',all(p>=0 for p in positions) and positions==sorted(positions))
    import re
    check('report figures resolve',all((ROOT/'reports'/path).exists() for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)))
    result=dict(passed=not failures,checks=checks,failures=failures,gpu_occupied_seconds=seconds,gpu_hours=seconds/3600,
                note='This verifier checks evidence completeness; scientific conclusions require reading raw results and report.')
    write_json('reports/COMPLETION_AUDIT.json',result)
    manifest=[]
    for file in sorted(ROOT.rglob('*')):
        if not file.is_file() or '__pycache__' in str(file) or file.suffix=='.pid' or file.name in ['ARTIFACT_MANIFEST.json','finalize.log']:continue
        h=hashlib.sha256()
        with file.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        manifest.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=h.hexdigest()))
    write_json('reports/ARTIFACT_MANIFEST.json',dict(files=manifest,excluded='__pycache__, PID files, this manifest and the actively written nonexperimental finalize.log wrapper. Experimental raw data and logs are included.'))
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
