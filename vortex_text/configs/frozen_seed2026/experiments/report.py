"""CPU-only report and paired contiguous-block bootstrap; no torch dependency."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw'
OUT = ROOT / 'reports'
METHODS = ['rope_safe', 'vortex', 'spectral_rope']


def read(name):
    return json.loads((RAW / name).read_text())


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8 << 20),b''):
            digest.update(part)
    return digest.hexdigest()


def block_interval(terms, repetitions=10000):
    """Paired sample uncertainty, not random-seed uncertainty; ~8192-token blocks."""
    vectors = []
    weights = None
    for coefficient, condition in terms:
        seq = condition['sequences']
        w = np.array([s['tokens'] for s in seq], dtype=np.float64)
        assert weights is None or np.array_equal(w, weights)
        weights = w
        vectors.append(coefficient*np.array([s['nll'] for s in seq])*w)
    total = sum(vectors)
    group = max(1, 8192//terms[0][1]['context'])
    numerator = np.array([total[i:i+group].sum() for i in range(0, len(total), group)])
    denominator = np.array([weights[i:i+group].sum() for i in range(0, len(weights), group)])
    rng = np.random.default_rng(20261111)
    indices = rng.integers(len(numerator), size=(repetitions, len(numerator)))
    means = numerator[indices].sum(1)/denominator[indices].sum(1)
    return dict(estimate=float(total.sum()/weights.sum()),
                low=float(np.quantile(means,.025)), high=float(np.quantile(means,.975)),
                blocks=len(numerator), block_tokens=8192, repetitions=repetitions)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not all((RAW / f'ptq_{m}_s2026.json').exists() for m in METHODS):
        status = dict(updated_unix=time.time(), complete=False)
        for m in ['rope_safe','vortex']:
            path = RAW / f'train_{m}_s2026_curve.jsonl'
            if path.exists():
                status[m] = json.loads(path.read_text().splitlines()[-1])
        (OUT / 'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status))
        return
    reports = {m: read(f'ptq_{m}_s2026.json') for m in METHODS}
    conditions = {(m,c['context'],c['condition']):c for m,r in reports.items() for c in r['conditions']}
    assert len(conditions) == 33, 'Wait for every full-validation condition to finish'
    rows = []
    for (m, context, label), c in conditions.items():
        base = conditions[m,context,'bf16']
        interval = block_interval([(1,c),(-1,base)])
        rows.append(dict(method=m,context=context,condition=label,tokens=c['tokens'],
            nll=c['nll'],perplexity=c['perplexity'],quant_penalty=c['nll']-base['nll'],
            penalty_ci_low=interval['low'],penalty_ci_high=interval['high'],
            elapsed_seconds=c['elapsed_seconds']))
    with (OUT/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    comparisons = {}
    for context in [512,256,128]:
        vb=conditions['vortex',context,'bf16']
        rb=conditions['rope_safe',context,'bf16']
        sb=conditions['spectral_rope',context,'bf16']
        comparisons[f'context{context}_vortex_minus_rope_bf16']=block_interval([(1,vb),(-1,rb)])
        comparisons[f'context{context}_spectral_minus_vortex_bf16']=block_interval([(1,sb),(-1,vb)])
        for label in ['qk_int4_head','qk_int4_group16']+(['k_int4_head','qk_int4_ring7'] if context==512 else []):
            v=conditions['vortex',context,label]; r=conditions['rope_safe',context,label]; s=conditions['spectral_rope',context,label]
            comparisons[f'context{context}_{label}_absolute_vortex_minus_rope']=block_interval([(1,v),(-1,r)])
            comparisons[f'context{context}_{label}_penalty_vortex_minus_rope']=block_interval([(1,v),(-1,vb),(-1,r),(1,rb)])
            comparisons[f'context{context}_{label}_penalty_vortex_minus_spectral']=block_interval([(1,v),(-1,vb),(-1,s),(1,sb)])
    (OUT/'COMPARISONS.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    protocol={m:read(f'train_{m}_s2026_protocol.json')['config'] for m in ['rope_safe','vortex']}
    checks=[]
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
    for field in ['seed','batch','context','steps','target_tokens','lr','min_lr','warmup_steps','scheduler','betas',
                  'weight_decay_matrix','weight_decay_vector','grad_clip','stream_sha256','consumed_prefix_sha256',
                  'tokenizer_sha256','initial_state_sha256','parameters','supervision','data_order']:
        check('matched_'+field,protocol['rope_safe'][field]==protocol['vortex'][field])
    for m in ['rope_safe','vortex']:
        final=read(f'train_{m}_s2026.json')
        check(m+'_full_token_budget',final['protocol']['target_tokens']==500006912)
        check(m+'_full_validation',final['validation']['tokens']==255095)
        trace=[json.loads(line) for line in (RAW/f'train_{m}_s2026_curve.jsonl').read_text().splitlines()]
        check(m+'_finite_training',all(np.isfinite(r['loss']) and np.isfinite(r['grad_norm']) for r in trace))
    check('FP32_equivalence',read('fp32_equivalence.json')['passed'])
    check('permutation_quantization_commutation',read('text_commutation.json')['passed'])
    check('all_33_conditions_full_validation',all(c['tokens']==255095 for c in conditions.values()))
    check('all_conditions_finite',all(np.isfinite(c['nll']) for c in conditions.values()))
    check('derived_control_no_training',reports['spectral_rope']['source_checkpoint']==reports['vortex']['source_checkpoint'])
    check('shifted_cyclic_stream_checks',read('stream_shift_checks.json')['passed'])
    check('checkpoint_restore_and_evaluation_smoke',read('eval_pipeline_smoke.json')['passed'])
    runner=read('train_runner.json')
    check('runner_success',all(j.get('exit_code')==0 for j in runner['jobs']) and 'completed_unix' in runner)
    ledger=[json.loads(line) for line in (RAW/'gpu_ledger.jsonl').read_text().splitlines()]
    seconds=sum(r['seconds'] for r in ledger)
    check('cumulative_under_36h',seconds<36*3600,seconds)
    audit=dict(passed=sum(c['passed'] for c in checks),total=len(checks),checks=checks)
    (OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Vortex 真实文本预实验','',
        '## 研究问题','',
        '带符号置换位置算子能否在保留语言建模能力时，降低 Q/K INT4 量化带来的实际损失？本轮是合成检索修复后的探索性 P3，不将先前混合的 P1 证据重新表述为明确阳性。','',
        '## 方法与基线','',
        'RoPE 使用 FP64 构造角度的安全实现；Vortex 保留原始六环及 16 个不变维度。spectral_rope 由同一 Vortex checkpoint 离线换基，未另行训练。三者分别使用自己的 BF16 NLL 计算量化增量；FP32 等价不意味着 BF16 舍入后输出逐位相同。','',
        '## 实验设置','',
        f'WikiText-103 raw 标准 train/validation 切分，训练集独立拟合 16,384 byte BPE。6 层、d=512、4 heads、FFN=2048、tied embedding，共 {protocol["vortex"]["parameters"]:,} 参数；context 512，batch 16，seed 2026。两条训练初始化及数据顺序相同，各训练 500,006,912 token，约 4.098 次训练流。仅使用普通 next-token CE，未使用合成检索的辅助监督。预算根据 pilot 在主训练开始前冻结。','',
        'AdamW lr 5e-4、warmup 200 steps、余弦降至 5e-5、betas=(.9,.95)、矩阵 weight decay=.1、clip=1。MTT S4000、MUSA SDPA、BF16 autocast / FP32 权重。验证集全部 255,095 个 next-token 目标参与评测，不与训练流拼接。PPL 是该 BPE 下的 token PPL，不能直接与公开的 word-level WikiText PPL 比较。','',
        '## 核心结果表与曲线','',
        '|方法|条件（context 512）|NLL|PPL|量化增量|',
        '|---|---|---:|---:|---:|']
    for r in rows:
        if r['context']==512:
            lines.append(f'|{r["method"]}|{r["condition"]}|{r["nll"]:.6f}|{r["perplexity"]:.3f}|{r["quant_penalty"]:+.6f}|')
    lines+=['','完整的 context 128/256/512、位置分桶及逐序列损失见原始 JSON，全部聚合结果见 RESULTS.csv。训练和验证曲线见 training_curves.png，量化增量见 quant_penalty.png。','',
        '## 是否证明了什么','',
        f'FP32 整模型等谱校验：{read("fp32_equivalence.json")["passed"]}；Vortex per-head/ring7 先量化与后量化整模型逐位一致：{read("text_commutation.json")["passed"]}。绝对质量与量化增量均单独列出，避免只看较小增量而忽略 BF16 质量差距。','',
        'COMPARISONS.json 给出相同目标 token 上的配对 8,192-token 连续块 bootstrap（10,000 次）。它反映这份验证文本的采样不确定性，不估计训练种子方差；单种子不能证明训练稳定性。','',
        '## 未证明什么','',
        '所有低比特评测都是对称 [-7,7] INT4 fake quant，attention 仍在 BF16/FP32 执行。没有原生 4-bit 速度结论。head/group16/ring7 分别使用 1/8/7 个 FP32 尺度，每种格式在各方法之间使用相同分组预算；ring7 不能称为原生 group16。group16 不具有置换交换律，既有失败证据仍然成立。','',
        '此前合成任务依赖更强训练监督才学会，不能改写为 answer-only 成功；其满分 PTQ 仍有上限效应，不能据此推断文本收益。','',
        '## 下一步建议','',
        '最终研究判断需同时核对：Vortex 的 BF16 绝对 NLL、相对标准 RoPE 的量化后绝对 NLL、相对等谱旋转的量化增量，以及单种子限制。不得仅凭精确交换律作 GO 判断。','',
        f'本阶段结束后累计设备占用 {seconds/3600:.3f} h（含全部此前实验）；本地审计 {audit["passed"]}/{audit["total"]} 通过。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m in ['rope_safe','vortex']:
        trace=[json.loads(line) for line in (RAW/f'train_{m}_s2026_curve.jsonl').read_text().splitlines()]
        axes[0].plot([x['tokens']/1e6 for x in trace],[x['loss'] for x in trace],label=m,alpha=.7)
        vals=[json.loads(p.read_text()) for p in RAW.glob(f'train_{m}_s2026_valid_step*.json')]
        vals.sort(key=lambda x:x['training_tokens'])
        axes[1].plot([x['training_tokens']/1e6 for x in vals],[x['nll'] for x in vals],label=m)
    axes[0].set_ylabel('Train NLL (logged batches)'); axes[1].set_ylabel('Fixed 32,768-token validation NLL')
    for ax in axes:
        ax.set_xlabel('Training tokens (M)'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'training_curves.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    labels=['qk_int4_head','qk_int4_group16','k_int4_head','qk_int4_ring7']
    for j,m in enumerate(METHODS):
        selected=[next(r for r in rows if r['method']==m and r['context']==512 and r['condition']==label) for label in labels]
        ax.bar(np.arange(4)+(j-1)*.25,[r['quant_penalty'] for r in selected],width=.25,label=m)
    ax.set_xticks(np.arange(4),labels); ax.set_ylabel('NLL(PTQ) - NLL(BF16)'); ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'quant_penalty.png',dpi=180); plt.close(fig)
    manifest={}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.tmp':
            continue
        relative=str(path.relative_to(ROOT))
        if relative.startswith('reports/') and path.name in ['MANIFEST.json','mirror.log','MIRROR_STATUS.json']:
            continue
        manifest[relative]=dict(bytes=path.stat().st_size,sha256=file_hash(path))
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,comparisons=comparisons),indent=2))
    assert audit['passed']==audit['total'], audit


if __name__=='__main__':
    main()
