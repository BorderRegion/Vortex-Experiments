"""Summarize the two prespecified seeds without treating text blocks as seeds."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    records=[]
    for seed in [2026,2027]:
        for method in ['rope_safe','vortex','spectral_rope']:
            path=ROOT/f'results/raw/ptq_{method}_s{seed}.json'
            if not path.exists():
                print('Two-seed summary pending:',path.name)
                return
            report=json.loads(path.read_text())
            assert len(report['conditions'])==11
            for c in report['conditions']:
                if c['context']==512:
                    baseline=next(b['nll'] for b in report['conditions'] if b['context']==512 and b['condition']=='bf16')
                    records.append(dict(seed=seed,method=method,condition=c['condition'],nll=c['nll'],
                        perplexity=c['perplexity'],quant_penalty=c['nll']-baseline,
                        tokens=c['tokens'],trained_tokens=report['trained_tokens']))
    out=ROOT/'reports'
    with (out/'TWO_SEED_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    def row(seed,method,condition):
        return next(r for r in records if r['seed']==seed and r['method']==method and r['condition']==condition)
    comparisons=[]
    for seed in [2026,2027]:
        for label in ['bf16','qk_int4_head','qk_int4_group16','k_int4_head','qk_int4_ring7']:
            r,v,s=[row(seed,m,label) for m in ['rope_safe','vortex','spectral_rope']]
            comparisons.append(dict(seed=seed,condition=label,absolute_vortex_minus_rope=v['nll']-r['nll'],
                penalty_vortex_minus_rope=v['quant_penalty']-r['quant_penalty'],
                penalty_vortex_minus_spectral=v['quant_penalty']-s['quant_penalty']))
    (out/'TWO_SEED_COMPARISONS.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    head=[c for c in comparisons if c['condition']=='qk_int4_head']
    wins=sum(c['absolute_vortex_minus_rope']<0 for c in head)
    group=[c for c in comparisons if c['condition']=='qk_int4_group16']
    group_wins=sum(c['absolute_vortex_minus_rope']<0 for c in group)
    lines=['# 两个训练种子的真实文本结果','',
        '两个种子（2026、2027）均使用相同的冻结模型、数据、优化器、500,006,912-token 预算和完整验证集。每个种子内 RoPE/Vortex 初始权重相同；种子之间改变初始化。数据顺序保持相同。spectral_rope 各自由对应 Vortex checkpoint 换基，未额外训练。','',
        '|种子|方法|BF16 NLL|QK INT4 per-head NLL|per-head 量化增量|QK group16 NLL|',
        '|---|---|---:|---:|---:|---:|']
    for seed in [2026,2027]:
        for m in ['rope_safe','vortex','spectral_rope']:
            b=row(seed,m,'bf16');h=row(seed,m,'qk_int4_head');g=row(seed,m,'qk_int4_group16')
            lines.append(f'|{seed}|{m}|{b["nll"]:.6f}|{h["nll"]:.6f}|{h["quant_penalty"]:+.6f}|{g["nll"]:.6f}|')
    lines+=['',f'per-head 下 Vortex 的量化后绝对 NLL 在 {wins}/2 个种子优于标准 RoPE；group16 为 {group_wins}/2。',
        f'per-head 的两种子平均 Vortex−RoPE NLL 差为 {np.mean([c["absolute_vortex_minus_rope"] for c in head]):+.6f}；每个种子的原始结果均保留，两个种子不足以精确估计训练方差。','']
    if wins==2:
        lines+=['per-head 的最终质量收益在两个训练种子上重复，支持继续研究这一具体场景。BF16 的代价、group16 的结果和等谱对照仍需同时报告；不能外推为通用低比特优势。']
    elif wins==1:
        lines+=['per-head 的最终质量收益未在两个种子中一致重复，说明训练变异足以影响这一级别的收益。目前应保留量化机制信号，并将论文方向评为 MODIFY。']
    else:
        lines+=['这两个种子未建立 per-head 的量化后绝对质量优势，不能只用更小量化增量来主张方法胜出。']
    lines+=['','所有 INT4 均为 fake quant；没有原生 4-bit 性能结论。每种格式的尺度预算在各方法之间保持一致。逐种子的配对文本块 bootstrap 反映验证文本采样，不替代训练种子方差。',
        '完整结果见 TWO_SEED_RESULTS.csv 和 TWO_SEED_COMPARISONS.json；逐种子报告分别为 REPORT.md 与 seed2027/REPORT.md。','']
    (out/'TWO_SEED_REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    main()
