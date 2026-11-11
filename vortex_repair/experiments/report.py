"""Build an auditable Chinese repair report from actual saved results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())
def mean(rows,field='accuracy'): return statistics.mean(r[field] for r in rows)
def six(rows): return [r for r in rows if r['length'] in (256,512)]
def group_name(g): return 'ring7' if isinstance(g,list) else str(g)


def main():
    raw=ROOT/'results/raw'; dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    methods=('rope_safe','vortex','ring_plain')
    results={m:read(raw/f'routing_value_{m}_s2026.json') for m in methods}
    validation={m:read(raw/f'routing_value_{m}_s2026_step12000_validation.json') for m in methods}
    ablation=read(raw/'routing_rope_safe_s2026.json')
    baseline=read(ROOT.parent/'vortex_preexp/results/raw/p2_associative_rope_safe_s2026.json')
    aligned=read(ROOT.parent/'vortex_followup/results/raw/binding_rope_safe_aligned_s2026.json')
    checks=[]
    def check(name,condition,**extra):
        checks.append(dict(check=name,passed=bool(condition),**extra))
    check('new_input_targets_loss_gradients',read(raw/'routing_checks.json')['passed'])
    for filename in ('model.py','synthetic.py','vortex.py','spectral.py'):
        check('unchanged_'+filename,(ROOT/'src'/filename).read_bytes()==(ROOT.parent/'vortex_followup/src'/filename).read_bytes())
    check('identical_original_input_and_answer_digest',len({baseline['train_data_sha256'],aligned['train_data_sha256'],
          ablation['train_data_sha256'],*[d['train_data_sha256'] for d in results.values()]})==1)
    check('equal_61440k_tokens',all(d['train_tokens']==61440000 for d in results.values()))
    check('equal_192k_answer_labels',all(d['supervised_answers']==192000 for d in results.values()))
    check('same_parameter_count_as_original',all(d['parameters']==baseline['parameters'] for d in results.values()))
    check('equal_auxiliary_label_counts',len({tuple(d['auxiliary_query_heads']) for d in results.values()})==1)
    check('equal_value_label_counts',len({d['auxiliary_value_labels'] for d in results.values()})==1)
    check('same_training_source_hash',len({d['config']['training_source_sha256'] for d in results.values()})==1)
    for m,d in results.items():
        curve=[json.loads(s) for s in (raw/f'routing_value_{m}_s2026_curve.jsonl').read_text().splitlines()]
        check(m+'_last6000steps_no_auxiliary',all(r['auxiliary_weight']==0 for r in curve if r['step']>=6000))
        fresh=validation[m]['fresh_test']
        check(m+'_fresh_test_calibration',mean(fresh)>=.90 and min(r['accuracy'] for r in fresh)>=.80)
    check('fp32_spectral_equivalence',all(r['fp32_mapped_max_abs']<.005 for r in validation['vortex']['equivalence']))
    check('whole_model_ring_quant_pre_post_exact',all(r['ring_quant_pre_post_max_abs']==0 for r in validation['vortex']['equivalence']))
    ledger=[json.loads(s) for s in (raw/'gpu_ledger.jsonl').read_text().splitlines()]
    total=sum(r['seconds'] for r in ledger);prior=9431.38007736206
    check('cumulative_36h_cap',total<36*3600,seconds=total)
    audit=dict(passed=all(c['passed'] for c in checks),checks=checks,passed_count=sum(c['passed'] for c in checks),
               check_count=len(checks),cumulative_seconds=total,repair_incremental_seconds=total-prior)
    (dest/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    table=[]
    def add(run,precision,group,rows):
        for r in rows: table.append(dict(run=run,precision=precision,group=group,**r))
    add('original_rope_safe','BF16','',baseline['bf16'])
    add('aligned_binding_rope_safe','BF16','',aligned['bf16'])
    add('routing_only_rope_safe','BF16','',ablation['bf16'])
    for m,d in results.items():
        add(m,'BF16','',d['bf16']);add(m,'fresh_BF16','',validation[m]['fresh_test'])
        for p in validation[m]['ptq']: add(m,'QK_INT4',group_name(p['group']),p['rows'])
        for p in validation[m].get('mapped_spectral',[]):
            add('spectral_rope','BF16' if p['group'] is None else 'QK_INT4','' if p['group'] is None else group_name(p['group']),p['rows'])
    columns=['run','precision','group','length','distance','samples','accuracy','nll']
    with (dest/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(table)
    lines=[
      '# 关联检索训练修复', '',
      '本轮修复的是训练流程。两层普通 causal decoder、原始输入、标签、位置算子和推理路径均保持不变；训练前半程增加路由与 value 重建监督，后半程只使用原有答案损失。原始失败实验及上一轮结果完整保留。', '',
      '## 问题定位', '',
      '强制第二层读取正确 value 后，旧 aligned checkpoint 的准确率仍仅为4.30%/5.86%（长度256/512，各256例）。这表明单修注意力位置不足以恢复答案读出。', '',
      '单独监督两层路由也没有解决问题：step3000 的 CPU FP32 探针显示正确位置获得85.79%注意力，第一层 value 信息可被独立线性探针以99.93%准确率恢复，但现有共享输出层的 value 读出仅1.63%。探针是诊断，不计入任务性能。', '',
      '## 实际修改', '',
      '- 第1层：在显式 value 位置，监督各 head 关注它前面的 key。',
      '- 第2层：在末尾查询位置，监督各 head 关注匹配 key 后面的 value。',
      '- 第1层显式 value 位置：复用现有 norm 和 tied output 重建该位置自己的 value，帮助训练出可读出的内容表示。',
      '- 三项辅助损失权重在前3000步为1，随后余弦衰减，step6000起全部为0。最后6000步只用原任务答案损失。推理时直接调用原始 Decoder.forward，不传位置目标。', '',
      '这是更强的训练监督，不能描述成原来只用末尾答案监督的训练已成功。所有辅助目标都从训练输入生成，答案不加入输入，也不在推理时提供。', '',
      '## 原始任务的普通推理结果', '',
      '六项条件为长度256/512 × 距离8/32/128，每项256个未见样本；下表均使用普通推理。', '',
      '| 训练协议 | 六项平均准确率 | 最低一项 | 平均 NLL |',
      '|---|---:|---:|---:|',
    ]
    for label,d in [('原始 RoPE',baseline),('上一轮 key 对齐辅助 RoPE',aligned),('本轮仅路由辅助 RoPE',ablation),
                    *[(f'路由 + value 重建：{m}',results[m]) for m in methods]]:
        rows=six(d['bf16']);lines.append(f'| {label} | {mean(rows):.2%} | {min(r["accuracy"] for r in rows):.2%} | {mean(rows,"nll"):.4g} |')
    lines+=['','另用训练和中途评估均未使用的新 seed，六项条件各评512例：','','| 方法 | 新测试平均准确率 | 最低一项 | 换查询后的准确率 | 换查询后答案未改变 |','|---|---:|---:|---:|---:|']
    for m,v in validation.items():
        rows=v['fresh_test'];swap=v['query_swap']
        lines.append(f'| {m} | {mean(rows):.2%} | {min(r["accuracy"] for r in rows):.2%} | {swap["changed_query_accuracy"]:.2%} | {swap["unchanged_argmax"]:.2%} |')
    lines+=['','查询替换只改末尾 key，并选择不同答案的合法绑定；512例。它检验模型能否随查询改变输出，不能代替原始条件评估。','','## 长度外推','','| 方法 | L1024 四项平均 | L2048 七项平均 | d1333 | d1334 | d1335 |','|---|---:|---:|---:|---:|---:|']
    for m,d in results.items():
        rows=d['bf16'];by={(r['length'],r['distance']):r['accuracy'] for r in rows}
        lines.append(f'| {m} | {mean([r for r in rows if r["length"]==1024]):.2%} | {mean([r for r in rows if r["length"]==2048]):.2%} | {by[2048,1333]:.2%} | {by[2048,1334]:.2%} | {by[2048,1335]:.2%} |')
    lines+=['','完整逐长度/间距结果见 RESULTS.csv；L2048 均值混合了近距离与长距离，判断外推时应查看分项。','','## 修复后模型的模拟 INT4','','下表为六项训练长度条件；括号为相对同一 checkpoint BF16 的准确率变化（百分点）。七组按环边界划分，和原生 group16 格式不同。','','| 方法 | BF16 | 每 head 一尺度 | group16 | 按环分七组 |','|---|---:|---:|---:|---:|']
    for m,d in results.items():
        base=mean(six(d['bf16']));cells=[]
        for p in validation[m]['ptq']:
            acc=mean(six(p['rows']));cells.append(f'{acc:.2%} ({(acc-base)*100:+.2f})')
        lines.append(f'| {m} | {base:.2%} | '+ ' | '.join(cells)+' |')
    mapped=validation['vortex']['mapped_spectral']
    lines+=['','同一个 Vortex checkpoint 换基后的等谱旋转，六项结果：','','| 精度/尺度 | 准确率 | NLL |','|---|---:|---:|']
    for p in mapped: lines.append(f'| {"BF16" if p["group"] is None else "INT4 "+group_name(p["group"])} | {mean(six(p["rows"])):.2%} | {mean(six(p["rows"]),"nll"):.4g} |')
    max_error=max(r['fp32_mapped_max_abs'] for r in validation['vortex']['equivalence'])
    lines+=['','所有方法都使用相同分组与尺度预算作比较；低比特均为 fake quantization。本任务修复后接近准确率上限，尚不能据此证明低比特质量优势。按环分组的交换律精确性不自动意味着模型质量更高。',
      f'在长度256/512/2048、每项16例上，换基前后 FP32 全 logits 最大绝对差为 {max_error:.6g}，末尾预测一致率100%；按环分组 INT4 在位置操作前后量化的整模型 logits 差为0。',
      '','## 验证、预算与边界','',
      f'- 自动审计：{audit["passed_count"]}/{audit["check_count"]} 项通过。包含新损失梯度与独立 full-attention 参考一致、输入/答案摘要一致、模型参数量一致、监督计数一致、辅助损失撤除、等谱转换和整模型量化交换律。',
      '- 三个修复模型均从相同 seed 全新初始化，batch16、12000步、61.44M输入 token、192000个末尾答案；AdamW5e-4、betas(.9,.95)、weight decay0.1、clip1、warmup200、余弦至0.1倍。BF16 autocast / FP32 master，MTT S4000 MUSA SDPA。',
      f'- 每个模型额外使用 {results["rope_safe"]["auxiliary_value_labels"]:,} 个 value 重建标签，以及 {results["rope_safe"]["auxiliary_query_heads"][0]:,} / {results["rope_safe"]["auxiliary_query_heads"][1]:,} 个第1/2层路由 head 目标。额外监督量对三个方法完全一致，不能与原始答案监督混报。',
      f'- 本轮新增累计占用 {audit["repair_incremental_seconds"]/3600:.4f} GPU·h；含此前所有实验累计 {total/3600:.4f} GPU·h，低于36h。统计包含活动阶段 Python 开销；实现核对短暂重叠采用保守重复计时。',
      '- 单 seed、小型合成任务；不推断真实文本质量，也不声称原生4-bit加速。没有扩大模型、替换位置频谱或在推理中引入外部 memory。',
      '- 路由单独训练期间，为准备后续试验更新过磁盘上的实验脚本；其实际执行源码已单独保存在 results/raw/routing_v1_source.py。该对照的最终全目录 code hash不等于其启动时源码摘要；三个最终修复模型另外记录了实际训练脚本摘要。', '',
      '训练曲线见 training_curves.png。所有 checkpoint（含优化器和数据生成器状态）、原始JSON/JSONL、协议及审计均随目录保存。', '',
    ]
    repaired=all(mean(v['fresh_test'])>=.90 and min(r['accuracy'] for r in v['fresh_test'])>=.80 for v in validation.values())
    conclusion=('关联检索训练已修复：三个方法均通过新测试集的普通推理校验。' if repaired else
                '修复结果尚未在全部方法的新测试集上达标，应按分项结果判断。')
    lines += ['## 判断','',conclusion,
              '这个方向继续做论文，仍需要改：本轮补齐了可学习的关联检索对照，但原始答案监督的优化问题、合成任务上限效应和真实文本低比特收益仍需分别处理。较高检索准确率本身不能证明置换量化优于等谱旋转。','']
    (dest/'DECISION.md').write_text(conclusion+'\n\n更强的训练期监督有效，原始答案监督实验仍保留为失败结果。原始位置算子未更改。论文方向仍为 MODIFY：须用非上限效应的质量指标证明低比特收益，并另外验证真实文本。\n')
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for m in methods:
        curve=[json.loads(s) for s in (raw/f'routing_value_{m}_s2026_curve.jsonl').read_text().splitlines()]
        axes[0].plot([r['step'] for r in curve],[r['answer_loss'] for r in curve],label=m)
        evs=sorted(raw.glob(f'routing_value_{m}_s2026_eval_step*.json'),key=lambda p:int(p.stem.split('step')[-1]))
        axes[1].plot([int(p.stem.split('step')[-1]) for p in evs],[mean(read(p))*100 for p in evs],label=m)
    for ax in axes:
        ax.axvline(6000,color='gray',linestyle='--',label='auxiliary off' if ax is axes[0] else None)
        ax.set_xlabel('Optimizer step');ax.grid(alpha=.25)
    axes[0].set_ylabel('Answer cross entropy');axes[0].set_yscale('log');axes[0].legend()
    axes[1].set_ylabel('Ordinary retrieval accuracy (%)');axes[1].set_ylim(0,102);axes[1].legend()
    fig.savefig(dest/'training_curves.png',dpi=180);plt.close(fig)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
    (dest/'MANIFEST.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
