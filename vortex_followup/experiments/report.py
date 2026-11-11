"""CPU-only follow-up report from saved evidence; never edit original results."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.common import ROOT, write_json

ORIGINAL = Path('/root/autodl-tmp/vortex_preexp')


def read(path):
    return json.loads(path.read_text())


def six(rows):
    rows = [r for r in rows if r['length'] in (256,512) and r['distance'] in (8,32,128)]
    assert len(rows) == 6
    return sum(r['accuracy'] for r in rows)/6, min(r['accuracy'] for r in rows), sum(r['nll'] for r in rows)/6


def main():
    raw = ROOT/'results/raw'; report = ROOT/'reports'; report.mkdir(exist_ok=True)
    completed = read(raw/'stage2_complete.json')
    ring = read(raw/'ring_groups.json'); swap = read(raw/'query_swap.json')
    runs = {p.stem:read(p) for p in sorted(raw.glob('supervision_*.json')) if '_eval_step' not in p.name and p.stem != 'supervision_oracle'}
    binding = {p.stem:read(p) for p in sorted(raw.glob('binding_*_s2026.json'))}
    runs.update(binding)
    binding_stage = read(raw/'binding_stage_complete.json')
    alignment_stage = read(raw/'alignment_stage_complete.json')
    binding_probe = read(raw/'binding_probe.json')['rows']+read(raw/'binding_probe_aligned.json')['rows']
    if (raw/'query_swap_binding.json').exists():
        swap['rows'] += read(raw/'query_swap_binding.json')['rows']
    swap['rows'] += read(raw/'query_swap_aligned.json')['rows']
    baseline_names = ['p2_associative_rope_safe_s2026', 'p2b_associative_rope_safe_s2026', 'p2c_associative_rope_safe_s2026']
    old = {n:read(ORIGINAL/f'results/raw/{n}.json') for n in baseline_names}
    flatdense = runs['supervision_rope_safe_dense_s2026']
    flatsparse = runs['supervision_rope_safe_sparse_s2026']
    ledger = [json.loads(s) for s in (raw/'gpu_ledger.jsonl').read_text().splitlines()]
    prior = [json.loads(s) for s in (ORIGINAL/'results/raw/gpu_ledger.jsonl').read_text().splitlines()]
    total_seconds = sum(r['seconds'] for r in ledger)
    extra_seconds = total_seconds-sum(r['seconds'] for r in prior)
    checks = {
        'auxiliary_answer_oracle':read(raw/'supervision_oracle.json')['passed'],
        'identical_flat_training_inputs':flatdense['train_input_sha256'] == flatsparse['train_input_sha256'],
        'identical_flat_token_budget':flatdense['train_tokens'] == flatsparse['train_tokens'] == 61440000,
        'eight_times_supervised_answers':flatdense['supervised_answers'] == 8*flatsparse['supervised_answers'],
        'same_flat_model':flatdense['model_config'] == flatsparse['model_config'],
        'original_source_copies_unchanged':all((ROOT/'src'/name).read_bytes() == (ORIGINAL/'src'/name).read_bytes()
             for name in ('synthetic.py','quant.py','vortex.py','spectral.py','common.py')),
        'model_only_quantizer_import_changed':(ROOT/'src/model.py').read_text().replace('from .ring_quant import quantize','from .quant import quantize') == (ORIGINAL/'src/model.py').read_text(),
        'ring_commutation_exact_int4_fp4':all(r['code_match'] == 1 and r['max_dequant_difference'] == 0 for r in ring['commutation'] if isinstance(r['group'],list)),
        'fp32_spectral_equivalence':all(r['relative_l2'] < 2e-5 for r in ring['fp32_equivalence']),
        'checkpoint_for_every_run':all((ROOT/f'results/checkpoints/{name}_final.pt').exists() for name in runs),
        'within_cumulative_36h_budget':total_seconds < 36*3600,
        'training_stages_finished':completed['complete'],
        'binding_stage_finished':binding_stage['complete'],
        'alignment_stage_finished':alignment_stage['complete'],
        'actual_length_full_model_gradients':read(raw/'full_gradient.json')['passed'],
        'ring_group_whole_model_pre_post_equal':all(r['pre_post_max_abs']==0 for r in read(raw/'ring_model.json')['rows']),
        'binding_uses_original_data':all(r['train_data_sha256'] == old['p2_associative_rope_safe_s2026']['train_data_sha256'] for r in binding.values()),
    }
    assert all(checks.values()), checks
    csvrows = []
    for name, data in {**old, **runs}.items():
        for r in data['bf16']:
            csvrows.append(dict(run=name, precision='bf16', group='', **r))
        if name in runs:
            for r in data['ptq']:
                csvrows.append(dict(run=name, precision='int4_qk', **r))
    with (report/'RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['run','precision','group','task','length','distance','samples','accuracy','nll'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(csvrows)

    labels = ['Original RoPE', 'Extended RoPE', '4-layer RoPE', 'Aux inputs / sparse loss', 'Aux inputs / dense loss']
    keys = baseline_names+['supervision_rope_safe_sparse_s2026','supervision_rope_safe_dense_s2026']
    allruns = {**old, **runs}
    for key in runs:
        if key not in keys:
            keys.append(key); labels.append(key.replace('supervision_','').replace('_s2026','').replace('_',' '))
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.barh(labels, [six(allruns[k]['bf16'])[0]*100 for k in keys], color='#326eaa')
    ax.axvline(90, color='#b34c36', linestyle='--', label='Calibration gate (mean)')
    ax.set(xlim=(0,100), xlabel='Accuracy (%) on original six evaluation conditions', title='Associative recall: original generator, unseen samples')
    ax.invert_yaxis(); ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(report/'accuracy.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, task in zip(axes, ('relative','associative')):
        for method, color in [('rope_safe','#777777'),('vortex','#326eaa'),('spectral_rope','#b34c36')]:
            rr = [r for r in ring['rows'] if r['task']==task and isinstance(r['group'],list) and r['method']==method]
            ax.plot([r['length'] for r in rr], [r['output_nrmse'] for r in rr], 'o-', label=method, color=color)
        ax.set(title=f'{task}: seven identical scale groups', xlabel='Context length', ylabel='Actual-V attention output NRMSE')
        ax.set_xticks([256,512,2048]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(report/'ring_groups.png',dpi=160); plt.close(fig)

    lines = ['# VortexHash 补充诊断与优化', '',
        '本轮单独保存于 `vortex_followup/`，原始实验与结论未覆盖。位置算子、频谱、模型结构均保持原样。', '',
        '## 找到了什么', '',
        '旧模型没有建立可靠的按 key 检索。删除 query 的诊断之外，本轮还把末尾 query 换成上下文中另一个确有绑定、答案不同的有效 key，观察输出是否跟随改变。这个反事实包含部分重复绑定；主准确率仍采用原始生成器。', '',
        '| checkpoint | 原 query 准确率 | 换 key 后准确率 | 预测答案不变比例 | 概率分布平均变化 TV |',
        '|---|---:|---:|---:|---:|']
    for r in swap['rows']:
        lines.append(f"| {r['checkpoint']} | {r['original_accuracy']:.2%} | {r['changed_query_accuracy']:.2%} | {r['unchanged_argmax']:.2%} | {r['query_probability_change']:.2%} |")
    lines += ['', '因此，旧检索结果主要反映尚未学会任务，不能据此判定 Vortex 的表达能力比 RoPE 弱。稀疏监督和多步配对学习是候选原因；本轮优化实验检验其作用，没有把候选原因写成已证实的唯一根因。', '',
        '## 训练优化的实测结果', '',
        '在同一原始样本的 filler 位置加入 7 个辅助 query，答案只作为标签、不进入输入。一组监督末尾答案，另一组监督全部 8 个答案；配对输入 SHA256、token 预算与模型配置完全相同。辅助 query 的 key 从其之前已有的绑定中抽取，因此辅助任务平均更靠近、更早的内容，存在比末尾检索更容易的情况。', '',
        f"输入 SHA256：`{flatdense['train_input_sha256']}`。每组 61.44M tokens；监督答案数分别为 {flatsparse['supervised_answers']:,} 和 {flatdense['supervised_answers']:,}。", '',
        '| 实验 | 六条件平均准确率 | 最差条件 | 平均 NLL | 训练 tokens |', '|---|---:|---:|---:|---:|']
    for key, label in zip(keys, labels):
        data = allruns[key]; mean, minimum, nll = six(data['bf16'])
        lines.append(f"| {label} | {mean:.2%} | {minimum:.2%} | {nll:.4f} | {data['train_tokens']:,} |")
    lines += ['', '六条件为原始生成器的长度 256/512 × 距离 8/32/128，每条件 256 个未见样本。额外保存了 1024/2048 和 1333/1334/1335，以及 INT4 per-head/group16 结果，见 `RESULTS.csv`。', '',
        '短序列课程方案固定为 3000 步长度 64、3000 步长度 128、最后 6000 步回到原始 256/512 调度，共 39.936M tokens，模型、优化器、学习率调度不变。它同时改变了训练长度分布，不能把其结果单独归因为监督密度。', '',
        f"RoPE 最终校准：均值 {completed['rope_gate']['mean']:.2%}，最差 {completed['rope_gate']['minimum']:.2%}；门槛为均值≥90%、最差≥80%。", '',
        ('该方案通过门槛，按冻结方案为 Vortex 和 ring_plain 执行同样预算；具体结果见上表。' if completed['matched_comparisons_run'] else '该方案未通过门槛，因此没有为 Vortex/ring_plain 扩大训练；检索质量比较仍未解决。密集监督降低训练 loss 或测试 NLL，不能替代检索准确率。'), '',
        '### 单独的相邻绑定辅助目标', '',
        '在密集监督及课程实验之后，另冻结 `BINDING_PLAN.json`：对第一层 value 位置的表示，用已有 norm 和 tied output head 恢复前一个 key。训练 loss 为末尾检索 CE + 0.25 × 相邻 key 恢复 CE。模型参数、推理图、输入及最终答案均不变；训练输入/标签摘要与原始 P2 完全相同。它是事后诊断指导的新实验，单独保留，不能当作先前假设的事前确认。', '',
        f"该辅助方案的 RoPE 原任务均值 {binding_stage['gate']['mean']:.2%}、最差 {binding_stage['gate']['minimum']:.2%}。" + ('按门槛执行了 Vortex/plain 同预算比较。' if binding_stage['matched_comparisons_run'] else '未通过检索门槛，没有扩大另两种方法的训练。'), '',
        '随后针对第一层共享解码器对 query key 的低恢复率，单独冻结 `ALIGNMENT_PLAN.json`：仍用 0.25 辅助权重，额外让所有 key token 位置（含末端 query）恢复自身 key，保留 value 位置的前一个 key 目标；其余数据、模型和预算不变。', '',
        f"对齐方案 RoPE 六条件均值 {alignment_stage['gate']['mean']:.2%}、最差 {alignment_stage['gate']['minimum']:.2%}。" + ('按门槛执行了 Vortex/plain 同预算比较。' if alignment_stage['matched_comparisons_run'] else '仍未通过检索门槛，没有扩大另两种方法的训练。'), '',
        '| 第一层表示检查 | 长度 | 相邻绑定 key 恢复 | query key 恢复（限制 key 词表） | 余弦检索 value 准确率 | 第二层目标 value 平均 attention 权重 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in binding_probe:
        lines.append(f"| {r['checkpoint']} | {r['length']} | {r['first_layer_binding_accuracy']:.2%} | {r['first_layer_query_key_recovery']:.2%} | {r['first_layer_cosine_value_retrieval']:.2%} | {r['second_layer_target_value_mass']:.2%} |")
    lines += ['', '对齐后，第一层 query key 与绑定 key 的恢复率均达到100%，仅在 value 位置中按第一层表示余弦检索可达约95%–96%，但整模型仍只有7.62%的六条件平均准确率。上游表示已有可用于匹配的信息，最终能力仍受后续 attention 利用和读取所限；本轮未用独立干预把 Q/K 路由与 V/readout 各自的影响分开，不能断言唯一根因。', '',
        '这些共享解码器及原始余弦探针描述了学习到的表示，不能证明所有可能的解码器都无法从隐状态恢复信息，也不能把低恢复率直接当作不可逆的信息丢失。CPU eager 与 MUSA SDPA 的整模型末端监督梯度在实际 256/512 长度、FP32/BF16 下全部通过，数据与平台错误的证据仍未出现。', '',
        '![原始任务准确率](accuracy.png)', '',
        '## 实现了按环分组的量化修正版', '',
        '分组长度为 `[11,13,17,19,23,29,16]`，每 head 使用 7 个 FP32 尺度。每个组在原始置换下封闭，因此其绝对最大值尺度保持不变；对称量化也保持符号。这修复了连续 group16 跨环移动数值造成的交换性失效。未改变任何环或位置频率。', '',
        '| 分组 | 精度 | 尺度数/head | code 一致率 | 最大反量化差 |', '|---|---|---:|---:|---:|']
    for r in ring['commutation']:
        lines.append(f"| {r['group']} | {r['kind']} | {r['scales_per_head']} | {r['code_match']:.2%} | {r['max_dequant_difference']:.6g} |")
    lines += ['', '在完全相同的 7 个组长度、尺度预算下，比较 Vortex 与等谱旋转；以下是第一层真实末端 Q、完整 K、实际 V 的输出 NRMSE。投影后的 BF16 输入提升为 FP32 再做数值诊断，和原先 replay 的定义一致。', '',
        '| checkpoint 任务 | 长度 | Vortex | 等谱旋转 | Vortex 相对误差降低 |', '|---|---:|---:|---:|---:|']
    for task in ('relative','associative'):
        for length in (256,512,2048):
            rr = {r['method']:r for r in ring['rows'] if r['task']==task and r['length']==length and isinstance(r['group'],list)}
            v, s = rr['vortex']['output_nrmse'], rr['spectral_rope']['output_nrmse']
            lines.append(f'| {task} | {length} | {v:.6f} | {s:.6f} | {1-v/s:.2%} |')
    lines += ['', '按环分组也已接入 decoder 的 `set_quant` 路径：给 `group` 传 `GROUPS` 即可。原有整数 128/16 分组委托原量化器，模型其余代码保持一致。现有已学会的 relative checkpoint 在长度256/512/2048 的完整两层推理中，pre/post 两种量化顺序的所有 logits 最大差均为0；这验证接入与交换性，不作为该天花板任务的质量优势。', '',
        '这是变量长度组的软件模拟，不等同于原生 group16 格式。各方法在该比较中使用相同的 7 个组及尺度数；与原 group16 的 8 尺度、per-head 的 1 尺度结果另列。修复交换性不保证 attention 误差或整模型准确率改善，数值结果须逐项阅读。没有测试原生低比特速度或真实文本质量。', '',
        '![按环分组的 attention 误差](ring_groups.png)', '',
        '## 可复现性与结论边界', '',
        f'本轮新增设备占用 {extra_seconds:.1f} 秒（{extra_seconds/3600:.4f} h），包含活跃阶段 Python、数据生成和评估开销；加上原实验累计 {total_seconds/3600:.4f} h，低于 36 h 上限。等待依赖阶段的调度进程不分配 GPU。', '',
        f"审计 {sum(checks.values())}/{len(checks)} 项通过，见 `AUDIT.json`。计划、门槛、每步曲线、逐条件测试和最终 checkpoint 均保留。训练优化是有限尝试；未用低 loss 冒充已学会检索。原始论文方向仍需可靠下游任务和真实文本证据，不能升级为 GO。", '',
        '运行方法见 `../README.md`。完整数值见 `RESULTS.csv`、`../results/raw/ring_groups.json`、`../results/raw/query_swap.json`。']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json('reports/AUDIT.json', dict(passed=all(checks.values()), checks=checks,
        incremental_gpu_seconds=extra_seconds, cumulative_gpu_seconds=total_seconds,
        runs=list(runs), original_results_untouched=True))
    manifest = []
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('MANIFEST.json',) and p.suffix not in ('.log','.pyc'):
            manifest.append(dict(path=str(p.relative_to(ROOT)), size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json('reports/MANIFEST.json', dict(files=manifest))
    print(json.dumps(dict(audit_passed=True, checks=len(checks), new_gpu_hours=extra_seconds/3600, cumulative_gpu_hours=total_seconds/3600)), flush=True)


if __name__ == '__main__':
    main()
