import argparse
import json
import numpy as np
from src.common import ROOT


def load(name):return json.loads((ROOT/'results/raw'/name).read_text())
def fmt(x,n=5):return f'{x:.{n}g}'
def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def input_label(name):
    if '/' not in name:return name
    if 'relative' in name:return '模型 Q/K：局部选择'
    return '模型 Q/K：关联检索'+('（四层）' if 'p2c_' in name else '（延长后）' if 'p2b_' in name else '（原始）')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--decision',choices=['GO','MODIFY','STOP'],required=True)
    ap.add_argument('--strongest',required=True);ap.add_argument('--fatal',required=True)
    args=ap.parse_args()
    env=load('p0_environment.json');math=load('p0_math.json');pilot=load('p0_pilot.json');corr=load('p0_correlation.json')
    gate=load('p3_gate.json');p4=load('p4.json')
    p2={(task,method):load(f'p2_{task}_{method}_s2026.json') for task in ['relative','associative'] for method in ['rope_safe','vortex','ring_plain']}
    ledger=[json.loads(s) for s in (ROOT/'results/raw/gpu_ledger.jsonl').read_text().splitlines()]
    hours=sum(r['seconds'] for r in ledger)/3600
    p1files=sorted((ROOT/'results/raw').glob('p1*.json'))
    p1rows=[];comm=[];p1table=[]
    for file in p1files:
        data=json.loads(file.read_text());p1rows+=data['rows'];comm+=data['commutation']
        for distribution in dict.fromkeys(r['distribution'] for r in data['rows']):
            label=input_label(distribution)
            for group in [128,16]:
                r={m:next(r for r in data['rows'] if r['distribution']==distribution and r['method']==m and r['kind']=='int4' and r['group']==group and r['scope']=='qk' and r['offset']==0) for m in ['rope_safe','vortex_post','vortex_pre','spectral_rope']}
                advantage=1-r['vortex_post']['logit_nrmse']/r['spectral_rope']['logit_nrmse']
                p1table.append([label,group,*[fmt(r[m]['logit_nrmse']) for m in ['rope_safe','vortex_post','vortex_pre','spectral_rope']],pct(advantage)])
    sections=[]
    sections.append('# VortexHash：MTT S4000 可行性预实验报告\n\n## 研究问题\n\n在语言建模能力尚未验证的前提下，先检验带符号置换的量化交换律是否转化为 attention 误差和可学习任务上的实质收益。区分精确置换、频谱差异和量化坐标系差异。所有结果均来自单张 MTT S4000；4-bit 均为软件模拟。')
    sections.append('## 方法与基线\n\n固定 Route A，head_dim=128，反周期环 `[11,13,17,19,23,29]` 占 112 维，另 16 维不变。`rope_safe` 先用 CPU FP64 计算并约化角度，保存 FP32 sin/cos，正确旋转后仅量化一次。`vortex_pre/post` 分别表示量化后置换与置换后量化。`ring_plain` 仅用于合成任务的反周期消融。\n\n`spectral_rope` 使用逐环解析实正交基 U，满足 P=URUᵀ；训练后把每个 head 的 Q/K 投影改为 UᵀW，未另训模型。INT4 只使用 [-7,7]；FP4 E2M1 使用 {0,0.5,1,1.5,2,3,4,6} 与对称符号，group16 absmax/6 缩放。量化后还原浮点 attention，未实现原生低比特 kernel。')
    settings='## 实验设置\n\n'+table(['项目','实际设置'],[
        ['硬件','MTT S4000，设备报告总内存 49049 MiB；完整发现输出见原始环境 JSON'],['Python / torch / torch_musa',f"{env['python']} / {env['torch']} / {env['torch_musa']}"],
        ['驱动 / Runtime','musaInfo 5.2 / 5.2；mthreads-gmi 驱动 5.2.0-server'],['Attention','统一 MUSA SDPA，BF16 前向/反向已验证；FP32 等价性走可用的数学路径'],
        ['合成模型','2 layers，d_model=512，4 heads，FFN=2048，tie embedding，6,362,112 参数'],
        ['统一训练预算','每任务/方法 12,000 steps × batch 16；长度循环 256,256,256,512；61.44M tokens'],
        ['训练精度','BF16 autocast，FP32 master 参数'],['优化器','AdamW，LR=5e-4，betas=(0.9,0.95)，weight_decay=0.1，clip=1；200-step warmup 后 cosine 到 5e-5'],
        ['随机数','模型 seed=2026；独立训练生成器 seed=12026；测试 seed=900000+17×length'],
        ['测试','每个长度/距离条件 256 个样本；同条件所有模型与 PTQ 使用相同数据'],['GPU 占用累计',f'{hours:.4f} h（保守阶段墙钟口径，硬上限 36 h）']])
    settings+='\n\nA：在随机符号流中插入距离查询，要求返回该距离之前的符号；已训练距离为 1/2/8/32/128。只有查询位置计 loss，来源位置保证是符号而非另一个查询。B：16 个随机 key→value 绑定，加 8 个非目标绑定重复，共 24 对；目标 key 在历史中恰好出现一次，末端按 key 查询。各距离条件使用相同内容随机源，1333/1334/1335 的 key/value、答案和干扰数量配对；为避免重叠，极少数干扰位置可随目标移动而调整。\n\n测试生成器与训练分离，已独立验证标签及查询距离。各方法的数据 SHA256 一致性在完成审计中验证。1024/2048 是长度外推。单 seed 初筛；若关键信号没有接近决策边界，不追加 seed 来扩大试验。\n\nP1 初轮离群值分布没有在同一 attention 行共享 query 内容，该项作废归档于 `results/diagnostic_superseded/`；修复并增加共享 query 不变量后重跑两种分布，报告只使用重跑结果。旧轮占用时间仍保留在 GPU 账本。'
    settings+='\n\nINT4 的尺度为每 token/head 一个 FP32 absmax/7，或每连续 16 维一个 FP32 尺度（8 个/head）；同一设置内各方法的尺度数量与精度相同。per-head 与 group16 属于不同尺度预算，不直接据其精度高低推断算子优劣。'
    sections.append(settings)
    result='## 核心结果表与曲线\n\n### P0：数学与真实设备\n\n'+table(['检查','实测'],[
        ['整数参考、负位移、Gᴸ=-I、G²ᴸ=I、群关系、相对整数点积',f"{len(math['integer_checks'])} 项全部通过"],
        ['UᵀU-I 最大绝对差',fmt(math['orthogonality_max_abs'])],['P-URUᵀ 最大绝对差',fmt(math['decomposition_max_abs'])],
        ['MUSA FP32 attention 分数最大绝对差 / 相对 L2',fmt(math['device_fp32_logit_max_abs'])+' / '+fmt(math['device_fp32_logit_relative_l2'])],
        ['完整模型换基最大绝对差 / 相对 L2',fmt(math['full_checkpoint_fp32_max_abs'])+' / '+fmt(math['full_checkpoint_relative_l2'])]])
    backend=load('backend_dispatch.json');gradient=load('sparse_gradient_check.json')
    result+='\n\n'+table(['SDPA 诊断长度','实际输出设备','实际分派算子','反向有限'],[[r['length'],r['output_device'],'; '.join(op for op in r['dispatch_operators'] if op.startswith('aten::_scaled_dot')),r['backward_finite']] for r in backend['rows']])
    gradrows=[]
    for r in gradient['rows']:
        errors=[r['gradient']['relative_l2']] if r['operator']=='cross_entropy' else [v['relative_l2'] for v in r['gradients'].values()]
        gradrows.append([r['operator'],r['dtype'],r['last_query_only'],fmt(max(errors)),r['passed']])
    result+='\n\n'+table(['梯度对照','精度','仅末端查询监督','CPU/MUSA 最大梯度相对 L2','通过'],gradrows)
    result+='\n\nCPU/MUSA 接收相同 FP32 或 BF16 舍入后的输入，另查最后查询位置的交叉熵与 causal SDPA 梯度。此检查排除被测输入上的明显稀疏监督/attention 反向错误，不等于证明完整训练实现不存在任何问题。独立整数矩阵正交性、identity 尾段及 trace 复算见 `p0_math_supplement.json`。\n\n'+table(['Pilot 方法','tokens/s','step median (ms)','step P95 (ms)','峰值 MiB'],[[r['method'],fmt(r['tokens_per_second']),fmt(1000*r['median_step_s']),fmt(1000*r['p95_step_s']),fmt(r['peak_memory_bytes']/2**20)] for r in pilot['rows']])
    result+='\n\nPilot 是约 3.5 分钟的三方法吞吐测量，使用重复的小批量，仅用于预算选择，不作为质量结论。\n\n'+table(['Δ','tr(T(Δ))/128'],[[d,fmt(v)] for d,v in corr['selected'].items()])
    result+='\n\n最长单环周期为 58，联合周期 61,616,126 仍不能消除部分重合。Δ=1334 的相关性为 0.53125，而相邻 1333/1335 均为 0.125；这是数学结构诊断。\n\n![相关性与特征谱](../results/plots/p0_correlation_spectrum.png)\n\n### P1：一次量化后的 attention 误差\n\n下表为 Q+K INT4、绝对偏移 p=0，候选相对距离 1/2/8/32/128/512/1333/1334/1335。每种算子对自己的 FP32 参考计算误差；Vortex 和等谱旋转的参考函数等价。随机分布各 1024×4 个 head-query，训练模型 Q/K 各取 455×4 个；原始抽样数可查每行 `queries`。正收益表示 Vortex 的 logit NRMSE 更低。\n\n'+table(['输入','尺度维数','RoPE','Vortex 后量化','Vortex 先量化','等谱旋转','Vortex 对等谱收益'],p1table)
    softrows=[]
    for name in dict.fromkeys(r['distribution'] for r in p1rows):
        label=input_label(name)
        for group in [128,16]:
          for method in ['rope_safe','vortex_post','spectral_rope']:
            r=next(r for r in p1rows if r['distribution']==name and r['method']==method and r['kind']=='int4' and r['group']==group and r['scope']=='qk' and r['offset']==0)
            softrows.append([label,group,method,fmt(r['softmax_kl']),pct(r['top1_change']),fmt(r['output_nrmse'])])
    result+='\n\n'+table(['输入','尺度维数','方法','Softmax KL','top-1 改变率','输出 NRMSE'],softrows)
    result+='\n\n输出误差使用所有方法共享的固定随机 V，是数值诊断。模型 Q/K 来自 Vortex checkpoint 第一层的独立合成测试批次，打乱抽样后重组成九个候选距离；不是完整训练 attention 的回放，也不是文本激活或各自训练好的 RoPE 激活。此设计用于在同一内容输入上比较算子及等谱坐标；实际任务质量另看 P2。K-only、FP4、每个 Δ、全部绝对偏移的结果见机器表及原始 JSON。\n\n各向同性输入的 per-head INT4 误差基本相同；离群值分布中等谱旋转反而更好。局部选择模型 Q/K 的 Vortex logit NRMSE 虽低于标准 RoPE，但 KL/输出误差没有相应改善，不能只按归一化 logit 指标报喜。关联检索模型的 Q/K 数值差异还须结合其任务是否学会解释。\n\n![INT4 attention 误差](../results/plots/p1_int4_errors.png)\n\n### 缩放交换律与绝对偏移\n\n'
    replay=load('attention_replay.json');rt=[]
    for r in replay['rows']:
        if r['scope']=='qk' and r['method']!='vortex_pre':
            rt.append([r['source'],r['length'],r['group'],r['method'],fmt(r['logit_nrmse']),fmt(r['softmax_kl']),pct(r['top1_change']),fmt(r['output_nrmse'])])
    replay_text='### P1 补测：实际 causal attention 回放\n\n为核对随机 V 与重排候选距离的限制，另从相同 checkpoint 的第一层保留真实末端查询 Q、整段 K、实际 V 与位置，直接重放末端 causal attention。每个长度256/512/2048、距离32固定64个未见序列，BF16 投影后提升到 FP32，再比较一次 INT4。最后位置可以看见整段 token，故无需另加 causal mask。下表列全部长度的 Q+K 量化，K-only、Vortex 先量化在完整 CSV/JSON。标准 RoPE 行仍是相同内容上的位置算子数值对照，不是换掉训练模型位置编码后的质量测试。\n\n'
    replay_text+=table(['checkpoint','长度','尺度维数','方法','logit NRMSE','softmax KL','top-1 改变率','实际 V 输出 NRMSE'],rt)
    replay_text+=f"\n\nVortex/等谱 FP32 分数回放最大相对 L2 为 {fmt(max(r['relative_l2'] for r in replay['fp32_equivalence']))}。此处的输出误差已经使用模型真实 V，但仍限于第一层及合成分布；不能代替文本 NLL。\n\n"
    replay_text+='回放提供了局部数值正信号：已学会的局部任务中，per-head Vortex 对等谱的实际 V 输出误差在三个长度分别下降约46.9%、8.6%、21.7%，但 logit NRMSE 仅下降约3.4%–8.9%；group16 在长度512略有反向变化。关联模型的部分误差收益更大，但该模型未学会检索。故不能写成“完全没有数值收益”，也不能据此写成下游任务或语言建模收益。\n\n'
    result=result.replace('### 缩放交换律与绝对偏移',replay_text+'### 缩放交换律与绝对偏移')
    commrows=[]
    for kind,group in [('int4',128),('int4',16),('fp4',16)]:
        rs=[r for r in comm if r['kind']==kind and r['group']==group]
        commrows.append([kind,group,pct(min(r['code_match_rate'] for r in rs)),fmt(max(r['dequant_max_abs'] for r in rs)),fmt(max(r['dequant_mae'] for r in rs))])
    result+=table(['量化','尺度维数','最小码值一致率（所有分布/位移）','最大反量化差','最大 MAE'],commrows)
    driftrows=[]
    for group in [128,16]:
        for method in ['rope_safe','vortex_pre','vortex_post','spectral_rope']:
            r=next(r for r in p1rows if r['distribution']=='isotropic' and r['method']==method and r['kind']=='int4' and r['group']==group and r['scope']=='qk' and r['offset']==1000000)
            driftrows.append([method,group,fmt(r['absolute_shift_score_mae']),fmt(r['absolute_shift_score_max'])])
    result+='\n\n'+table(['方法','尺度维数','p=10⁶ 对 p=0 分数漂移 MAE','最大漂移'],driftrows)
    high=[r for r in p1rows if r['kind']=='fp32_shift_diagnostic']
    result+=f"\n\n所有输入/算子的 FP32 绝对偏移诊断最大差为 {fmt(max(r['max_absolute_shift'] for r in high))}；safe RoPE 没有使用 BF16 角度。上表量化后的坐标网格随绝对位置变化属于量化效应，不能归为角度精度漏洞。"
    result+='\n\nper-head absmax 在带符号置换下不变，码值和反量化值都可以精确搬移。连续 group16 的分组集合会被六个奇数长度环打乱；码值一致也不保证尺度一致，必须同时检查反量化值。因此原设计**不具备原生 block-scale 精确性**。先量化再置换的 group16 路径不再等于“在输出连续 16 维上量化”。\n\n### P2：同 token 预算的学习与 PTQ\n\n训练曲线如下；每个 run 中期及最终 checkpoint 均保留。\n\n![训练曲线](../results/plots/p2_training_curves.png)\n\n'
    p2rows=[]
    for task in ['relative','associative']:
        for method in ['rope_safe','vortex','ring_plain']:
            data=p2[task,method]
            for length in [256,512,1024,2048]:
                filt=lambda r:r['length']==length and (task=='relative' or r['distance'] in [8,32,128])
                rs=[r for r in data['bf16'] if filt(r)];acc=np.mean([r['accuracy'] for r in rs]);nll=np.mean([r['nll'] for r in rs])
                qacc=[np.mean([r['accuracy'] for r in data['ptq'] if filt(r) and r['method']==method and r['scope']=='qk' and r['group']==g]) for g in [128,16]]
                rope=np.mean([r['accuracy'] for r in p2[task,'rope_safe']['bf16'] if filt(r)])
                p2rows.append([task,method,length,pct(acc),f'{100*(acc-rope):+.2f} pp',fmt(nll),pct(qacc[0]),pct(qacc[1])])
    result+=table(['任务','方法','长度','BF16 准确率','对 RoPE 差值','查询 NLL','INT4 head','INT4 g16'],p2rows)
    result+='\n\n关联检索的长度表统一只平均共同距离 8/32/128，避免不同长度混入不同距离。局部选择平均五个训练距离。局部任务 BF16 与 QK INT4 均到达准确率天花板；原始关联检索三者均未学好，不能将共同失败归因于 Vortex。这里的 NLL 是合成任务查询交叉熵，不是文本 NLL。所有单条件、K-only 和量化 NLL 增量见 RESULTS_TABLE.csv。\n\n![长度与距离准确率](../results/plots/p2_accuracy.png)\n\n1333/1334/1335 内容配对、同数量干扰的控制（长度 2048）：\n\n'
    collision=[]
    for method in ['rope_safe','vortex','ring_plain']:
        data=p2['associative',method]
        for distance in [1333,1334,1335]:
            base=next(r for r in data['bf16'] if r['length']==2048 and r['distance']==distance)
            qa=[next(r for r in data['ptq'] if r['length']==2048 and r['distance']==distance and r['scope']=='qk' and r['group']==g) for g in [128,16]]
            collision.append([method,distance,pct(base['accuracy']),pct(qa[0]['accuracy']),pct(qa[1]['accuracy'])])
    result+=table(['方法','距离','BF16','INT4 head','INT4 g16'],collision)
    result+='\n\n每个比例由 256 样本估计，1 个样本约 0.391 个百分点；接近满分时存在明显天花板效应，不能据此声称量化更稳。单个 1334 点位的好坏也不能独自证明或否定频谱碰撞机制。\n\n训练后等谱换基检查：\n\n'+table(['任务','FP32 完整模型最大绝对差','相对 L2'],[[task,fmt(p2[task,'vortex']['spectral_fp32_equivalence']['fp32_output_max_abs']),fmt(p2[task,'vortex']['spectral_fp32_equivalence']['relative_l2'])] for task in ['relative','associative']])
    result+='\n\nFP32 等价通过后才解释 PTQ。BF16 换基会改变浮点舍入，不能把 FP32 的函数等价宣称为 BF16 bitwise 相同。映射模型的 BF16/INT4 逐条件结果已单列 `spectral`。\n\n### P3：条件性真实文本阶段\n\n'+gate['reason']+'\n\n未进行文本训练，文本 validation NLL、perplexity、量化 NLL 增量均不存在；不能把本轮合成任务当作可用语言建模能力证据。\n\n### P4：当前 MUSA 参考实现的性能\n\nB=4、H=4、D=128、BF16，统一 SDPA，15 次 warm-up、60 次重复。预计算位置表；decode 计入当前 Q 的位置操作和对完整 K/V cache 的 attention，历史 cache 构建不在单步延迟中。\n\n'
    result+=table(['方法','路径','S','Wall median ms','Wall P95 ms','GPU Event median ms','峰值 MiB'],[[r['method'],r['path'],r['sequence'],fmt(r['median_ms']),fmt(r['p95_ms']),fmt(r['gpu_event_median_ms']),fmt(r['peak_memory_bytes']/2**20)] for r in p4['rows']])
    result+='\n\n![MUSA 延迟](../results/plots/p4_latency.png)\n\nRoPE 使用保持角度精度的未融合 FP32 中间计算，Vortex 使用预计算 gather/sign；结果描述这两段可复现 PyTorch 实现。没有开发等强度融合 kernel，因此不能外推为算法固有速度差。训练 tokens/s 的顺序也可能与位置微基准不同。未使用 CUDA Events、CUDA-only 扩展、torch.compile 或 Triton-MUSA。这些是 MUSA BF16 测量，不是原生 4-bit attention 加速，也不是 RTX 5080 性能。'
    extended=[json.loads(p.read_text()) for p in sorted((ROOT/'results/raw').glob('p2b_associative_*_s2026.json'))]
    if extended:
        extension='### P2B：关联检索的统一延长\n\n原始 RoPE 在长度 256、距离 8/32/128 上平均准确率仅 8.46%，未形成已学会任务的正控制。因此在原始 Vortex 关联检索结果产生前，冻结一次共同延长：三种方法均从自己的 12,000-step checkpoint 和保存的数据生成器状态继续 48,000 steps，总计 60,000 steps / 307.2M tokens。原始结果没有覆盖。原 checkpoint 未保存 AdamW 动量，故三者统一新建同参数优化器与 200-step warmup/cosine；这是一组新的共同训练阶段，不宣称与原优化器无缝衔接。没有按方法调整预算，也不再继续追加训练。\n\n'
        erows=[]
        rope=next(d for d in extended if d['config']['method']=='rope_safe')
        for d in extended:
            method=d['config']['method']
            for length in [256,512,1024,2048]:
                filt=lambda r:r['length']==length and r['distance'] in [8,32,128]
                acc=np.mean([r['accuracy'] for r in d['bf16'] if filt(r)])
                ref=np.mean([r['accuracy'] for r in rope['bf16'] if filt(r)])
                qs=[np.mean([r['accuracy'] for r in d['ptq'] if filt(r) and r['group']==g and r['scope']=='qk']) for g in [128,16]]
                nll=np.mean([r['nll'] for r in d['bf16'] if filt(r)])
                erows.append([method,length,pct(acc),f'{100*(acc-ref):+.2f} pp',fmt(nll),pct(qs[0]),pct(qs[1])])
        extension+=table(['方法','长度','BF16','对 RoPE 差值','查询 NLL','INT4 head','INT4 g16'],erows)
        extension+='\n\n![关联检索共同延长](../results/plots/p2b_associative.png)\n\n长度 2048 的配对距离控制：\n\n'
        erows=[]
        for d in extended:
            for distance in [1333,1334,1335]:
                r=next(r for r in d['bf16'] if r['length']==2048 and r['distance']==distance)
                qs=[next(q for q in d['ptq'] if q['length']==2048 and q['distance']==distance and q['group']==g and q['scope']=='qk') for g in [128,16]]
                erows.append([d['config']['method'],distance,pct(r['accuracy']),pct(qs[0]['accuracy']),pct(qs[1]['accuracy'])])
        extension+=table(['方法','距离','BF16','INT4 head','INT4 g16'],erows)
        extension+='\n\n'+table(['方法','追加 tokens','追加训练秒数','追加 tokens/s'],[[d['config']['method'],d['additional_tokens'],fmt(d['train_seconds']),fmt(d['tokens_per_second'])] for d in extended])
        ev=next(d for d in extended if d['config']['method']=='vortex')['spectral_fp32_equivalence']
        extension+=f"\n\n延长后 Vortex 完整模型的等谱 FP32 映射：最大绝对差 {fmt(ev['fp32_output_max_abs'])}，相对 L2 {fmt(ev['relative_l2'])}。该 checkpoint 的真实 Q/K 也补入 P1。\n\n"
        extension+='三者是否学会应先看 BF16，而不是低准确率模型的 INT4 掉点。延长训练若仍不能让正对照学会，则 B 无法对位置算子的任务能力作出有效优劣判定。\n\n'
        result=result.replace('### P3：条件性真实文本阶段',extension+'### P3：条件性真实文本阶段')
    calibrated=[json.loads(p.read_text()) for p in sorted((ROOT/'results/raw').glob('p2c_associative_*_s2026.json'))]
    if calibrated:
        control=load('p2c_gate.json')
        extra='### P2C：四层 RoPE 正对照校准\n\n两层统一延长后仍未获得已学会 B 的正对照，因此在任务书允许的 2–4 层范围内，另冻结四层、12,000 steps / 61.44M tokens 的 RoPE 校准。数据生成器、seed、序列长度、优化器与学习率不变；这是独立的架构校准，不与两层模型混作同参数比较。仅当 RoPE 在长度 256/512、距离 8/32/128 的六个条件上平均准确率≥90%、每条件≥80% 时，才为 Vortex 和 ring_plain 运行完全相同的四层预算。失败则停止救援，不以失败正对照评判 Vortex。\n\n'
        extra+=table(['正对照门槛','实际平均准确率','最差条件','通过'],[['mean≥90%，min≥80%',pct(control['mean_accuracy']),pct(control['minimum_accuracy']),control['passed']]])+'\n\n'
        cr=[]
        for d in calibrated:
            for length in [256,512,1024,2048]:
                filt=lambda r:r['length']==length and r['distance'] in [8,32,128]
                rs=[r for r in d['bf16'] if filt(r)]
                qs=[np.mean([r['accuracy'] for r in d['ptq'] if filt(r) and r['group']==g and r['scope']=='qk' and r['method']==d['config']['method']]) for g in [128,16]]
                cr.append([d['config']['method'],d['parameters'],length,pct(np.mean([r['accuracy'] for r in rs])),fmt(np.mean([r['nll'] for r in rs])),pct(qs[0]),pct(qs[1])])
        extra+=table(['方法','参数数','长度','BF16','查询 NLL','INT4 head','INT4 g16'],cr)
        extra+='\n\n![四层正对照校准](../results/plots/p2c_calibration.png)\n\n'+('正对照通过，四层三个算子按同预算比较，结果与两层分列。' if control['passed'] else '四层 RoPE 在这次固定预算下仍未学会关联检索；因此没有运行额外两个四层模型。B 的能力比较保持未解决，不能据此声称 Vortex 显著劣于 RoPE。')+'\n\n'
        result=result.replace('### P3：条件性真实文本阶段',extra+'### P3：条件性真实文本阶段')
    resources='### 各训练 run 的资源与复现记录\n\n'
    resources+=table(['阶段','任务','方法','层数','累计 tokens (M)','本阶段训练秒数','本阶段 tokens/s','峰值 MiB'],[[stage,d['config']['task'],d['config']['method'],d['model_config']['layers'],fmt(d['train_tokens']/1e6),fmt(d['train_seconds']),fmt(d['tokens_per_second']),fmt(d['peak_memory_bytes']/2**20)] for stage,ds in [('P2',list(p2.values())),('P2B',extended),('P2C',calibrated)] for d in ds])
    resources+='\n\nP2B 的训练耗时与吞吐对应追加 245.76M tokens，累计 tokens 含原始 61.44M。峰值内存覆盖该 run 的训练和评测（含等谱克隆），不是仅训练前向。每个 run 的完整配置、数据 SHA256、代码聚合 SHA256 和时间戳均保留在原始 JSON；新增报告工具会改变最终聚合 hash，计算核心对照未按方法修改。汇总 CSV 对等谱 PTQ 使用其自身 BF16 为量化损失基准。GPU 账本另包括评测、换基、数值试验与微基准占用。\n\n'
    result=result.replace('### P3：条件性真实文本阶段',resources+'### P3：条件性真实文本阶段')
    sections.append(result)
    sections.append('## 是否证明了什么\n\n已验证原设计与等谱换基的 FP32 函数等价，和 per-head 对称 INT4 的精确量化交换律；已用真实设备训练、固定测试集与统一 PTQ 比较学习及量化行为。\n\n最强证据：'+args.strongest+'\n\n数学交换律与 attention 数值误差/任务掉点是不同层面的证据；判断以后者为准。')
    sections.append('## 未证明什么\n\n最致命问题：'+args.fatal+'\n\n本轮没有证明真实语言建模能力，也没有文本 NLL 证据；没有原生 INT4/FP4 推理、硬件 block-scale 精确性或 NVIDIA 加速证据。每任务仅 1 个 seed，小模型与合成分布限制外推；饱和准确率无法分辨细微差别。P1 的 head-query 共享部分内容，不应把所有 head 当作独立统计重复。')
    judgment={'GO':'行','MODIFY':'需要改','STOP':'不行'}[args.decision]
    recommendation='唯一建议的修正版：**量化分组按环对齐**，每个 11/13/17/19/23/29 环与 16 维不变块各用一个尺度，共 7 个尺度/head。这样不改变原 T(p) 或频谱，并且每个分组在置换下封闭，可恢复该布局内的精确交换律。后续必须给 RoPE/等谱旋转相同的 7 组长度和尺度预算，再检查 attention/文本收益；它仍不是固定连续 group16 的原生硬件路径。此修正版仅提出，未实现或训练；原始结果全部保留。'
    sections.append('## 下一步建议\n\n**'+args.decision+'。这个方向继续做论文，'+judgment+'。**\n\n'+recommendation+f'\n\n实际 GPU 占用 {hours:.4f} h。阶段建议时长是预算分配，不是必须烧满的配额；条件阶段未通过时不消耗剩余预算扩模。原始 JSON/JSONL、完整表、代码、模型及完成审计均随实验目录交付。')
    (ROOT/'reports/PREEXP_REPORT.md').write_text('\n\n'.join(sections)+'\n')
    decision=f'# {args.decision}\n\n**这个方向继续做论文，{judgment}。**\n\n最强证据：{args.strongest}\n\n最致命问题：{args.fatal}\n\nP3：{gate["reason"]}\n\n{recommendation}\n\n所有结论限定于当前 MUSA 软件模拟量化与合成任务；详细数据见 PREEXP_REPORT.md 和 RESULTS_TABLE.csv。GPU 累计占用 {hours:.4f} h，未超 36 h。\n'
    (ROOT/'reports/DECISION.md').write_text(decision)
    print('Reports written:',args.decision)


if __name__=='__main__':main()
