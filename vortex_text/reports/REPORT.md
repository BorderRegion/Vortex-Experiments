# Vortex 真实文本预实验

## 研究问题

带符号置换位置算子能否在保留语言建模能力时，降低 Q/K INT4 量化带来的实际损失？本轮是合成检索修复后的探索性 P3，不将先前混合的 P1 证据重新表述为明确阳性。

## 方法与基线

RoPE 使用 FP64 构造角度的安全实现；Vortex 保留原始六环及 16 个不变维度。spectral_rope 由同一 Vortex checkpoint 离线换基，未另行训练。三者分别使用自己的 BF16 NLL 计算量化增量；FP32 等价不意味着 BF16 舍入后输出逐位相同。

## 实验设置

WikiText-103 raw 标准 train/validation 切分，训练集独立拟合 16,384 byte BPE。6 层、d=512、4 heads、FFN=2048、tied embedding，共 27,276,288 参数；context 512，batch 16，seed 2026。两条训练初始化及数据顺序相同，各训练 500,006,912 token，约 4.098 次训练流。仅使用普通 next-token CE，未使用合成检索的辅助监督。预算根据 pilot 在主训练开始前冻结。

AdamW lr 5e-4、warmup 200 steps、余弦降至 5e-5、betas=(.9,.95)、矩阵 weight decay=.1、clip=1。MTT S4000、MUSA SDPA、BF16 autocast / FP32 权重。验证集全部 255,095 个 next-token 目标参与评测，不与训练流拼接。PPL 是该 BPE 下的 token PPL，不能直接与公开的 word-level WikiText PPL 比较。

## 核心结果表与曲线

|方法|条件（context 512）|NLL|PPL|量化增量|
|---|---|---:|---:|---:|
|rope_safe|bf16|3.369918|29.076|+0.000000|
|rope_safe|qk_int4_head|3.401877|30.020|+0.031959|
|rope_safe|qk_int4_group16|3.381071|29.402|+0.011153|
|rope_safe|k_int4_head|3.389688|29.657|+0.019770|
|rope_safe|qk_int4_ring7|3.381584|29.417|+0.011667|
|vortex|bf16|3.384464|29.502|+0.000000|
|vortex|qk_int4_head|3.394042|29.786|+0.009578|
|vortex|qk_int4_group16|3.391016|29.696|+0.006552|
|vortex|k_int4_head|3.389845|29.661|+0.005381|
|vortex|qk_int4_ring7|3.391080|29.698|+0.006616|
|spectral_rope|bf16|3.384503|29.503|+0.000000|
|spectral_rope|qk_int4_head|3.396692|29.865|+0.012189|
|spectral_rope|qk_int4_group16|3.391948|29.724|+0.007445|
|spectral_rope|k_int4_head|3.391809|29.720|+0.007306|
|spectral_rope|qk_int4_ring7|3.392056|29.727|+0.007553|

完整的 context 128/256/512、位置分桶及逐序列损失见原始 JSON，全部聚合结果见 RESULTS.csv。训练和验证曲线见 training_curves.png，量化增量见 quant_penalty.png。

## 是否证明了什么

FP32 整模型等谱校验：True；Vortex per-head/ring7 先量化与后量化整模型逐位一致：True。绝对质量与量化增量均单独列出，避免只看较小增量而忽略 BF16 质量差距。

COMPARISONS.json 给出相同目标 token 上的配对 8,192-token 连续块 bootstrap（10,000 次）。它反映这份验证文本的采样不确定性，不估计训练种子方差；单种子不能证明训练稳定性。

## 未证明什么

所有低比特评测都是对称 [-7,7] INT4 fake quant，attention 仍在 BF16/FP32 执行。没有原生 4-bit 速度结论。head/group16/ring7 分别使用 1/8/7 个 FP32 尺度，每种格式在各方法之间使用相同分组预算；ring7 不能称为原生 group16。group16 不具有置换交换律，既有失败证据仍然成立。

此前合成任务依赖更强训练监督才学会，不能改写为 answer-only 成功；其满分 PTQ 仍有上限效应，不能据此推断文本收益。

## 下一步建议

seed 2026：BF16 的 Vortex−RoPE NLL=+0.014546；Q/K per-head INT4 的 Vortex−RoPE NLL=-0.007835，配对文本块 95% 区间 [-0.011333, -0.004486]。量化增量相对标准 RoPE 减少 70.0%，相对等谱换基对照减少 21.4%。group16 的量化后绝对 NLL 差为 +0.009945。

MODIFY / 继续验证 per-head 场景：真实文本出现了有判别力的低比特收益，但 BF16 质量仍有代价，group16 未取得全面胜利。这个方向继续做论文，需要改进并限定适用场景。

等谱旋转自身也可能比标准 RoPE 稳健，因此不能把相对 RoPE 的全部改进归因于离散置换。两个训练种子的预先计划对照用于检查重复性，文本块 bootstrap 不替代种子方差。

本阶段结束后累计设备占用 6.807 h（含全部此前实验）；本地审计 35/35 通过。
