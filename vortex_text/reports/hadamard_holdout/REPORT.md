# 固定 Hadamard 对照与官方 test：两种子、四方法

## 冻结方案与主指标

在下载官方 test、读取任何新 test 指标之前，评测代码和配置已提交至公开仓库：[bd5438b](https://github.com/BorderRegion/Vortex-Experiments/commit/bd5438b1142f7896602af66aae08741fc013d043)。方案 SHA256：`38bc5ea1ef440e07a7bcd6e946ccdde467f81acbd4ef85a6de99da5e005169bd`。Hadamard 的验证集指标也在冻结之后才计算。

**唯一主指标：WikiText-103 官方 test、context 512、Q/K per-head INT4 的最终 token-weighted NLL。** BF16 是对应基线；PPL 仅作为 NLL 的指数变换。所有方法、种子、条件全部报告，没有根据 test 选择配置或 checkpoint。

复用种子 2026/2027 的 500,006,912-token 最终 checkpoint，不额外训练。RoPE+Hadamard 与标准 RoPE 使用逐位相同的权重；等谱旋转由对应 Vortex 权重离线换基。

## 实现与浮点等价性

固定 `H[i,j]=(-1)^popcount(i & j)/sqrt(128)`，使用自然 Sylvester 顺序。每层、每 head、每个位置的 Q/K 在位置编码后乘同一 H，再应用原有 per-head INT4；V 不变。没有随机符号、搜索、校准或拟合。实现为 FP32 butterfly，随后转回输入 dtype。

独立 dense 参考误差 4.59e-07，矩阵正交误差 1.11e-16。两个种子共六个 validation 样本的 FP32 整模型 logits 检查通过。

|精度|最大绝对 logit 差（六例最大）|相对 L2（六例最大）|argmax 一致率（六例最小）|
|---|---:|---:|---:|
|fp32|4.196167e-05|9.3903975e-07|1.000000|
|bf16|0.375|0.006226514|0.990234|

FP32 gate 使用预先固定的最大 logit 差 <0.002、相对 L2 <1e-5。BF16 的舍入漂移单独列出，不要求逐位相等；完整 BF16 NLL 如下。

## 官方 test：主结果

官方 test 原始 parquet 与固定 revision 的 SHA256 一致。沿用只在 train 拟合的 tokenizer，未重拟合；与此前相同的文章 EOS 和打包方式。共 291,708 个 next-token 目标，全部参与，最后短行保留。

|种子|方法|BF16 NLL|Q/K per-head INT4 NLL（主指标）|量化增量|INT4 PPL|
|---|---|---:|---:|---:|---:|
|2026|rope_safe|3.370430|3.402950|+0.032520|30.0526|
|2026|rope_hadamard|3.370456|3.378218|+0.007762|29.3185|
|2026|spectral_rope|3.379525|3.391758|+0.012234|29.7182|
|2026|vortex|3.379554|3.388059|+0.008504|29.6084|
|2027|rope_safe|3.366874|3.391208|+0.024334|29.7018|
|2027|rope_hadamard|3.366897|3.374716|+0.007819|29.2160|
|2027|spectral_rope|3.373906|3.386378|+0.012472|29.5587|
|2027|vortex|3.373906|3.384364|+0.010458|29.4992|

## 验证集：固定 Hadamard 对照

|种子|方法|BF16 NLL|Q/K per-head INT4 NLL|量化增量|INT4 PPL|
|---|---|---:|---:|---:|---:|
|2026|rope_safe|3.369918|3.401877|+0.031959|30.0204|
|2026|rope_hadamard|3.369875|3.377853|+0.007978|29.3078|
|2026|spectral_rope|3.384503|3.396692|+0.012189|29.8652|
|2026|vortex|3.384464|3.394042|+0.009578|29.7861|
|2027|rope_safe|3.363926|3.390547|+0.026622|29.6822|
|2027|rope_hadamard|3.363931|3.371687|+0.007756|29.1276|
|2027|spectral_rope|3.371306|3.383828|+0.012522|29.4834|
|2027|vortex|3.371297|3.381170|+0.009873|29.4052|

## 直接回答两个问题

下表差值为前者减后者，负值表示前者更好。平均 NLL 对两个种子等权；PPL 百分比取平均 NLL 差的指数，不混淆文本块与训练种子。

|切分|比较|seed 2026 ΔNLL|seed 2027 ΔNLL|两种子平均 ΔNLL|对应 PPL 变化|
|---|---|---:|---:|---:|---:|
|validation|rope_hadamard − rope_safe|-0.024024|-0.018860|-0.021442|-2.121%|
|validation|vortex − rope_safe|-0.007835|-0.009377|-0.008606|-0.857%|
|validation|vortex − rope_hadamard|+0.016189|+0.009483|+0.012836|+1.292%|
|validation|vortex − spectral_rope|-0.002650|-0.002658|-0.002654|-0.265%|
|test|rope_hadamard − rope_safe|-0.024732|-0.016492|-0.020612|-2.040%|
|test|vortex − rope_safe|-0.014891|-0.006844|-0.010868|-1.081%|
|test|vortex − rope_hadamard|+0.009841|+0.009648|+0.009744|+0.979%|
|test|vortex − spectral_rope|-0.003700|-0.002014|-0.002857|-0.285%|

1. 验证集上，RoPE+固定 Hadamard 相对 Vortex 的两种子平均 NLL 差为 -0.012836。固定 Hadamard 的平均最终质量优于 Vortex，目前的 Vortex 优势可被这个现成组件追平。

2. 官方 test 上，Vortex 在 2/2 个种子的 INT4 最终 NLL 优于标准 RoPE；RoPE+Hadamard 相对 Vortex 的平均 NLL 差为 -0.009744。此前相对裸 RoPE 的反超在这两个新文本评测中重复。

结论必须同时考虑标准 RoPE、现成 Hadamard 对照、等谱对照和 BF16 代价；低量化增量本身不等于最优最终质量。两个种子仍不足以精确估计训练方差。

## 审计、数据和限制

本轮审计 **114/114**。原三方法 validation 的十二项 NLL 与既有记录复核；完整本机备份哈希一致。累计设备占用 10.267 h，保持在 36 h 上限内。

32 项原始 seed/method/condition 结果见 `RESULTS.csv`；配对连续 8192-token 文本块 bootstrap（10000 次、固定 seed）见 `COMPARISONS.json`，仅用于描述这两份文本的采样不确定性，不提供训练种子置信区间。完整逐序列损失和位置分桶在 `../../results/raw/hadamard_holdout/`。

本轮没有原生 INT4 内核、速度或额外训练结论。Hadamard 的 butterfly 成本未计为部署性能收益。NLL/PPL 采用自建 BPE，不能直接对比公开的 word-level WikiText PPL。官方 test 使用一次固定配置完整评测；后续研究若再据此调整方法，不能继续将其称为未参与选择的 test。
