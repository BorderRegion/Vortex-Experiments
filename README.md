# Vortex 位置编码实验归档

研究带符号置换位置编码在 Q/K 低比特量化下的表现，与标准 RoPE 和从 Vortex 权重离线换基得到的等谱对照比较。归档日期：2026-11-11。完整结果与历史失败记录均保留。

## 最新结论：加入固定 Hadamard 与官方 test（2026-11-11）

**固定归一化 Hadamard 已在验证集和官方 test 的两个种子上都超过 Vortex。Vortex 相对裸 RoPE 的收益在 test 复现，但当前证据不足以证明其优于 RoPE 加现成正交变换的基线。**

在读取任何新增指标、下载官方 test 之前，方案与评测代码已公开冻结于 [bd5438b](https://github.com/BorderRegion/Vortex-Experiments/commit/bd5438b1142f7896602af66aae08741fc013d043)。复用两个种子的最终 checkpoint，不额外训练。唯一主指标为官方 test、context 512、Q/K per-head INT4 的最终 NLL；同时报告每个方法对应的 BF16 NLL。

官方 test 全部 291,708 个 next-token 目标，NLL 越低越好：

|方法|2026 BF16|2026 INT4（主指标）|2027 BF16|2027 INT4（主指标）|
|---|---:|---:|---:|---:|
|标准 RoPE|3.370430|3.402950|3.366874|3.391208|
|RoPE + 固定 Hadamard|3.370456|**3.378218**|3.366897|**3.374716**|
|等谱旋转|3.379525|3.391758|3.373906|3.386378|
|Vortex|3.379554|3.388059|3.373906|3.384364|

按两个种子的平均 NLL 换算，test 上 RoPE+Hadamard 相对裸 RoPE 的 PPL 改善约 **2.04%**，Vortex 约 **1.08%**；Hadamard 相对 Vortex 约改善 **0.97%**。验证集上 Hadamard 相对 Vortex 约改善 **1.28%**，已追平此前约 0.86% 的收益。

Hadamard 固定为自然序 Sylvester `H128/sqrt(128)`，放在位置编码后，对 Q/K 使用完全相同的变换，再调用原量化器。FP32 整模型最大 logit 差约 `4.20e-5`，验证等价性通过；BF16 NLL 漂移约数万分之一。没有随机符号、搜索或额外校准。新增完整审计 **114/114** 通过，本机全量备份 SHA256 核对一致。

完整内容见 [新增实验报告](vortex_text/reports/hadamard_holdout/REPORT.md)、[全部 32 项结果 CSV](vortex_text/reports/hadamard_holdout/RESULTS.csv)、[配对比较](vortex_text/reports/hadamard_holdout/COMPARISONS.json)、[预先冻结的计划](vortex_text/configs/HADAMARD_HOLDOUT_PLAN.json)。报告中的文本块区间只衡量文本采样，不替代训练种子方差；两个种子不足以精确估计后者。全部 INT4 仍为模拟量化，没有原生 4-bit 速度结论。

## 先前两种子验证结果（未加入 Hadamard）

下列原始报告保留加入 Hadamard 之前的结果和当时解释。此前 Vortex 在两个种子的 per-head INT4 上都优于裸 RoPE，平均 PPL 改善约 0.86%；BF16 和 group16 的最终质量仍落后。新增对照改变了方法相对竞争基线的判断，不改写先前实验数据。

## 报告与数据入口

- [两个种子的汇总报告](vortex_text/reports/TWO_SEED_REPORT.md)
- [完整两种子结果 CSV](vortex_text/reports/TWO_SEED_RESULTS.csv) 与 [差值 JSON](vortex_text/reports/TWO_SEED_COMPARISONS.json)
- [种子 2026 报告](vortex_text/reports/REPORT.md)、[审计](vortex_text/reports/AUDIT.json)
- [种子 2027 报告](vortex_text/reports/seed2027/REPORT.md)、[审计](vortex_text/reports/seed2027/AUDIT.json)
- [完整 PTQ 评测 JSON、训练曲线、协议与日志](vortex_text/results/raw/)
- [数据版本、切分与 SHA256](vortex_text/data/manifest.json)、[原始 tokenizer](vortex_text/data/tokenizer.json)
- [第二种子启动前冻结的计划](vortex_text/configs/SECOND_SEED_PLAN.json)
- [完整本地备份的核对结果](vortex_text/reports/FINAL_BACKUP.json)

每个种子完成 33 个评测条件，覆盖 context 128/256/512；每个条件使用全部 255,095 个 next-token 目标。原始 PTQ JSON 保留逐序列损失和位置分桶。两轮完整审计各 35/35 通过。

## 实验设置

WikiText-103 raw 标准 train/validation，训练集独立拟合 16,384 byte BPE。6 层 decoder，d=512，4 heads，FFN=2048，tied embedding，共 27,276,288 参数；context 512、batch 16。四条训练（两个方法 × 两个种子）各 500,006,912 token，种子内初始权重和数据流相同；仅使用普通 next-token CE。

AdamW lr=5e-4，warmup=200，余弦降至 5e-5，betas=(0.9,0.95)，矩阵 weight decay=0.1，clip=1。平台为 MTT S4000、MUSA SDPA、BF16 autocast / FP32 权重。等谱模型由对应 Vortex checkpoint 换基，不额外训练。PPL 是自建 BPE 下的 token PPL，不能直接与公开 word-level WikiText PPL 比较。

## 早期实验与诊断

|目录|报告|作用与限制|
|---|---|---|
|`vortex_preexp`|[原始预实验](vortex_preexp/reports/PREEXP_REPORT.md)|保留 answer-only 合成关联检索失败、原始混合机制证据和当时未启动文本阶段的事实。|
|`vortex_followup`|[补充诊断](vortex_followup/reports/REPORT.md)|CPU/MUSA 校验、监督与路由诊断、按环量化数值验证。|
|`vortex_repair`|[合成检索修复](vortex_repair/reports/REPORT.md)|更强训练期监督使检索学会；plain 对照也能成功，满分量化有上限效应。|
|`vortex_text`|[两种子真实文本](vortex_text/reports/TWO_SEED_REPORT.md)|使用普通语言建模监督检验最终质量，作为当前结论依据。|

早期阶段的结论按当时证据保留，不应将旧门槛改写为明确阳性。首次真实文本训练的日志字段冲突与重启记录保留在 `vortex_text/results/raw/failed_attempt_01/`，失败耗时仍计入账本。

## 文件范围与复核

本仓库保留报告、图表、CSV/JSON、训练日志、冻结代码、实验计划、环境与数据哈希，以及约 1.14 MB 的 tokenizer。未提交模型权重、优化器状态、原始 parquet、tokenized 二进制语料和压缩包。缺省单文件上限为 5 MiB；这些大文件继续保存在原完整本地归档中，清单见 [EXCLUDED_LARGE_FILES.json](EXCLUDED_LARGE_FILES.json)。

[ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) 记录本次提交内容的大小和 SHA256。旧阶段的 `MANIFEST.json` / 备份清单描述的是当时完整实验目录，可能包含本仓库刻意省略的大文件；不能把它们当作 GitHub 文件清单。备份通过与审计通过均是完整原始目录上的已完成检查。

下载仓库后可直接验证本次归档：

```bash
sha256sum -c SHA256SUMS
```

无需模型权重即可重新生成原始两种子汇总（需要 Python 和 NumPy，无需 GPU）：

```bash
cd vortex_text
python -m experiments.summarize_seeds
```

新增 `experiments.report_holdout` 也从既有评测 JSON 重建报告，不再次运行 test；其中完整审计会核对 `data/test.bin`，因此需要在完整本地归档中执行，或先恢复清单所列的该二进制流。GitHub 上已提供生成好的报告、CSV 和审计结果。

历史训练代码按执行时内容保存。`configs/frozen_seed2026/` 与 `configs/frozen_seed2027/` 分别记录种子对应源码。平台依赖见各阶段环境 JSON、`vortex_preexp/requirements.txt` 与数据 manifest；不要直接在已有结果目录重新训练。脚本含当次租机的绝对路径、截止日期和备份调度，重新训练需在干净副本中调整为新的环境和截止日期，重新准备语料并重新计算权重；本仓库不提供权重恢复。原始 tokenizer 可用于核对分词器版本。
