# Vortex：真实文本阶段

WikiText-103 raw，训练集拟合 16k byte BPE，27.28M decoder；RoPE 与 Vortex 各 500,006,912 token，同初始化、同数据流、同优化预算。仅使用普通 next-token CE。

- `configs/PLAN.json`：主训练开始前冻结的预算与方案。
- `data/manifest.json`：数据版本、完整哈希、分词器及切分说明。
- `results/raw/train_runner.json`：远端顺序任务状态。
- `results/raw/train_*_curve.jsonl`：训练进度与损失。
- `results/raw/train_*_valid_step*.json`：固定验证子集的训练曲线。
- `reports/MIRROR_STATUS.json`：自动同步状态。
- `reports/REPORT.md`、`RESULTS.csv`、`AUDIT.json`：两条训练和完整 PTQ 完成后由本地 CPU 自动生成。

远端目录 `/root/autodl-tmp/vortex_text`；现有任务通过 `experiments.run_pair` 顺序运行：先 RoPE、再 Vortex、最后离线等谱换基与全部验证评测。`experiments.mirror` 每分钟同步结果、日志及 checkpoint，并在成功结束后生成报告。数据及 tokenizer 已另外完整同步并核对哈希。

```bash
# 仅在独立、尚未执行的目录中复现实验，脚本拒绝覆盖同名旧训练日志。
/usr/bin/python -m experiments.prepare_data
/usr/bin/python -m experiments.run_pair --mode pilot
/usr/bin/python -m experiments.run_pair --mode train --steps 61036
python -m experiments.report
```

平台：MTT S4000，MUSA，PyTorch 2.11。没有修改系统/驱动/SDK。INT4 仅 fake quant；spectral_rope 从同一 Vortex checkpoint 换基获得，不额外训练。per-head/group16/ring7 的尺度数分别为 1/8/7，同一格式在各方法之间一致；ring7 不代表原生 group16。

远端到期：2026-11-11 22:38:58（北京时间）；脚本在 22:08:58 前停止并保留 checkpoint，同时继续遵守此前阶段累计 36 GPU·h 的硬上限。首次写入验证结果时发生过字段名冲突，已修复；完整失败日志保留于 `results/raw/failed_attempt_01`，消耗仍计入账本。

## 第二随机种子

第一种子（2026）已完成两条各 500M token 训练、33 项完整验证条件及本地完整备份。为检验 per-head 小幅最终质量收益是否能重复，追加一次 seed 2027，配置及 token 预算完全相同，未调超参数。

`results/raw/train_s2027_runner.json` 为第二对训练状态，`reports/MIRROR_STATUS_s2027.json` 为其本地同步状态。第二种子全部文件名带 s2027，与 s2026 分开保存；其报告写入 `reports/seed2027/`。全部完成后自动生成 `reports/TWO_SEED_REPORT.md`，并重新备份和核对所有文件哈希。到期前 22:10 的本地备份任务覆盖整个目录及两个种子的结果。
