# 关联检索修复实验

此目录保留了原始模型和数据生成器，通过训练期路由监督与 value 重建引导两层检索电路收敛。推理路径未修改。

- 原始研究：`../vortex_preexp/`
- 上一轮失败修复：`../vortex_followup/`
- 本轮协议：`PLAN.json`（仅路由对照）、`VALUE_PLAN.json`（路由 + value 重建）。
- 结果、局限和验证：`reports/REPORT.md`、`reports/RESULTS.csv`、`reports/AUDIT.json`。
- 模型/优化器/生成器状态：`results/checkpoints/`。

环境为现有 MTT S4000 48GB，torch/torch_musa 2.11，MUSA 5.2，`/usr/bin/python`。不需要更换 SDK 或驱动。主机须能 `import torch_musa`，设备为 `musa`。

在一个新结果目录中，保持 src/ 与 experiments/ 布局，运行：

```bash
/usr/bin/python -m experiments.check_routing
/usr/bin/python -m experiments.routing --method rope_safe --preserve-value
/usr/bin/python -m experiments.gate
/usr/bin/python -m experiments.routing --method vortex --preserve-value
/usr/bin/python -m experiments.routing --method ring_plain --preserve-value
/usr/bin/python -m experiments.validate
```

每个训练命令都拒绝覆盖已有曲线。继续使用本卡累计预算时，必须先复制最新 `results/raw/gpu_ledger.jsonl`；累计硬上限为36 GPU·h。查看账本时应使用最新 `vortex_repair` 账本。

仅路由对照的实际执行源码保存在 `results/raw/routing_v1_source.py`。它不包含 value 重建；已有结果不能被后续源码变化覆盖。三个最终修复模型在 `*_protocol.json` 中另存了训练脚本及生成器文件的 SHA256。

诊断命令 `experiments.intervene` 依赖前一轮 aligned/sparse checkpoint；`experiments.value_probe_cpu` 读取本轮仅路由 step3000 checkpoint。诊断用 oracle 路由或线性探针的结果均不算任务准确率。

报告可在本地（无需 torch）生成，依赖 stdlib、matplotlib：

```bash
python -m experiments.report
```

报告脚本读取相邻原始/上一轮结果，核对数据摘要、预算和参数量。`reports/MANIFEST.json` 覆盖本轮源码、日志、报告和 checkpoint；排除 manifest 自身与 Python 缓存。
