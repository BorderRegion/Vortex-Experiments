# VortexHash MTT S4000 可行性预实验

原始任务书：[configs/TASK_SPEC.md](configs/TASK_SPEC.md)。研究 Route A：六个带符号置换环 `[11,13,17,19,23,29]` 加 16 个不变维度。所有 4-bit 结果都是软件模拟量化。

远程目录：`/root/autodl-tmp/vortex_preexp`。使用镜像已有的 `/usr/bin/python`、`torch_musa` 和 MUSA SDK，未升级驱动或替换系统环境。环境详情在 `results/raw/p0_environment.json`。

```bash
cd /root/autodl-tmp/vortex_preexp
/usr/bin/python -u -m experiments.p0
/usr/bin/python -u -m experiments.p0_supplement
/usr/bin/python -u -m experiments.check_data
/usr/bin/python -u -m experiments.p1
/usr/bin/python -u -m experiments.p4
/usr/bin/python -u -m experiments.run_p2
/usr/bin/python -u -m experiments.p1 --real results/raw/p2_relative_vortex_s2026_qk.pt --name p1_real_relative
/usr/bin/python -u -m experiments.p1 --real results/raw/p2_associative_vortex_s2026_qk.pt --name p1_real_associative
```

P2 每任务/方法 12,000 steps，batch=16，每四步长度为 `256,256,256,512`，每个模型 61,440,000 tokens。两层、512 hidden、4 heads、head_dim=128，共 6,362,112 参数。训练只用 BF16 autocast，FP32 master 参数，统一 SDPA。优化器和 seed 见 `configs/p2.json`。同任务的训练流 SHA256 应完全一致。

原始关联检索的 RoPE 控制未学会（长度 256 平均准确率 8.46%）。在原始 Vortex 结果产生前冻结一次统一延长，见 `configs/p2b.json`：所有原始结果保留，三者均从原 checkpoint/data-generator state 继续 48,000 steps，总计 307.2M tokens。因原 checkpoint 未保存 AdamW 动量，三者统一重置优化器，不宣称无缝恢复原优化器。原始和延长结果分开保存；不进行方法专属延长。

```bash
# 原始六组与 P1 补测完成后，统一追加（每种方法一次）
/usr/bin/python -u -m experiments.p2b --method rope_safe
/usr/bin/python -u -m experiments.p2b --method vortex
/usr/bin/python -u -m experiments.p2b --method ring_plain
/usr/bin/python -u -m experiments.p1 --real results/raw/p2b_associative_vortex_s2026_qk.pt --name p1_real_associative_extended
/usr/bin/python -u -m experiments.verify_backend
/usr/bin/python -u -m experiments.gradient_check
```

两层共同延长后仍未得到已学会关联检索的 RoPE 正对照，因此另冻结 `configs/p2c.json` 中的四层校准。先单独运行四层 RoPE，12,000 steps / 61.44M tokens；长度256/512、距离8/32/128的六个条件须平均≥90%、最差≥80%，才为另两个位置算子运行相同四层预算。校准失败则停止追加；不能用共同失败证明 Vortex 表达能力更弱。四层结果前缀为 `p2c_`，保留两层全部结果。原始两层计算路径未更改。

```bash
# CPU/MUSA 稀疏交叉熵及 causal SDPA 梯度验证通过后
/usr/bin/python -u -m experiments.p2 --task associative --method rope_safe --layers 4 --steps 12000 --batch 16
# 依 configs/p2c.json 的冻结门槛，决定是否运行 vortex / ring_plain
# experiments.run_p2c 是本次运行的顺序调度器，依赖其等待的上游 PID/完成日志；
# 在全新目录重现实验时应使用上述单实验命令，并按相同规则计算门槛。
```

合成任务的符号与标签独立随机生成。A 在随机位置插入已知距离的查询，训练标签仅出现在查询位置；B 检索 16 个随机 key/value 绑定之一，并放入非目标模式的重复干扰，目标 key 在上下文只出现一次。测试 seed 与训练完全分开；各距离控制使用相同内容随机源，尤其 1333/1334/1335 的绑定、目标答案与干扰数量相同。每个条件 256 样本。训练长度为 256/512，1024/2048 为长度外推。

每个 checkpoint 的高精度、INT4 per-head/group16 和 K-only 评估使用同一测试样本。Vortex 的 spectral 模型由训练权重离线换基得到，未另行训练；FP32 等价性与 BF16/量化结果分开报告。

P3 是条件阶段：先检查 P1 的真实 Q/K 和 P2。如果未达到启动条件，`results/raw/p3_gate.json` 记录原因，不生成任何虚构文本 NLL。`experiments/p3.py` 负责记录此门槛；若门槛通过，需要按任务书另行运行真实文本训练。

P1 把相同 Q 内容与九个候选距离组合；模型 Q/K 从 Vortex 第一层的独立合成测试批次采集，固定打乱后重排。输出误差中的 V 是共享随机值。这是数值机制对照，不是完整已训练 attention 的回放，不是文本验证。每种算子对自身 FP32 分数算误差，Vortex 与等谱表示的 FP32 函数相同。INT4 的尺度为 FP32，per-head 每128值1个，group16每128值8个；同组设置内尺度预算一致。FP4 仅为 E2M1 软件模拟。

完成所有训练后另执行 `/usr/bin/python -m experiments.attention_replay`，用实际第一层末端查询、完整 K、实际 V 和位置重放 causal attention（固定64序列，长度256/512/2048，距离32），补查候选重排及随机 V 的限制。此项仍是合成数据上的第一层数值分析，结果存 `attention_replay.json`，没有把它当成文本 NLL。

GPU 预算在 `results/raw/gpu_ledger.jsonl` 逐阶段记录，以同步后的阶段占用墙钟计时，包含该阶段 Python 开销；阶段之间编辑与离线报告不计入 GPU 时间。硬上限 36 h。P4 同时保存 MUSA Events 与同步 wall-clock 全部采样，位置表预计算，decode 的历史 K cache 在测量前准备。它是当前 PyTorch 参考实现的 BF16 测量，不代表原生 INT4/FP4 或 NVIDIA 性能。

训练运行避免覆盖：已有曲线文件时拒绝重复启动同名 run。需要复现时，请在新输出目录执行，或先保存旧结果再清理对应 run。源文件聚合 SHA256 随 run 保存；最终文件清单另存于 `reports/ARTIFACT_MANIFEST.json`。

环境没有新增或升级 torch、torch_musa、驱动；`requirements.txt` 记录使用的 Python 依赖。复现时将代码复制到新的独立目录，创建 `results/{raw,plots,checkpoints}` 和 `reports`，再执行上面的阶段命令；不要直接在交付结果目录上重跑。所有 GPU 阶段顺序执行，避免训练与微基准相互干扰。

交付结果阅读入口为 [研究报告](reports/PREEXP_REPORT.md)、[决策](reports/DECISION.md)、[结果 CSV](reports/RESULTS_TABLE.csv)、[完成审计](reports/COMPLETION_AUDIT.json)。原始 JSON/JSONL 保存完整配置、逐条件指标、训练数据 SHA256、代码聚合 SHA256、耗时与峰值内存。新增报告工具和兼容的 `--layers` 选项会使最终代码聚合 hash 与早期 run 不同；各阶段原始 hash 保留，最终交付文件另有 SHA256 清单。

报告由真实文件生成，不启动 GPU 计算：

```bash
/usr/bin/python -m experiments.summarize
# 完成所有条件实验后，以事实填写原因、最强证据、最致命问题
/usr/bin/python -m experiments.p3 --decision skip --reason '见最终 p3_gate.json 中的实验事实'
/usr/bin/python -m experiments.write_report --decision MODIFY --strongest '见最终 DECISION.md' --fatal '见最终 DECISION.md'
/usr/bin/python -m experiments.audit
```

这些命令中的说明文字是参数占位提示；执行 `/usr/bin/python -m experiments.finalize` 会直接读取 `results/raw/final_disposition.json` 中最终采用的事实与决策，重建门槛记录、图表、报告及审计，无需重新训练。`experiments.audit` 检查已执行阶段的证据覆盖、数据/预算公平性、checkpoint、GPU 时间及报告链接；科学解释仍应对照原始表阅读。紧凑审阅 ZIP 省略权重及 Q/K `.pt`，完整权重在远程目录和完整本地副本保留。
