# VortexHash 补充诊断与优化

原始结果保留在相邻 `vortex_preexp/`。本目录是独立后续实验，计划见 `PLAN.json`，结论入口为 `reports/REPORT.md`。

使用同一台已授权 MTT S4000 机器，系统 `/usr/bin/python` 和原有 torch_musa；不更改驱动、SDK 或位置算子。`src/` 原有文件复制自原实验；仅 `model.py` 的量化函数 import 改为新增的 `src/ring_quant.py`，其整数128/16分组原样委托旧函数，因此原有训练和量化路径保持相同。所有原始 checkpoint 仅只读加载。源代码中的原实验路径为 `/root/autodl-tmp/vortex_preexp`。

在一个新目录创建 `results/{raw,checkpoints}`、`reports`，复制原实验 `gpu_ledger.jsonl` 以继续累计 36h 预算，再顺序执行：

```bash
/usr/bin/python -u -m experiments.run
/usr/bin/python -u -m experiments.continue_stage
/usr/bin/python -u -m experiments.final_checks
/usr/bin/python -u -m experiments.run_binding
/usr/bin/python -u -m experiments.query_swap --binding-only --output query_swap_binding
/usr/bin/python -u -m experiments.binding_probe
/usr/bin/python -u -m experiments.full_gradient
/usr/bin/python -u -m experiments.run_alignment
/usr/bin/python -u -m experiments.query_swap --binding-only --pattern '*aligned*' --output query_swap_aligned
/usr/bin/python -u -m experiments.binding_probe --pattern '*aligned*' --output binding_probe_aligned
/usr/bin/python -u -m experiments.ring_model
/usr/bin/python -u -m experiments.report
```

`run` 检查辅助标签、读取旧模型并执行配对的 RoPE 密集/稀疏监督试验。`continue_stage` 按预先记录的门槛决定是否执行唯一的短序列课程方案，以及是否运行同预算 Vortex/plain 对照。`final_checks` 完成按环分组数值回放和有效 query 替换诊断。等待调度不占 GPU；所有 GPU 工作顺序执行。

`run_binding` 是之后单独冻结的中间绑定诊断，见 `BINDING_PLAN.json`。它只增加训练辅助 loss：从第一块的 value 位置表示恢复前一个 key，复用现有 norm/output，无新增参数和推理操作。数据与原始 P2 完全一致，12,000步/61.44Mtokens。先 RoPE，只有通过相同检索门槛才训练 Vortex/plain。query_swap 的最后一条命令用于补充该阶段的有效 key 替换结果。

`run_alignment` 是根据该阶段的表示探针另冻结的单次对齐试验，见 `ALIGNMENT_PLAN.json`。除 value 位置恢复前一个 key 外，第一块的所有 key token 位置也恢复自身 key，辅助权重仍为0.25，输入、预算与推理图不变。`binding_probe` 和 `full_gradient` 分别检查表示路径和真实训练尺寸下的 CPU/MUSA 整模型梯度。

每个训练12,000步，batch16。flat 方案61.44M tokens；课程方案39.936M tokens。输入训练流、标签 SHA256、监督答案数和完整模型配置保留。辅助答案不进入模型输入，最终评价始终使用原始未见测试生成器。相同输入的 sparse/dense 两组分别提供每序列1和8个训练标签。已有结果和日志时脚本拒绝覆盖，复现应使用干净目录。

`ring_quant.quantize(x, group=GROUPS)` 按 `[11,13,17,19,23,29,16]` 分组，返回反量化数值；`return_codes=True` 还返回有符号码和7个FP32尺度。整数 `group=128/16` 委托原量化器。此实现用于数值验证，没有原生4-bit内核。标准 RoPE/等谱旋转采用相同变量长度组作尺度预算相同的对照。

整模型使用示例：

```python
from src.ring_quant import GROUPS
model.set_quant(dict(group=GROUPS, scope='qk', order='post'))
```

`order='pre'` 可验证 Vortex 按环分组的精确交换性。该接入在本轮所有训练结束后完成；量化关闭时训练计算不变。`experiments.ring_model` 验证完整 decoder 的 pre/post logits 相等。

补充实验增加了一个可选课程调度参数，flat 行为保持一致。最早的 dense run 在该可选参数加入前启动，因此其 `config` 中省略 `schedule` 字段，含义为默认 `flat`。后续脚本的源代码聚合 hash 对应保存时文件；原始逐实验 hash 保留。本轮最终 SHA256 清单在 `reports/MANIFEST.json`。
