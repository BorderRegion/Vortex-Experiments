"""CPU-only reporting and integrity audit for the frozen four-method experiment."""
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/hadamard_holdout'
OUT = ROOT / 'reports/hadamard_holdout'
METHODS = ['rope_safe', 'rope_hadamard', 'spectral_rope', 'vortex']
SEEDS = [2026, 2027]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def blocks(condition):
    groups = {}
    expected_offset = 0
    for sequence in condition['sequences']:
        assert sequence['offset'] == expected_offset
        index = sequence['offset'] // 8192
        assert (sequence['offset'] + sequence['tokens'] - 1) // 8192 == index
        aggregate = groups.setdefault(index, [0., 0])
        aggregate[0] += sequence['nll'] * sequence['tokens']
        aggregate[1] += sequence['tokens']
        expected_offset += sequence['tokens']
    assert expected_offset == condition['tokens']
    return np.array([groups[index] for index in sorted(groups)])


def paired_difference(first, second):
    a, b = blocks(first), blocks(second)
    assert np.array_equal(a[:, 1], b[:, 1])
    generator = np.random.default_rng(20261111)
    counts = generator.multinomial(len(a), np.full(len(a), 1 / len(a)), size=10000)
    samples = (counts @ (a[:, 0] - b[:, 0])) / (counts @ a[:, 1])
    return dict(nll_difference=first['nll'] - second['nll'],
                ppl_percent_change=100 * math.expm1(first['nll'] - second['nll']),
                paired_text_block_ci95=np.quantile(samples, [.025, .975]).tolist(),
                blocks=len(a), resamples=10000, block_tokens=8192,
                uncertainty_scope='validation/test text sampling only; not training-seed uncertainty')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / 'configs/HADAMARD_HOLDOUT_PLAN.json'
    plan = read(plan_path)
    plan_hash = digest(plan_path)
    provenance = read(ROOT / 'configs/HADAMARD_HOLDOUT_PREREGISTRATION.json')
    correct = read(RAW / 'correctness.json')
    test_manifest = read(ROOT / 'data/official_test_manifest.json')
    checks = []

    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    check('preregistered_plan_hash', provenance['plan_sha256'] == plan_hash and provenance['public_commit_verified'], provenance['github_commit'])
    check('sources_match_frozen_plan', all(digest(ROOT / name) == expected for name, expected in plan['frozen_source_sha256'].items()))
    check('hadamard_fp32_logit_equivalence_gate', correct['passed'])
    check('same_train_only_tokenizer', test_manifest['tokenizer_sha256'] == plan['tokenizer_sha256'] and digest(ROOT / 'data/tokenizer.json') == plan['tokenizer_sha256'])
    check('pinned_official_test_parquet', test_manifest['raw_sha256'] == plan['official_test']['raw_sha256'] and test_manifest['revision'] == plan['official_test']['revision'])
    validation_manifest = read(ROOT / 'data/manifest.json')
    data_hashes = dict(validation=validation_manifest['splits']['validation']['sha256'], test=test_manifest['token_stream']['sha256'])
    counts = dict(validation=validation_manifest['splits']['validation']['tokens']-1, test=test_manifest['token_stream']['next_token_targets'])
    status = {split: read(RAW / split / 'RUN_STATUS.json') for split in ['validation', 'test']}
    frozen_time = datetime.fromisoformat(provenance['verified_utc']).timestamp()
    check('public_freeze_precedes_all_new_metrics', frozen_time < status['validation']['started_unix'])
    check('validation_precedes_test_preparation', status['validation']['completed_unix'] <= test_manifest['prepared_unix'])
    check('test_prepared_before_evaluation', test_manifest['prepared_unix'] <= status['test']['started_unix'])
    check('both_splits_completed_once', all(value['status'] == 'completed' for value in status.values()))
    check('one_primary_metric_context512_qk_head_nll', plan['primary_metric'] == {'split': 'test', 'context': 512, 'condition': 'qk_int4_head', 'statistic': 'final token-weighted NLL'})
    rows, conditions, reports = [], {}, {}
    for split, seed, method in itertools.product(['validation', 'test'], SEEDS, METHODS):
        record = read(RAW / split / f'{method}_s{seed}.json')
        reports[split, seed, method] = record
        source = ('vortex' if method in ['spectral_rope', 'vortex'] else 'rope_safe')
        expected_checkpoint = f'train_{source}_s{seed}_step61036.pt'
        labels = [condition['condition'] for condition in record['conditions']]
        check(f'{split}_{seed}_{method}_protocol', record['plan_sha256'] == plan_hash and
              record['source_checkpoint'] == expected_checkpoint and
              record['source_checkpoint_sha256'] == plan['checkpoint_sha256'][expected_checkpoint] and
              record['trained_tokens'] == 500006912 and record['fake_quant_only'] and record['scales_per_head'] == 1 and
              record['stream_sha256'] == data_hashes[split] and record['tokenizer_sha256'] == plan['tokenizer_sha256'] and
              labels == ['bf16', 'qk_int4_head'])
        baseline = record['conditions'][0]['nll']
        for condition in record['conditions']:
            label = condition['condition']
            expected_quant = None if label == 'bf16' else dict(group=128, scope='qk', order='post', kind='int4')
            check(f'{split}_{seed}_{method}_{label}_coverage_and_quant', condition['tokens'] == counts[split] and
                  condition['context'] == 512 and condition['precision'] == 'bf16' and condition['quant'] == expected_quant and math.isfinite(condition['nll']))
            parts = blocks(condition)
            check(f'{split}_{seed}_{method}_{label}_sequence_nll_aggregation', abs(float(parts[:, 0].sum()/parts[:, 1].sum())-condition['nll']) < 1e-10)
            conditions[split, seed, method, label] = condition
            rows.append(dict(split=split, seed=seed, method=method, condition=label, context=512,
                nll=condition['nll'], perplexity=condition['perplexity'], quant_penalty=condition['nll']-baseline, tokens=condition['tokens']))
    for seed in SEEDS:
        for method in ['rope_safe', 'spectral_rope', 'vortex']:
            old = read(ROOT / f'results/raw/ptq_{method}_s{seed}.json')
            for label in ['bf16', 'qk_int4_head']:
                previous = next(c for c in old['conditions'] if c['context'] == 512 and c['condition'] == label)
                delta = conditions['validation', seed, method, label]['nll']-previous['nll']
                check(f'validation_anchor_{seed}_{method}_{label}', abs(delta) < 1e-7, delta)
    for split, seed in itertools.product(['validation', 'test'], SEEDS):
        check(f'{split}_{seed}_hadamard_same_rope_weights',
              reports[split, seed, 'rope_hadamard']['source_state_sha256'] == reports[split, seed, 'rope_safe']['source_state_sha256'])
        check(f'{split}_{seed}_spectral_same_vortex_source',
              reports[split, seed, 'spectral_rope']['source_state_sha256'] == reports[split, seed, 'vortex']['source_state_sha256'])
    comparisons = []
    for split, seed in itertools.product(['validation', 'test'], SEEDS):
        for first, second in [('rope_hadamard', 'rope_safe'), ('vortex', 'rope_safe'), ('vortex', 'rope_hadamard'), ('vortex', 'spectral_rope')]:
            comparison = paired_difference(conditions[split, seed, first, 'qk_int4_head'], conditions[split, seed, second, 'qk_int4_head'])
            comparison.update(split=split, seed=seed, first=first, second=second, condition='qk_int4_head')
            comparisons.append(comparison)
    means = []
    for split, method, label in itertools.product(['validation', 'test'], METHODS, ['bf16', 'qk_int4_head']):
        nll = float(np.mean([conditions[split, seed, method, label]['nll'] for seed in SEEDS]))
        means.append(dict(split=split, method=method, condition=label, mean_nll=nll, exp_mean_nll=math.exp(nll)))
    backup = read(ROOT / 'reports/FINAL_BACKUP.json')
    backup_manifest = read(ROOT / 'reports/FINAL_BACKUP_MANIFEST.json')
    required_backup = [f'results/raw/hadamard_holdout/{split}/{method}_s{seed}.json' for split, seed, method in itertools.product(['validation', 'test'], SEEDS, METHODS)]
    required_backup += ['data/official_test_manifest.json', 'data/test.bin', 'configs/HADAMARD_HOLDOUT_PLAN.json', 'configs/HADAMARD_HOLDOUT_PREREGISTRATION.json']
    check('full_local_backup_includes_new_evidence', backup['passed'] and not backup['mismatches'] and all(name in backup_manifest for name in required_backup))
    check('new_evidence_matches_verified_backup_hashes', all(name in backup_manifest and digest(ROOT / name) == backup_manifest[name]['sha256'] for name in required_backup))
    ledger = [json.loads(line) for line in (ROOT / 'results/raw/gpu_ledger.jsonl').read_text().splitlines()]
    seconds = sum(entry['seconds'] for entry in ledger)
    check('cumulative_36h_budget', seconds < 36*3600, seconds)
    check('both_new_gpu_phases_succeeded', all(any(entry['stage'] == name and entry['success'] for entry in ledger) for name in ['hadamard_correctness_and_frozen_validation', 'one_shot_official_test_four_methods_two_seeds']))
    audit = dict(passed=sum(check['passed'] for check in checks), total=len(checks), checks=checks)
    with (OUT / 'RESULTS.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'COMPARISONS.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'MEANS.json').write_text(json.dumps(means, indent=2)+'\n')
    (OUT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')

    def get(split, seed, method, label):
        return conditions[split, seed, method, label]['nll']

    def average(split, first, second):
        return float(np.mean([get(split, seed, first, 'qk_int4_head')-get(split, seed, second, 'qk_int4_head') for seed in SEEDS]))

    def comparison_line(split, first, second):
        values = [get(split, seed, first, 'qk_int4_head')-get(split, seed, second, 'qk_int4_head') for seed in SEEDS]
        delta = float(np.mean(values))
        return f'|{split}|{first} − {second}|{values[0]:+.6f}|{values[1]:+.6f}|{delta:+.6f}|{100*math.expm1(delta):+.3f}%|'

    lines = ['# 固定 Hadamard 对照与官方 test：两种子、四方法', '',
        '## 冻结方案与主指标', '',
        f'在下载官方 test、读取任何新 test 指标之前，评测代码和配置已提交至公开仓库：[{provenance["github_commit"][:7]}]({provenance["github_commit_url"]})。方案 SHA256：`{plan_hash}`。Hadamard 的验证集指标也在冻结之后才计算。', '',
        '**唯一主指标：WikiText-103 官方 test、context 512、Q/K per-head INT4 的最终 token-weighted NLL。** BF16 是对应基线；PPL 仅作为 NLL 的指数变换。所有方法、种子、条件全部报告，没有根据 test 选择配置或 checkpoint。', '',
        '复用种子 2026/2027 的 500,006,912-token 最终 checkpoint，不额外训练。RoPE+Hadamard 与标准 RoPE 使用逐位相同的权重；等谱旋转由对应 Vortex 权重离线换基。', '',
        '## 实现与浮点等价性', '',
        '固定 `H[i,j]=(-1)^popcount(i & j)/sqrt(128)`，使用自然 Sylvester 顺序。每层、每 head、每个位置的 Q/K 在位置编码后乘同一 H，再应用原有 per-head INT4；V 不变。没有随机符号、搜索、校准或拟合。实现为 FP32 butterfly，随后转回输入 dtype。', '',
        f'独立 dense 参考误差 {correct["butterfly_vs_dense_max_abs_error"]:.3g}，矩阵正交误差 {correct["orthogonality_max_abs_error"]:.3g}。两个种子共六个 validation 样本的 FP32 整模型 logits 检查通过。', '',
        '|精度|最大绝对 logit 差（六例最大）|相对 L2（六例最大）|argmax 一致率（六例最小）|',
        '|---|---:|---:|---:|']
    for precision in ['fp32', 'bf16']:
        cases = [case for case in correct['cases'] if case['precision'] == precision]
        lines.append(f'|{precision}|{max(case["max_abs_logit_difference"] for case in cases):.8g}|{max(case["relative_l2"] for case in cases):.8g}|{min(case["argmax_agreement"] for case in cases):.6f}|')
    lines += ['', 'FP32 gate 使用预先固定的最大 logit 差 <0.002、相对 L2 <1e-5。BF16 的舍入漂移单独列出，不要求逐位相等；完整 BF16 NLL 如下。', '',
        '## 官方 test：主结果', '',
        f'官方 test 原始 parquet 与固定 revision 的 SHA256 一致。沿用只在 train 拟合的 tokenizer，未重拟合；与此前相同的文章 EOS 和打包方式。共 {counts["test"]:,} 个 next-token 目标，全部参与，最后短行保留。', '',
        '|种子|方法|BF16 NLL|Q/K per-head INT4 NLL（主指标）|量化增量|INT4 PPL|',
        '|---|---|---:|---:|---:|---:|']
    for seed, method in itertools.product(SEEDS, METHODS):
        b, q = get('test', seed, method, 'bf16'), get('test', seed, method, 'qk_int4_head')
        lines.append(f'|{seed}|{method}|{b:.6f}|{q:.6f}|{q-b:+.6f}|{math.exp(q):.4f}|')
    lines += ['', '## 验证集：固定 Hadamard 对照', '',
        '|种子|方法|BF16 NLL|Q/K per-head INT4 NLL|量化增量|INT4 PPL|',
        '|---|---|---:|---:|---:|---:|']
    for seed, method in itertools.product(SEEDS, METHODS):
        b, q = get('validation', seed, method, 'bf16'), get('validation', seed, method, 'qk_int4_head')
        lines.append(f'|{seed}|{method}|{b:.6f}|{q:.6f}|{q-b:+.6f}|{math.exp(q):.4f}|')
    lines += ['', '## 直接回答两个问题', '',
        '下表差值为前者减后者，负值表示前者更好。平均 NLL 对两个种子等权；PPL 百分比取平均 NLL 差的指数，不混淆文本块与训练种子。', '',
        '|切分|比较|seed 2026 ΔNLL|seed 2027 ΔNLL|两种子平均 ΔNLL|对应 PPL 变化|',
        '|---|---|---:|---:|---:|---:|']
    for split in ['validation', 'test']:
        for first, second in [('rope_hadamard', 'rope_safe'), ('vortex', 'rope_safe'), ('vortex', 'rope_hadamard'), ('vortex', 'spectral_rope')]:
            lines.append(comparison_line(split, first, second))
    h_v = average('validation', 'rope_hadamard', 'vortex')
    h_test = average('test', 'rope_hadamard', 'vortex')
    v_test = [get('test', seed, 'vortex', 'qk_int4_head') < get('test', seed, 'rope_safe', 'qk_int4_head') for seed in SEEDS]
    lines += ['',
        f'1. 验证集上，RoPE+固定 Hadamard 相对 Vortex 的两种子平均 NLL 差为 {h_v:+.6f}。' + ('固定 Hadamard 的平均最终质量优于 Vortex，目前的 Vortex 优势可被这个现成组件追平。' if h_v < 0 else '固定 Hadamard 的平均最终质量仍落后于 Vortex；它是否改善标准 RoPE，应结合上表分别判断。'), '',
        f'2. 官方 test 上，Vortex 在 {sum(v_test)}/2 个种子的 INT4 最终 NLL 优于标准 RoPE；RoPE+Hadamard 相对 Vortex 的平均 NLL 差为 {h_test:+.6f}。' + ('此前相对裸 RoPE 的反超在这两个新文本评测中重复。' if all(v_test) else '此前相对裸 RoPE 的反超未在两个种子上全部重复，应收窄结论。'), '',
        '结论必须同时考虑标准 RoPE、现成 Hadamard 对照、等谱对照和 BF16 代价；低量化增量本身不等于最优最终质量。两个种子仍不足以精确估计训练方差。', '',
        '## 审计、数据和限制', '',
        f'本轮审计 **{audit["passed"]}/{audit["total"]}**。原三方法 validation 的十二项 NLL 与既有记录复核；完整本机备份哈希一致。累计设备占用 {seconds/3600:.3f} h，保持在 36 h 上限内。', '',
        '32 项原始 seed/method/condition 结果见 `RESULTS.csv`；配对连续 8192-token 文本块 bootstrap（10000 次、固定 seed）见 `COMPARISONS.json`，仅用于描述这两份文本的采样不确定性，不提供训练种子置信区间。完整逐序列损失和位置分桶在 `../../results/raw/hadamard_holdout/`。', '',
        '本轮没有原生 INT4 内核、速度或额外训练结论。Hadamard 的 butterfly 成本未计为部署性能收益。NLL/PPL 采用自建 BPE，不能直接对比公开的 word-level WikiText PPL。官方 test 使用一次固定配置完整评测；后续研究若再据此调整方法，不能继续将其称为未参与选择的 test。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(audit_passed=audit['passed'], audit_total=audit['total'], report=str(OUT/'REPORT.md'),
                         validation_hadamard_minus_vortex=h_v, test_hadamard_minus_vortex=h_test, test_vortex_wins=sum(v_test)), ensure_ascii=False))
    assert audit['passed'] == audit['total'], [check for check in checks if not check['passed']]


if __name__ == '__main__':
    main()
