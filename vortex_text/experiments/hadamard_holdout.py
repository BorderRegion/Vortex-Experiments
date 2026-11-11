"""Frozen post-RoPE Hadamard control and one-shot official-test evaluation.

No training, weight fitting, transform search, or calibration is performed.
Original training/model/quantization code remains unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch

from src.common import ROOT, Meter, setup, autocast, write_json
from src.model import Attention
from experiments.evaluate import load
from experiments.text import batch, evaluate, state_hash

PLAN = ROOT / 'configs/HADAMARD_HOLDOUT_PLAN.json'
RAW = ROOT / 'results/raw/hadamard_holdout'
METHODS = ['rope_safe', 'rope_hadamard', 'spectral_rope', 'vortex']
CONDITIONS = [('bf16', None), ('qk_int4_head', dict(group=128, scope='qk', order='post', kind='int4'))]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_hadamard(x):
    """Natural-order Sylvester H128/sqrt(128); FP32 butterfly then input dtype."""
    assert x.shape[-1] == 128
    shape = x.shape
    z = x.float()
    for width in [1, 2, 4, 8, 16, 32, 64]:
        groups = z.reshape(*shape[:-1], -1, 2, width)
        a, b = groups.unbind(-2)
        z = torch.stack((a + b, a - b), dim=-2).reshape(shape)
    return (z * (128 ** -0.5)).to(x.dtype)


def hadamard_position(self, x, offset=0):
    return normalized_hadamard(Attention.position(self, x, offset))


def install_hadamard(model):
    assert model.cfg.method == 'rope_safe'
    for block in model.blocks:
        block.attn.position = MethodType(hadamard_position, block.attn)


def remove_hadamard(model):
    for block in model.blocks:
        block.attn.position = MethodType(Attention.position, block.attn)


def save(name, result):
    write_json(f'results/raw/hadamard_holdout/{name}', result)


def check_plan():
    plan = json.loads(PLAN.read_text())
    assert plan['methods'] == METHODS
    assert plan['seeds'] == [2026, 2027]
    assert plan['primary_metric'] == {'split': 'test', 'context': 512, 'condition': 'qk_int4_head', 'statistic': 'final token-weighted NLL'}
    assert plan['conditions'] == [label for label, _ in CONDITIONS]
    assert time.time() < plan['wall_deadline_unix'], 'Stop before rental expiry reserve'
    for relative, expected in plan['frozen_source_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    provenance = json.loads((ROOT / 'configs/HADAMARD_HOLDOUT_PREREGISTRATION.json').read_text())
    assert provenance['plan_sha256'] == digest(PLAN)
    assert provenance['github_commit'] and provenance['public_commit_verified']
    return plan, provenance


@torch.no_grad()
def correctness(device, stream):
    plan, provenance = check_plan()
    path = RAW / 'correctness.json'
    assert not path.exists(), 'Refuse to replace prior correctness gate'
    h = np.array([[(-1.) ** ((i & j).bit_count()) for j in range(128)] for i in range(128)], dtype='<f8') / np.sqrt(128.)
    assert hashlib.sha256(h.tobytes()).hexdigest() == plan['hadamard']['matrix_float64_le_sha256']
    orthogonality = float(np.max(np.abs(h @ h.T - np.eye(128))))
    # Independent dense CPU reference checks both arithmetic and natural-order convention.
    generator = np.random.default_rng(314159)
    sample = generator.normal(size=(2, 7, 4, 128)).astype('float32')
    actual = normalized_hadamard(torch.from_numpy(sample).to(device)).cpu().numpy()
    dense = sample.astype('float64') @ h
    transform_error = float(np.max(np.abs(actual - dense)))
    cases = []
    for seed in plan['seeds']:
        base, checkpoint = load(ROOT / f'results/checkpoints/train_rope_safe_s{seed}_step61036.pt', device)
        original_state_hash = state_hash(base.state_dict())
        for precision in ['fp32', 'bf16']:
            for offset in plan['correctness']['validation_offsets']:
                x, _ = batch(stream, offset, 1, 512, device)
                remove_hadamard(base)
                with autocast(device, enabled=precision == 'bf16'):
                    a = base(x)[0].float()
                install_hadamard(base)
                with autocast(device, enabled=precision == 'bf16'):
                    b = base(x)[0].float()
                difference = (a - b).abs()
                cases.append(dict(seed=seed, precision=precision, offset=offset,
                    max_abs_logit_difference=float(difference.max()),
                    mean_abs_logit_difference=float(difference.mean()),
                    relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),
                    argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean())))
        assert state_hash(base.state_dict()) == original_state_hash
        del base, checkpoint
        torch.musa.empty_cache()
    thresholds = plan['correctness']
    fp32 = [case for case in cases if case['precision'] == 'fp32']
    passed = (orthogonality < 1e-12 and transform_error < 2e-6 and
        all(case['max_abs_logit_difference'] < thresholds['fp32_max_abs_logit_difference'] and
            case['relative_l2'] < thresholds['fp32_relative_l2'] for case in fp32))
    result = dict(plan_sha256=digest(PLAN), preregistration=provenance,
        orthogonality_max_abs_error=orthogonality, butterfly_vs_dense_max_abs_error=transform_error,
        cases=cases, thresholds=thresholds, passed=passed,
        bf16_interpretation='Finite-precision drift is measured, not required to be bitwise equal; full BF16 NLL accompanies each method.')
    save('correctness.json', result)
    assert passed, result


def run_split(split, device, stream, manifest, plan, provenance):
    assert json.loads((RAW / 'correctness.json').read_text())['passed']
    target_dir = RAW / split
    assert not target_dir.exists(), 'One-shot evaluation: refusing to overwrite/re-run a split'
    target_dir.mkdir()
    started = time.time()
    save(f'{split}/RUN_STATUS.json', dict(status='running', started_unix=started,
         plan_sha256=digest(PLAN), primary_metric=plan['primary_metric'], preregistration=provenance))
    if split == 'test':
        manifest = json.loads((ROOT / 'data/official_test_manifest.json').read_text())
        data_hash = manifest['token_stream']['sha256']
        assert len(stream) == manifest['token_stream']['tokens']
    else:
        data_hash = manifest['splits']['validation']['sha256']
    assert digest(ROOT / f'data/{split}.bin') == data_hash
    for seed in plan['seeds']:
        for method in METHODS:
            assert time.time() < plan['wall_deadline_unix']
            source_method = 'vortex' if method in ['vortex', 'spectral_rope'] else 'rope_safe'
            name = f'train_{source_method}_s{seed}_step61036.pt'
            path = ROOT / 'results/checkpoints' / name
            assert digest(path) == plan['checkpoint_sha256'][name]
            model, checkpoint = load(path, device, spectral=method == 'spectral_rope')
            if method == 'rope_hadamard':
                install_hadamard(model)
            report = dict(split=split, seed=seed, method=method, source_checkpoint=name,
                source_checkpoint_sha256=digest(path), source_state_sha256=state_hash(checkpoint['state']),
                trained_tokens=checkpoint['tokens'], plan_sha256=digest(PLAN),
                tokenizer_sha256=plan['tokenizer_sha256'], stream_sha256=data_hash,
                fake_quant_only=True, scales_per_head=1, conditions=[])
            for label, quant in CONDITIONS:
                model.set_quant(quant)
                tick = time.time()
                result = evaluate(model, stream, device, context=512, batch_size=16, per_sequence=True, precision='bf16')
                result.update(condition=label, quant=quant, elapsed_seconds=time.time()-tick)
                assert result['tokens'] == len(stream)-1
                report['conditions'].append(result)
                save(f'{split}/{method}_s{seed}.json', report)
                print(json.dumps(dict(split=split, seed=seed, method=method, condition=label,
                    nll=result['nll'], perplexity=result['perplexity'], tokens=result['tokens'])), flush=True)
            del model, checkpoint
            torch.musa.empty_cache()
    save(f'{split}/RUN_STATUS.json', dict(status='completed', started_unix=started,
        completed_unix=time.time(), plan_sha256=digest(PLAN), preregistration=provenance,
        seeds=plan['seeds'], methods=METHODS, conditions=plan['conditions'], tokens=len(stream)-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['validation', 'test'], required=True)
    args = parser.parse_args()
    plan, provenance = check_plan()
    RAW.mkdir(parents=True, exist_ok=True)
    device = setup(2026)
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    assert digest(ROOT / 'data/tokenizer.json') == plan['tokenizer_sha256']
    if args.phase == 'validation':
        assert not (ROOT / 'data/official_test_manifest.json').exists(), 'Validation gate must precede official-test preparation'
        stream = np.memmap(ROOT / 'data/validation.bin', mode='r', dtype='<u2')
        with Meter('hadamard_correctness_and_frozen_validation'):
            correctness(device, stream)
            run_split('validation', device, stream, manifest, plan, provenance)
    else:
        assert json.loads((RAW / 'validation/RUN_STATUS.json').read_text())['status'] == 'completed'
        stream = np.memmap(ROOT / 'data/test.bin', mode='r', dtype='<u2')
        with Meter('one_shot_official_test_four_methods_two_seeds'):
            run_split('test', device, stream, manifest, plan, provenance)


if __name__ == '__main__':
    main()
