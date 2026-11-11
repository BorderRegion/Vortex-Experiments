"""Whole-validation PTQ, FP32 change-of-basis check, matched context diagnostics."""
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
import torch
from src.common import ROOT, setup, autocast, Meter, write_json
from src.model import Decoder, ModelConfig
from src.spectral import convert_checkpoint
from src.ring_quant import GROUPS
from experiments.text import streams, batch, evaluate, state_hash


QUANTS = {
    'bf16': None,
    'qk_int4_head': dict(group=128, scope='qk', order='post'),
    'qk_int4_group16': dict(group=16, scope='qk', order='post'),
    'k_int4_head': dict(group=128, scope='k', order='post'),
    'qk_int4_ring7': dict(group=GROUPS, scope='qk', order='post'),
}


def load(path, device, spectral=False):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**ckpt['model_config'])
    state = ckpt['state']
    if spectral:
        cfg.method = 'spectral_rope'
        state = convert_checkpoint(state, cfg.heads)
    model = Decoder(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def equivalence(vortex, spectral, stream, device, seed=2026):
    results = []
    for offset in [0, 8192, 65536]:
        x, _ = batch(stream, offset, 1, 512, device)
        vortex.set_quant(None)
        spectral.set_quant(None)
        a = vortex(x)[0].float()
        b = spectral(x)[0].float()
        delta = (a-b).abs()
        results.append(dict(offset=offset, max_abs_logit_difference=float(delta.max()),
            mean_abs_logit_difference=float(delta.mean()),
            relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),
            argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean())))
    passed = max(r['max_abs_logit_difference'] for r in results) < .002 and max(r['relative_l2'] for r in results) < 1e-5
    report = dict(precision='FP32 weights and activations; no autocast', cases=results, passed=passed)
    suffix = '' if seed == 2026 else f'_s{seed}'
    write_json(f'results/raw/fp32_equivalence{suffix}.json', report)
    assert passed, report
    return report


@torch.no_grad()
def commutation(model, stream, device, seed=2026):
    x, _ = batch(stream, 0, 2, 512, device)
    results = []
    for label, group in [('head', 128), ('ring7', GROUPS)]:
        model.set_quant(dict(group=group, scope='qk', order='pre'))
        with autocast():
            a = model(x)[0]
        model.set_quant(dict(group=group, scope='qk', order='post'))
        with autocast():
            b = model(x)[0]
        results.append(dict(group=label, max_abs_logit_difference=float((a-b).abs().max()),
                            bitwise_equal=bool(torch.equal(a, b))))
    model.set_quant(None)
    report = dict(cases=results, passed=all(r['bitwise_equal'] for r in results))
    suffix = '' if seed == 2026 else f'_s{seed}'
    write_json(f'results/raw/text_commutation{suffix}.json', report)
    assert report['passed'], report
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6104)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    device = setup(args.seed)
    manifest, data = streams()
    paths = {m: ROOT / f'results/checkpoints/train_{m}_s{args.seed}_step{args.steps}.pt'
             for m in ['rope_safe', 'vortex']}
    with Meter('text_final_ptq_and_equivalence'):
        v, vc = load(paths['vortex'], device)
        s, _ = load(paths['vortex'], device, True)
        equivalence(v, s, data['validation'], device, args.seed)
        commutation(v, data['validation'], device, args.seed)
        # Persist the derived weights so that the control needs no retraining.
        spectral_path = ROOT / f'results/checkpoints/spectral_rope_s{args.seed}_step{args.steps}.pt'
        torch.save(dict(state={k: z.cpu() for k, z in s.state_dict().items()},
                        model_config=asdict(s.cfg), source_checkpoint=paths['vortex'].name,
                        source_state_sha256=state_hash(vc['state']),
                        transformation='Q/K per-head W -> U.T @ W; remaining weights unchanged'), spectral_path)
        del v, s, vc
        torch.musa.empty_cache()
        for method in ['rope_safe', 'vortex', 'spectral_rope']:
            path = paths['vortex' if method == 'spectral_rope' else method]
            model, ckpt = load(path, device, method == 'spectral_rope')
            report = dict(method=method, source_checkpoint=path.name, seed=args.seed,
                trained_tokens=ckpt['tokens'], validation_sha256=manifest['splits']['validation']['sha256'],
                fake_quant_only=True, scales_per_head={'head':1,'group16':8,'ring7':7}, conditions=[])
            for context in [512, 256, 128]:
                for label, quant in QUANTS.items():
                    if context != 512 and label not in ['bf16', 'qk_int4_head', 'qk_int4_group16']:
                        continue
                    model.set_quant(quant)
                    start = time.time()
                    result = evaluate(model, data['validation'], device, context=context, per_sequence=True)
                    result.update(condition=label, quant=quant, elapsed_seconds=time.time()-start)
                    report['conditions'].append(result)
                    write_json(f'results/raw/ptq_{method}_s{args.seed}.json', report)
                    print(method, context, label, result['nll'], result['perplexity'], flush=True)
            del model, ckpt
            torch.musa.empty_cache()


if __name__ == '__main__':
    main()
