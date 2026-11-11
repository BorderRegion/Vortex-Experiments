"""Matched plain next-token training; no synthetic-task auxiliary supervision."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from src.common import ROOT, setup, sync, autocast, Meter, write_json, append_jsonl, metadata
from src.model import Decoder, ModelConfig


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def streams():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    arrays = {s: np.memmap(ROOT / f'data/{s}.bin', mode='r', dtype='<u2')
              for s in ['train', 'validation']}
    for s, a in arrays.items():
        assert len(a) == manifest['splits'][s]['tokens']
    return manifest, arrays


def batch(stream, offset, batch_size, context, device, allow_wrap=False):
    n = batch_size * context
    if allow_wrap:
        offset %= len(stream)
    if offset+n < len(stream):
        source = np.array(stream[offset:offset+n+1], dtype=np.int64)
    else:
        assert allow_wrap and n < len(stream), (offset, n, len(stream))
        source = np.concatenate([stream[offset:], stream[:offset+n+1-len(stream)]]).astype(np.int64)
    # Exactly n adjacent next-token targets; packed rows restart causal context.
    x = torch.from_numpy(source[:-1].copy()).reshape(batch_size, context).to(device)
    y = torch.from_numpy(source[1:].copy()).reshape(batch_size, context).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, stream, device, context=512, batch_size=16, max_tokens=None,
             per_sequence=False, precision='bf16'):
    model.eval()
    usable = min(len(stream)-1, max_tokens or len(stream)-1)
    offset = 0
    total = 0.
    count = 0
    losses = []
    bucket_sum = np.zeros(4)
    bucket_n = np.zeros(4, dtype=np.int64)
    edges = [0, 32, 128, 256, 512]
    while offset < usable:
        remaining = usable - offset
        b = min(batch_size, remaining // context)
        s = context
        if b == 0:
            b, s = 1, remaining
        x, y = batch(stream, offset, b, s, device)
        with autocast(device, enabled=precision == 'bf16'):
            logits, _ = model(x)
        ce = F.cross_entropy(logits.float().reshape(-1, model.cfg.vocab_size), y.reshape(-1), reduction='none').reshape(b, s)
        values = ce.double().cpu().numpy()
        total += float(values.sum())
        count += values.size
        if per_sequence:
            losses.extend(dict(offset=offset+i*s, tokens=s, nll=float(row.mean())) for i, row in enumerate(values))
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            part = values[:, lo:min(hi, s)]
            bucket_sum[j] += part.sum()
            bucket_n[j] += part.size
        offset += b*s
    assert count == usable and math.isfinite(total)
    result = dict(nll=total/count, perplexity=math.exp(total/count), tokens=count,
                  context=context, precision=precision,
                  position_buckets=[dict(start=edges[j], end=min(edges[j+1], context),
                      tokens=int(bucket_n[j]), nll=float(bucket_sum[j]/bucket_n[j]))
                      for j in range(4) if bucket_n[j]])
    if per_sequence:
        result['sequences'] = losses
    return result


def create(method, seed, vocab, device):
    setup(seed, str(device))
    config = ModelConfig(vocab_size=vocab, layers=6, method=method)
    model = Decoder(config).to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': .1},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.},
    ], lr=5e-4, betas=(.9, .95))
    return model, optimizer


def save_checkpoint(path, model, optimizer, step, protocol, elapsed):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    value = dict(state=state, model_config=asdict(model.cfg), optimizer=optimizer.state_dict(),
                 step=step, tokens=step*protocol['batch']*protocol['context'],
                 protocol=protocol, elapsed_seconds=elapsed, torch_rng=torch.get_rng_state())
    if hasattr(torch.musa, 'get_rng_state_all'):
        value['musa_rng'] = torch.musa.get_rng_state_all()
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'train'], required=True)
    ap.add_argument('--method', choices=['rope_safe', 'vortex'], required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--steps', type=int, default=6104)
    ap.add_argument('--pilot-steps', type=int, default=60)
    args = ap.parse_args()
    manifest, data = streams()
    device = setup(args.seed)
    model, optimizer = create(args.method, args.seed, manifest['vocab_size'], device)
    n = args.batch*512
    steps = args.pilot_steps if args.mode == 'pilot' else args.steps
    prefix_digest = hashlib.sha256()
    left = n*steps+1
    while left:
        take = min(left, len(data['train']))
        prefix_digest.update(data['train'][:take].tobytes())
        left -= take
    prefix_hash = prefix_digest.hexdigest()
    protocol = dict(mode=args.mode, method=args.method, seed=args.seed, batch=args.batch,
        context=512, steps=steps, target_tokens=n*steps, lr=5e-4, min_lr=5e-5,
        warmup_steps=min(200, max(1, steps//10)), scheduler='cosine', betas=[.9,.95],
        weight_decay_matrix=.1, weight_decay_vector=0., grad_clip=1.,
        precision='BF16 autocast; FP32 weights', backend='MUSA SDPA causal',
        supervision='ordinary shifted next-token cross-entropy only',
        stream_sha256=manifest['splits']['train']['sha256'],
        consumed_prefix_sha256=prefix_hash, tokenizer_sha256=manifest['tokenizer_sha256'],
        data_order='Original token order, deterministic cyclic repetition after split-end EOS; no shuffle',
        train_stream_passes=n*steps/len(data['train']),
        initial_state_sha256=state_hash(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()), model_config=asdict(model.cfg),
        wall_deadline_unix=1794406138)  # 2026-11-11 22:08:58 Asia/Shanghai; 30 min before expiry.
    name = f'{args.mode}_{args.method}_s{args.seed}'
    write_json(f'results/raw/{name}_protocol.json', metadata(protocol))
    trace = ROOT / f'results/raw/{name}_curve.jsonl'
    assert not trace.exists(), f'Refuse to overwrite prior run: {trace}'
    times = []
    checkpoints = sorted(set([math.ceil(t/n) for t in [10_000_000,25_000_000,50_000_000,250_000_000] if math.ceil(t/n)<steps]+[steps]))
    start = time.time()
    torch.musa.reset_peak_memory_stats()
    with Meter(name) as meter:
        for step in range(1, steps+1):
            if meter.remaining() < 600 or time.time() > protocol['wall_deadline_unix']:
                save_checkpoint(ROOT / f'results/checkpoints/{name}_interrupted.pt', model, optimizer, step-1, protocol, time.time()-start)
                raise RuntimeError('Stop before budget/deadline with recoverable checkpoint')
            sync()
            t = time.perf_counter()
            model.train()
            if args.mode == 'train':
                warmup = protocol['warmup_steps']
                factor = step/warmup if step <= warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
                for group in optimizer.param_groups:
                    group['lr'] = protocol['lr']*factor
            x, y = batch(data['train'], (step-1)*n, args.batch, 512, device, allow_wrap=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                _, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(loss).item() and torch.isfinite(norm).item(), 'Nonfinite training state'
            optimizer.step()
            sync()
            dt = time.perf_counter()-t
            times.append(dt)
            record = dict(step=step, tokens=step*n, loss=float(loss.detach()), grad_norm=float(norm.detach()),
                          lr=optimizer.param_groups[0]['lr'], seconds=dt, elapsed_seconds=time.time()-start)
            if step == 1 or step % (10 if args.mode == 'pilot' else 100) == 0 or step == steps:
                append_jsonl(f'results/raw/{name}_curve.jsonl', record)
                print(json.dumps(record), flush=True)
            if args.mode == 'train' and (step % 1000 == 0 or step in checkpoints):
                val = evaluate(model, data['validation'], device, max_tokens=32768)
                write_json(f'results/raw/{name}_valid_step{step}.json', dict(step=step, training_tokens=step*n, **val))
                print('validation', step, val['nll'], flush=True)
            if args.mode == 'train' and step in checkpoints:
                save_checkpoint(ROOT / f'results/checkpoints/{name}_step{step}.pt', model, optimizer, step, protocol, time.time()-start)
        final = evaluate(model, data['validation'], device, max_tokens=32768 if args.mode == 'pilot' else None)
        result = dict(protocol=protocol, validation=final, elapsed_seconds=time.time()-start,
            peak_allocated_bytes=torch.musa.max_memory_allocated(),
            peak_reserved_bytes=torch.musa.max_memory_reserved(),
            steady_step_seconds_median=float(np.median(times[10:])),
            steady_step_seconds_p95=float(np.quantile(times[10:],.95)),
            steady_tokens_per_second=n/float(np.median(times[10:])),
            final_state_sha256=state_hash(model.state_dict()))
        write_json(f'results/raw/{name}.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
