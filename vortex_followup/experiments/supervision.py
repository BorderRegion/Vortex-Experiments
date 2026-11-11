"""Isolate auxiliary retrieval loss while keeping identical inputs and token budgets.

Original evaluation generator and the position operators are unchanged. Seven extra
queries replace filler tokens; their answers never enter the input. In the sparse
control the same queries are present but only the final answer receives loss.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
import torch
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import ROOT, setup, autocast, sync, Meter, append_jsonl, write_json, metadata


def add_queries(x, y, generator, queries=7):
    x, dense = x.clone(), y.clone()
    for b in range(x.shape[0]):
        # Derive bindings from the original, untouched sequence. All keys preceding
        # values are explicit bindings; later auxiliary queries cannot be bindings.
        binding = ((x[b, :-1] >= 4) & (x[b, :-1] < 36) &
                   (x[b, 1:] >= 36) & (x[b, 1:] < 68)).nonzero().flatten()
        free = ((x[b] >= 96) & (x[b] < 112)).nonzero().flatten()
        candidates = free[torch.randperm(len(free), generator=generator)]
        used = 0
        for p in candidates.tolist():
            eligible = binding[binding+1 < p]
            if len(eligible) == 0:
                continue
            src = eligible[int(torch.randint(len(eligible), (1,), generator=generator))]
            x[b, p] = x[b, src]
            dense[b, p] = x[b, src+1]
            used += 1
            if used == queries:
                break
        assert used == queries
    return x, dense


@torch.no_grad()
def evaluate(model, device, samples=256, full=False):
    model.eval(); rows = []
    lengths = (256, 512, 1024, 2048) if full else (256, 512)
    for length in lengths:
        distances = (8, 32, 128, 512, 1333, 1334, 1335) if full else (8, 32, 128)
        for distance in distances:
            if distance >= length-1:
                continue
            gen = torch.Generator().manual_seed(900000+length*17)
            correct = nll = 0.
            for start in range(0, samples, 16):
                n = min(16, samples-start)
                x, y = associative(n, length, gen, distance=distance, evaluation=True)
                with autocast():
                    logits, loss = model(x.to(device), y.to(device))
                correct += (logits[:, -1].argmax(-1) == y[:, -1].to(device)).sum().item()
                nll += loss.item()*n
            rows.append(dict(length=length, distance=distance, samples=samples,
                             accuracy=correct/samples, nll=nll/samples))
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['rope_safe', 'vortex', 'ring_plain'], default='rope_safe')
    ap.add_argument('--loss', choices=['sparse', 'dense'], required=True)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--schedule', choices=['flat', 'curriculum'], default='flat')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--eval-samples', type=int, default=256)
    args = ap.parse_args()
    device = setup(args.seed)
    name = f'supervision_{args.method}_{args.loss}' + ('_curriculum' if args.schedule == 'curriculum' else '') + f'_s{args.seed}'
    curve = f'results/raw/{name}_curve.jsonl'
    assert not (ROOT/curve).exists(), 'Refuse to overwrite a previous run.'
    cfg = ModelConfig(method=args.method)
    model = Decoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9,.95), weight_decay=.1)
    base_gen = torch.Generator().manual_seed(args.seed+10000)
    aux_gen = torch.Generator().manual_seed(args.seed+20000)
    xhash, yhash = hashlib.sha256(), hashlib.sha256()
    tokens = labels = 0
    start = time.perf_counter(); losses = []
    with Meter(name) as meter:
        for step in range(1, args.steps+1):
            length = (64 if step <= 3000 else 128 if step <= 6000 else
                      512 if step%4 == 0 else 256) if args.schedule == 'curriculum' else (512 if step%4 == 0 else 256)
            x, y = associative(16, length, base_gen)
            x, dense = add_queries(x, y, aux_gen)
            if args.loss == 'dense':
                y = dense
            xhash.update(x.numpy().tobytes()); yhash.update(y.numpy().tobytes())
            tokens += x.numel(); labels += int((y != -100).sum())
            mult = min(1., step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0, step-200)/max(1,args.steps-200))))
            for group in opt.param_groups:
                group['lr'] = 5e-4*mult
            opt.zero_grad(set_to_none=True)
            with autocast():
                _, loss = model(x.to(device), y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            losses.append(loss.detach())
            if step%500 == 0 or step == args.steps:
                sync()
                row = dict(step=step, tokens=tokens, supervised_answers=labels,
                           length=length,
                           loss=torch.stack(losses).mean().item(), elapsed_s=time.perf_counter()-start,
                           lr=opt.param_groups[0]['lr'])
                append_jsonl(curve, row); print(json.dumps(row), flush=True); losses=[]
                if meter.remaining() < 120:
                    raise RuntimeError('36h cumulative budget guard')
            if step%3000 == 0:
                ev = evaluate(model, device, samples=128)
                write_json(f'results/raw/{name}_eval_step{step}.json', ev)
                print(json.dumps(dict(step=step, original_generator_eval=ev)), flush=True)
        bf16 = evaluate(model, device, samples=args.eval_samples, full=True)
        ptq = []
        for group in (128, 16):
            model.set_quant(dict(group=group, scope='qk', order='post'))
            ptq += [dict(group=group, **r) for r in evaluate(model, device, samples=args.eval_samples, full=True)]
        model.set_quant(None)
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()}, model_config=asdict(cfg),
                        config=vars(args), step=args.steps, tokens=tokens, optimizer=opt.state_dict(),
                        base_generator_state=base_gen.get_state(), auxiliary_generator_state=aux_gen.get_state()),
                   ROOT/f'results/checkpoints/{name}_final.pt')
        write_json(f'results/raw/{name}.json', dict(**metadata(vars(args)), model_config=asdict(cfg),
                   train_input_sha256=xhash.hexdigest(), train_label_sha256=yhash.hexdigest(),
                   train_tokens=tokens, supervised_answers=labels, bf16=bf16, ptq=ptq,
                   elapsed_s=time.perf_counter()-start))
    print(name+' COMPLETE', flush=True)


if __name__ == '__main__':
    main()
