"""Train-time first-layer binding objective, with original data/model unchanged.

At an explicit value position, reconstruct its preceding key from block1 output.
The auxiliary decoder reuses the existing final norm and tied output weights.
Nothing is added to inference, and neither query labels nor answers enter input.
This is a separate, explicitly post-diagnosis protocol, not stage1 confirmation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
import torch
from torch.nn import functional as F
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import ROOT, setup, autocast, sync, Meter, append_jsonl, write_json, metadata
from .supervision import evaluate


def binding_targets(x, preserve_query=False):
    pair = ((x[:, :-1] >= 4) & (x[:, :-1] < 36) & (x[:, 1:] >= 36) & (x[:, 1:] < 68))
    labels = torch.full_like(x, -100)
    labels[:, 1:] = torch.where(pair, x[:, :-1], -100)
    if preserve_query:
        keys = (x >= 4) & (x < 36)
        labels[keys] = x[keys]
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['rope_safe','vortex','ring_plain'], default='rope_safe')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--preserve-query', action='store_true')
    args = ap.parse_args()
    device = setup(2026); cfg = ModelConfig(method=args.method)
    name = f'binding_{args.method}' + ('_aligned' if args.preserve_query else '') + '_s2026'; curve = f'results/raw/{name}_curve.jsonl'
    assert not (ROOT/curve).exists(), 'Do not overwrite a prior run.'
    model = Decoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(.9,.95), weight_decay=.1)
    gen = torch.Generator().manual_seed(12026)
    captured = []
    hook = model.blocks[0].register_forward_hook(lambda module, inputs, output: captured.append(output))
    digest = hashlib.sha256(); tokens = answer_count = binding_count = 0
    start = time.perf_counter(); losses = []
    config = dict(**vars(args), seed=2026, auxiliary_weight=.25, auxiliary_layer=0,
                  batch=16, lr=5e-4, auxiliary='Previous key reconstruction at explicit value positions')
    with Meter(name) as meter:
        for step in range(1, args.steps+1):
            length = 512 if step%4 == 0 else 256
            x, y = associative(16, length, gen)
            digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
            aux = binding_targets(x, args.preserve_query).to(device); valid = aux != -100
            tokens += x.numel(); answer_count += 16; binding_count += int(valid.sum())
            mult = min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/max(1,args.steps-200))))
            for group in opt.param_groups:
                group['lr'] = 5e-4*mult
            opt.zero_grad(set_to_none=True); captured.clear()
            with autocast():
                _, main_loss = model(x.to(device), y.to(device))
                features = captured.pop()[valid]
                logits = model.output(model.norm(features))
                aux_loss = F.cross_entropy(logits.float(), aux[valid])
                loss = main_loss+.25*aux_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            losses.append(torch.stack((main_loss.detach(),aux_loss.detach(),(logits.argmax(-1)==aux[valid]).float().mean().detach())))
            if step%500 == 0 or step == args.steps:
                sync(); means=torch.stack(losses).mean(0).cpu().tolist(); losses=[]
                row=dict(step=step,tokens=tokens,loss=means[0],auxiliary_loss=means[1],binding_accuracy=means[2],
                         elapsed_s=time.perf_counter()-start)
                append_jsonl(curve,row); print(json.dumps(row),flush=True)
                if meter.remaining()<120:
                    raise RuntimeError('Cumulative36h guard')
            if step%3000 == 0:
                # Remove the training-only hook during evaluation, avoiding retained activations.
                hook.remove(); captured.clear()
                ev=evaluate(model,device,samples=128)
                write_json(f'results/raw/{name}_eval_step{step}.json',ev)
                print(json.dumps(dict(step=step,original_generator_eval=ev)),flush=True)
                hook=model.blocks[0].register_forward_hook(lambda module,inputs,output: captured.append(output))
        hook.remove(); captured.clear()
        bf16=evaluate(model,device,samples=256,full=True); ptq=[]
        for group in (128,16):
            model.set_quant(dict(group=group,scope='qk',order='post'))
            ptq += [dict(group=group,**r) for r in evaluate(model,device,samples=256,full=True)]
        model.set_quant(None)
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},model_config=asdict(cfg),
                        config=config,step=args.steps,tokens=tokens,generator_state=gen.get_state(),optimizer=opt.state_dict()),
                   ROOT/f'results/checkpoints/{name}_final.pt')
        write_json(f'results/raw/{name}.json',dict(**metadata(config),model_config=asdict(cfg),
                   train_data_sha256=digest.hexdigest(),train_tokens=tokens,supervised_answers=answer_count,
                   auxiliary_binding_labels=binding_count,bf16=bf16,ptq=ptq,elapsed_s=time.perf_counter()-start))
    print(name+' COMPLETE',flush=True)


if __name__=='__main__':
    main()
