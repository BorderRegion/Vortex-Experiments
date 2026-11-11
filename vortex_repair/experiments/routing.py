"""Bootstrap a two-hop retrieval circuit, then finish with answer-only loss.

Architecture, ordinary inference and original synthetic inputs are unchanged.
Auxiliary targets are computed from training inputs, never from answer labels.
This stronger training supervision is explicitly separate from the original study.
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


def targets(x):
    pair = ((x[:,:-1] >= 4) & (x[:,:-1] < 36) &
            (x[:,1:] >= 36) & (x[:,1:] < 68))
    batch, key_position = pair.nonzero(as_tuple=True)
    matching = pair & (x[:,:-1] == x[:,-1:])
    assert torch.all(matching.sum(-1) == 1), 'Unique target binding required.'
    return (batch, key_position+1, key_position), matching.long().argmax(-1)+1


def route_loss(attn, normalized, batch, query_pos, target_pos):
    b,s,_ = normalized.shape
    q,k,_ = attn.qkv(normalized).reshape(b,s,3,4,128).unbind(2)
    q,k = attn.position(q),attn.position(k)
    # Only selected training queries need the auxiliary attention matrix.
    score = torch.einsum('nhd,nshd->nhs',q[batch,query_pos].float(),k[batch].float())/128**.5
    score = score.masked_fill(torch.arange(s,device=score.device)[None,None,:] > query_pos[:,None,None],-float('inf'))
    wanted = target_pos[:,None].expand(-1,4)
    loss = F.cross_entropy(score.reshape(-1,s),wanted.reshape(-1))
    with torch.no_grad():
        hit = (score.argmax(-1)==wanted).float().mean()
    return loss,hit


def auxiliary_weight(step):
    if step <= 3000: return 1.
    if step >= 6000: return 0.
    return .5*(1+math.cos(math.pi*(step-3000)/3000))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['rope_safe','vortex','ring_plain'],default='rope_safe')
    ap.add_argument('--steps',type=int,default=12000)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--preserve-value',action='store_true')
    args=ap.parse_args()
    device=setup(args.seed); cfg=ModelConfig(method=args.method)
    name=('routing_value_' if args.preserve_value else 'routing_')+f'{args.method}_s{args.seed}'
    curve=f'results/raw/{name}_curve.jsonl'
    assert not (ROOT/curve).exists(), 'Refuse to overwrite a previous run.'
    model=Decoder(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,betas=(.9,.95),weight_decay=.1)
    gen=torch.Generator().manual_seed(args.seed+10000)
    captured=[[],[]]; value_features=[]
    hooks=[block.attn.register_forward_pre_hook(lambda module,inputs,i=i: captured[i].append(inputs[0]))
           for i,block in enumerate(model.blocks)]
    if args.preserve_value:
        hooks.append(model.blocks[0].register_forward_hook(lambda module,inputs,output: value_features.append(output)))
    def remove_hooks():
        for h in hooks: h.remove()
        hooks.clear()
        for c in captured: c.clear()
        value_features.clear()
    digest=hashlib.sha256(); tokens=0; aux_queries=[0,0]
    start=time.perf_counter(); losses=[]
    config=dict(**vars(args),batch=16,lr=5e-4,betas=[.9,.95],weight_decay=.1,
                warmup=200,schedule='cosine to 10% of base LR',precision='BF16 autocast, FP32 master',
                auxiliary='All heads: block1 value->preceding key; block2 final query->matching value',
                auxiliary_schedule='1 through step3000; cosine decay to0 at6000; zero for remaining6000 steps',
                initialization='fresh',generator='original associative, unchanged',
                training_source_sha256=hashlib.sha256((ROOT/'experiments/routing.py').read_bytes()).hexdigest(),
                synthetic_source_sha256=hashlib.sha256((ROOT/'src/synthetic.py').read_bytes()).hexdigest())
    if args.preserve_value:
        config['value_preservation']='During auxiliary phase, decode own value token at block1 explicit value positions with existing norm+tied output; weight1 times auxiliary schedule.'
    write_json(f'results/raw/{name}_protocol.json',metadata(config))
    with Meter(name) as meter:
        torch.musa.reset_peak_memory_stats()
        for step in range(1,args.steps+1):
            length=512 if step%4==0 else 256
            x,y=associative(16,length,gen)
            digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
            tokens+=x.numel(); weight=auxiliary_weight(step)
            if weight==0 and hooks: remove_hooks()
            mult=min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/max(1,args.steps-200))))
            for group in opt.param_groups: group['lr']=5e-4*mult
            opt.zero_grad(set_to_none=True)
            for c in captured: c.clear()
            value_features.clear()
            with autocast():
                _,main_loss=model(x.to(device),y.to(device))
                if weight:
                    local, retrieval=targets(x)
                    local=[z.to(device) for z in local]
                    route1,hit1=route_loss(model.blocks[0].attn,captured[0].pop(),*local)
                    route2,hit2=route_loss(model.blocks[1].attn,captured[1].pop(),
                        torch.arange(16,device=device),torch.full((16,),length-1,device=device,dtype=torch.long),retrieval.to(device))
                    if args.preserve_value:
                        h=value_features.pop()[local[0],local[1]]
                        value_logits=model.output(model.norm(h))
                        value_targets=x.to(device)[local[0],local[1]]
                        value_loss=F.cross_entropy(value_logits.float(),value_targets)
                        value_hit=(value_logits.argmax(-1)==value_targets).float().mean()
                    else: value_loss=value_hit=main_loss.detach()*0
                    loss=main_loss+weight*(route1+route2+value_loss)
                    aux_queries[0]+=int(local[0].numel())*4; aux_queries[1]+=16*4
                else:
                    loss=main_loss
                    route1=route2=hit1=hit2=main_loss.detach()*0
                    value_loss=value_hit=main_loss.detach()*0
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            losses.append(torch.stack([z.detach() for z in (main_loss,route1,route2,hit1,hit2,value_loss,value_hit)]))
            if step%250==0 or step==args.steps:
                sync(); means=torch.stack(losses).mean(0).cpu().tolist(); losses=[]
                row=dict(step=step,tokens=tokens,answer_loss=means[0],local_route_loss=means[1],
                    retrieval_route_loss=means[2],local_route_accuracy=means[3],retrieval_route_accuracy=means[4],
                    value_preservation_loss=means[5],value_preservation_accuracy=means[6],
                    auxiliary_weight=weight,elapsed_s=time.perf_counter()-start)
                append_jsonl(curve,row); print(json.dumps(row),flush=True)
                if meter.remaining()<120: raise RuntimeError('Cumulative36h budget guard')
            if step%1000==0:
                # Evaluation uses the exact Decoder.forward without any training hooks.
                remove_hooks()
                ev=evaluate(model,device,samples=128)
                write_json(f'results/raw/{name}_eval_step{step}.json',ev)
                print(json.dumps(dict(step=step,ordinary_inference=ev)),flush=True)
                if auxiliary_weight(step+1):
                    hooks=[block.attn.register_forward_pre_hook(lambda module,inputs,i=i: captured[i].append(inputs[0]))
                           for i,block in enumerate(model.blocks)]
                    if args.preserve_value:
                        hooks.append(model.blocks[0].register_forward_hook(lambda module,inputs,output: value_features.append(output)))
            if step in (3000,6000,args.steps):
                torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},model_config=asdict(cfg),
                    config=config,step=step,tokens=tokens,generator_state=gen.get_state(),optimizer=opt.state_dict()),
                    ROOT/f'results/checkpoints/{name}_step{step}.pt')
        remove_hooks()
        bf16=evaluate(model,device,samples=256,full=True)
        write_json(f'results/raw/{name}.json',dict(**metadata(config),model_config=asdict(cfg),
            parameters=sum(p.numel() for p in model.parameters()),train_data_sha256=digest.hexdigest(),train_tokens=tokens,
            supervised_answers=args.steps*16,auxiliary_query_heads=aux_queries,bf16=bf16,
            auxiliary_value_labels=aux_queries[0]//4 if args.preserve_value else 0,
            peak_device_memory_bytes=torch.musa.max_memory_allocated(),elapsed_s=time.perf_counter()-start))
    print(name+' COMPLETE',flush=True)


if __name__=='__main__': main()
