"""One common-budget continuation if the original associative positive control fails.

All original runs are retained. Same generator state, same added tokens, and the same
fresh optimizer/schedule for all three methods; no method-specific rescue.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
import torch
from src import spectral
from src.model import Decoder,ModelConfig
from src.synthetic import make_batch,VERSION
from src.common import setup,sync,autocast,write_json,append_jsonl,Meter,ROOT,metadata
from .p2 import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['rope_safe','vortex','ring_plain'],required=True)
    ap.add_argument('--steps',type=int,default=48000);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--eval-samples',type=int,default=256);args=ap.parse_args()
    # Only start a common continuation after all original comparisons exist.
    assert all((ROOT/f'results/raw/p2_{t}_{m}_s2026.json').exists() for t in ['relative','associative'] for m in ['rope_safe','vortex','ring_plain'])
    device=setup(args.seed);name=f'p2b_associative_{args.method}_s{args.seed}'
    path=f'results/raw/{name}_curve.jsonl';assert not (ROOT/path).exists()
    original=f'p2_associative_{args.method}_s{args.seed}'
    source=torch.load(ROOT/f'results/checkpoints/{original}_final.pt',map_location='cpu',weights_only=False)
    cfg=ModelConfig(**source['model_config']);model=Decoder(cfg).to(device);model.load_state_dict(source['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,betas=(.9,.95),weight_decay=.1)
    gen=torch.Generator();gen.set_state(source['generator_state'])
    digest=hashlib.sha256();tokens=source['tokens'];start=time.perf_counter();losses=[];last=start
    config=dict(**vars(args),task='associative',batch=16,lr=5e-4,base_tokens=tokens,
        base_steps=source['step'],optimizer_reset=True,reason='Original associative RoPE control did not learn; one equal continuation for all methods.')
    torch.musa.reset_peak_memory_stats()
    with Meter(name) as meter:
        for step in range(1,args.steps+1):
            length=512 if step%4==0 else 256
            x,y=make_batch('associative',16,length,gen);digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
            x,y=x.to(device),y.to(device)
            multiplier=min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/(args.steps-200))))
            for group in opt.param_groups:group['lr']=5e-4*multiplier
            opt.zero_grad(set_to_none=True)
            with autocast():_,loss=model(x,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            tokens+=x.numel();losses.append(loss.detach())
            if step%200==0 or step==args.steps:
                sync();now=time.perf_counter();row=dict(step=step,total_steps=source['step']+step,tokens=tokens,
                    loss=torch.stack(losses).mean().item(),lr=opt.param_groups[0]['lr'],elapsed_s=now-start,interval_seconds=now-last)
                append_jsonl(path,row);print(json.dumps(row),flush=True);losses=[];last=now
                if meter.remaining()<120:raise RuntimeError('Budget guard')
            if step in [args.steps//2,args.steps]:
                torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},model_config=asdict(cfg),config=config,
                    step=source['step']+step,tokens=tokens,data_sha256=digest.hexdigest(),generator_state=gen.get_state()),
                    ROOT/f'results/checkpoints/{name}_{"final" if step==args.steps else "mid"}.pt')
        sync();seconds=time.perf_counter()-start
        base=evaluate(model,'associative',device,args.eval_samples);ptq=[];mapped=[];equiv=None
        if args.method=='vortex':
            captured=[];model.blocks[0].attn.capture=captured
            with torch.no_grad(),autocast():
                x,_=make_batch('associative',16,256,torch.Generator().manual_seed(123456),distance=32,evaluation=True)
                model(x.to(device))
            model.blocks[0].attn.capture=None
            torch.save(dict(q=captured[0][0],k=captured[0][1],source=name),ROOT/f'results/raw/{name}_qk.pt')
        for group in [128,16]:
            for scope in ['qk','k']:
                model.set_quant(dict(group=group,scope=scope,order='post'))
                ptq += [dict(method=args.method,group=group,scope=scope,**r) for r in evaluate(model,'associative',device,args.eval_samples)]
        model.set_quant(None)
        if args.method=='vortex':
            clone=Decoder(ModelConfig(**{**asdict(cfg),'method':'spectral_rope'})).to(device)
            clone.load_state_dict(spectral.convert_checkpoint({k:v.cpu() for k,v in model.state_dict().items()}))
            x,_=make_batch('associative',2,256,torch.Generator().manual_seed(7654),distance=32,evaluation=True)
            model.eval();clone.eval()
            with torch.no_grad():a=model(x.to(device))[0];b=clone(x.to(device))[0]
            equiv=dict(fp32_output_max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/a.norm()).item())
            assert equiv['relative_l2']<2e-5,equiv
            mapped=[dict(method='spectral_rope',group=None,scope='bf16',**r) for r in evaluate(clone,'associative',device,args.eval_samples)]
            for group in [128,16]:
                clone.set_quant(dict(group=group,scope='qk',order='post'))
                mapped += [dict(method='spectral_rope',group=group,scope='qk',**r) for r in evaluate(clone,'associative',device,args.eval_samples)]
        write_json(f'results/raw/{name}.json',dict(**metadata(config),model_config=asdict(cfg),dataset_version=VERSION,
            train_data_sha256=digest.hexdigest(),base_train_data_sha256=source['data_sha256'],parameters=sum(p.numel() for p in model.parameters()),
            train_tokens=tokens,additional_tokens=tokens-source['tokens'],optimizer_steps=source['step']+args.steps,
            train_seconds=seconds,tokens_per_second=(tokens-source['tokens'])/seconds,peak_memory_bytes=torch.musa.max_memory_allocated(),
            bf16=base,ptq=ptq,spectral=mapped,spectral_fp32_equivalence=equiv))
    print(name+' COMPLETE',flush=True)


if __name__=='__main__':main()
