import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
import torch
from src import spectral
from src.model import Decoder,ModelConfig
from src.synthetic import make_batch,OFFSETS,VERSION
from src.common import setup,sync,autocast,write_json,append_jsonl,Meter,ROOT,metadata


@torch.no_grad()
def evaluate(model,task,device,samples=256,quick=False):
    model.eval(); rows=[]
    lengths=[256] if quick else [256,512,1024,2048]
    for length in lengths:
        distances=([1,2,8,32,128] if task=='relative' else [8,32,128,512,1333,1334,1335])
        for distance in distances:
            if distance>=length-1:continue
            # Each condition has an independent fixed generator; repeated across methods/PTQ.
            gen=torch.Generator().manual_seed(900000+length*17)
            count=correct=0; nll=0.
            while count<samples:
                batch=min(8,samples-count)
                x,y=make_batch(task,batch,length,gen,distance=distance,evaluation=True)
                x,y=x.to(device),y.to(device)
                with autocast(): logits,loss=model(x,y)
                valid=y!=-100
                correct+=(logits.argmax(-1)[valid]==y[valid]).sum().item()
                n=valid.sum().item(); nll+=loss.item()*n; count+=n
            rows.append(dict(task=task,length=length,distance=distance,samples=count,accuracy=correct/count,nll=nll/count))
    model.train()
    return rows


def train(args):
    device=setup(args.seed); name=f'{"p2" if args.layers==2 else "p2c"}_{args.task}_{args.method}_s{args.seed}'
    config=vars(args).copy();cfg=ModelConfig(method=args.method,backend=args.backend,layers=args.layers)
    model=Decoder(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.1)
    gen=torch.Generator().manual_seed(args.seed+10000)
    curve_path=f'results/raw/{name}_curve.jsonl'
    assert not (ROOT/curve_path).exists(),'Avoid overwriting a completed run; use resume explicitly.'
    total_tokens=0; digest=hashlib.sha256(); start=time.perf_counter(); last=start; losses=[]
    torch.musa.reset_peak_memory_stats()
    with Meter(name) as meter:
        for step in range(1,args.steps+1):
            length=512 if step%4==0 else 256
            x,y=make_batch(args.task,args.batch,length,gen)
            digest.update(x.numpy().tobytes()); digest.update(y.numpy().tobytes())
            x,y=x.to(device),y.to(device)
            # Fixed warmup+cosine schedule, independent of observed progress.
            warmup=min(200,max(20,args.steps//20))
            multiplier=min(1.,step/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.steps-warmup))))
            for group in opt.param_groups:group['lr']=args.lr*multiplier
            opt.zero_grad(set_to_none=True)
            with autocast(): _,loss=model(x,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            total_tokens+=x.numel(); losses.append(loss.detach())
            if step%args.log_every==0 or step==args.steps:
                sync(); now=time.perf_counter(); mean_loss=torch.stack(losses).mean().item(); losses=[]
                row=dict(step=step,tokens=total_tokens,loss=mean_loss,lr=opt.param_groups[0]['lr'],
                         elapsed_s=now-start,interval_seconds=now-last,peak_memory_bytes=torch.musa.max_memory_allocated())
                append_jsonl(curve_path,row);print(json.dumps(row),flush=True);last=now
                if meter.remaining()<120:raise RuntimeError('Budget guard: stop at completed optimizer step')
            if step in [args.steps//2,args.steps]:
                torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},model_config=asdict(cfg),
                    config=config,step=step,tokens=total_tokens,data_sha256=digest.hexdigest(),generator_state=gen.get_state()),
                    ROOT/f'results/checkpoints/{name}_{"final" if step==args.steps else "mid"}.pt')
        sync(); train_seconds=time.perf_counter()-start
        evalrows=evaluate(model,args.task,device,args.eval_samples)
        rawqk=[]
        if args.method=='vortex':
            model.blocks[0].attn.capture=rawqk
            with torch.no_grad(),autocast():
                x,_=make_batch(args.task,16,256,torch.Generator().manual_seed(123456),distance=32,evaluation=True)
                model(x.to(device))
            model.blocks[0].attn.capture=None
            torch.save(dict(q=rawqk[0][0],k=rawqk[0][1],source=name),ROOT/f'results/raw/{name}_qk.pt')
        ptq=[]
        for group in [128,16]:
            for scope in ['qk','k']:
                model.set_quant(dict(group=group,scope=scope,order='post'))
                ptq+= [dict(method=args.method,group=group,scope=scope,**r) for r in evaluate(model,args.task,device,args.eval_samples)]
        mapped=[]; equivalence=None
        if args.method=='vortex':
            model.set_quant(None); scfg=ModelConfig(**{**asdict(cfg),'method':'spectral_rope'})
            clone=Decoder(scfg).to(device)
            clone.load_state_dict(spectral.convert_checkpoint({k:v.cpu() for k,v in model.state_dict().items()}))
            x,_=make_batch(args.task,2,256,torch.Generator().manual_seed(7654),distance=32,evaluation=True)
            model.eval();clone.eval()
            with torch.no_grad():
                a=model(x.to(device))[0]; b=clone(x.to(device))[0]
            equivalence=dict(fp32_output_max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/a.norm()).item())
            assert equivalence['fp32_output_max_abs']<2e-4,equivalence
            mapped=[dict(method='spectral_rope',group=None,scope='bf16',**r) for r in evaluate(clone,args.task,device,args.eval_samples)]
            for group in [128,16]:
                clone.set_quant(dict(group=group,scope='qk',order='post'))
                mapped += [dict(method='spectral_rope',group=group,scope='qk',**r) for r in evaluate(clone,args.task,device,args.eval_samples)]
            # Per-head pre/post predictions are an extra exact commutation check on a trained model.
            model.set_quant(dict(group=128,scope='qk',order='pre'))
            ptq += [dict(method='vortex_pre',group=128,scope='qk',**r) for r in evaluate(model,args.task,device,args.eval_samples,quick=True)]
        result=dict(**metadata(config),model_config=asdict(cfg),dataset_version=VERSION,dataset_train_seed=args.seed+10000,
            dataset_test_seed_formula='900000+length*17 (same content RNG across distances)',train_data_sha256=digest.hexdigest(),
            parameters=sum(p.numel() for p in model.parameters()),train_tokens=total_tokens,optimizer_steps=args.steps,
            train_seconds=train_seconds,tokens_per_second=total_tokens/train_seconds,
            peak_memory_bytes=torch.musa.max_memory_allocated(),bf16=evalrows,ptq=ptq,spectral=mapped,
            spectral_fp32_equivalence=equivalence,eval_samples_per_condition=args.eval_samples)
        write_json(f'results/raw/{name}.json',result)
    print(name+' COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['rope_safe','vortex','ring_plain'],required=True)
    ap.add_argument('--task',choices=['relative','associative'],required=True)
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--layers',type=int,default=2,choices=[2,4])
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--backend',default='sdpa');ap.add_argument('--log-every',type=int,default=50)
    ap.add_argument('--eval-samples',type=int,default=256);args=ap.parse_args();train(args)


if __name__=='__main__':main()
