"""Paired numerical experiment; each method compared against its own FP32 operator."""
import argparse
import json
import time
import torch
from src import vortex,spectral
from src.quant import quantize
from src.common import setup,sync,write_json,metadata,Meter,ROOT

DELTAS=[1,2,8,32,128,512,1333,1334,1335]
OFFSETS=[0,256,4096,65536,1000000]


def metric(score,reference,values):
    err=score-reference
    p=reference.softmax(1); logp=reference.log_softmax(1); logq=score.log_softmax(1)
    out=torch.einsum('bsh,bshd->bhd',score.softmax(1),values)
    refout=torch.einsum('bsh,bshd->bhd',p,values)
    return dict(logit_mae=err.abs().mean().item(),logit_nrmse=(err.square().mean()/reference.square().mean().clamp_min(1e-16)).sqrt().item(),
        softmax_kl=(p*(logp-logq)).sum(1).mean().item(),top1_change=(score.argmax(1)!=reference.argmax(1)).float().mean().item(),
        output_nrmse=((out-refout).square().mean()/refout.square().mean().clamp_min(1e-16)).sqrt().item(),
        per_delta={str(d):dict(mae=err[:,i].abs().mean().item(),nrmse=(err[:,i].square().mean()/reference[:,i].square().mean().clamp_min(1e-16)).sqrt().item()) for i,d in enumerate(DELTAS)})


def positions(x,positions,method):
    if method in ('vortex','ring_plain'):
        return vortex.apply(x,*vortex.indices(positions,method=='vortex',x.device))
    return spectral.rotate(x,spectral.rotation_tables(positions,method,x.device))


@torch.no_grad()
def run_distribution(q,k,values,name,device):
    assert torch.equal(q,q[:,:1].expand_as(q)), 'A candidate attention row must share one query content.'
    rows=[]; drift={}; u=spectral.basis().float().to(device)
    for method in ['rope_safe','vortex_post','vortex_pre','spectral_rope']:
        operator='vortex' if method.startswith('vortex') else method
        q0,k0=(q@u,k@u) if method=='spectral_rope' else (q,k)
        references=[]
        for offset in OFFSETS:
            qp=positions(q0,torch.full((len(DELTAS),),offset),operator)
            kp=positions(k0,offset-torch.tensor(DELTAS),operator)
            ref=(qp*kp).sum(-1)/128**.5
            references.append(ref)
            for kind,group in [('int4',128),('int4',16),('fp4',16),('bf16',128)]:
                for scope in (['qk','k'] if kind!='bf16' else ['qk']):
                    if kind=='bf16':
                        qa,ka=qp.bfloat16().float(),kp.bfloat16().float()
                    elif method=='vortex_pre':
                        qa=positions(quantize(q0,group,kind),torch.full((len(DELTAS),),offset),operator) if scope=='qk' else qp
                        ka=positions(quantize(k0,group,kind),offset-torch.tensor(DELTAS),operator)
                    else:
                        qa=quantize(qp,group,kind) if scope=='qk' else qp
                        ka=quantize(kp,group,kind)
                    score=(qa.float()*ka.float()).sum(-1)/128**.5
                    key=(method,kind,group,scope)
                    if offset==OFFSETS[0]: drift[key]=score.clone()
                    row=dict(distribution=name,method=method,kind=kind,group=group,scope=scope,offset=offset,
                             queries=q.shape[0]*q.shape[2],candidate_distances=DELTAS,
                             absolute_shift_score_mae=(score-drift[key]).abs().mean().item(),
                             absolute_shift_score_max=(score-drift[key]).abs().max().item(),**metric(score,ref,values))
                    rows.append(row)
        rows.append(dict(distribution=name,method=method,kind='fp32_shift_diagnostic',
                         max_absolute_shift=max((r-references[0]).abs().max().item() for r in references)))
    # Quantizer commutation uses identical values before/after signed reindexing.
    comm=[]
    for kind,group in [('int4',128),('int4',16),('fp4',16)]:
        for p in [1,11,32,128,1334,1000000]:
            src,sign=vortex.indices([p]*k.shape[1],device=device)
            a,ac,_=quantize(vortex.apply(k,src,sign),group,kind,True)
            b,bc,_=quantize(k,group,kind,True)
            b=vortex.apply(b,src,sign)
            bc=vortex.apply(bc,src,sign)
            comm.append(dict(distribution=name,kind=kind,group=group,position=p,
                code_match_rate=(ac==bc).float().mean().item(),dequant_max_abs=(a-b).abs().max().item(),
                dequant_mae=(a-b).abs().mean().item(),dequant_exact_rate=(a==b).float().mean().item()))
    return rows,comm


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples',type=int,default=1024)
    ap.add_argument('--real',default=None); ap.add_argument('--name',default='p1'); args=ap.parse_args()
    device=setup(); rows=[]; comm=[]
    with Meter(args.name):
        generator=torch.Generator().manual_seed(2026)
        if args.real:
            source=torch.load(ROOT/args.real,map_location='cpu',weights_only=False)
            q0=source['q'].reshape(-1,4,128); k0=source['k'].reshape(-1,4,128)
            sample_order=torch.randperm(len(q0),generator=generator)
            q0=q0[sample_order];k0=k0[sample_order]
            n=min(args.samples,len(q0)//len(DELTAS)); n=max(n,1)
            q=q0[:n,None].expand(-1,len(DELTAS),-1,-1).to(device)
            k=k0[:n*len(DELTAS)].reshape(n,len(DELTAS),4,128).to(device)
            distributions=[(args.real,q,k)]
        else:
            shape=(args.samples,len(DELTAS),4,128)
            q=torch.randn((args.samples,1,4,128),generator=generator).expand(-1,len(DELTAS),-1,-1)
            k=torch.randn(shape,generator=generator)
            distributions=[('isotropic',q.to(device),k.to(device))]
            tq=q[:,:1].clone(); tk=k.clone()
            tq*=torch.where(torch.rand(tq.shape,generator=generator)<.015,12.,1.)
            tk*=torch.where(torch.rand(tk.shape,generator=generator)<.015,12.,1.)
            distributions.append(('outliers_1.5pct_x12',tq.expand(-1,len(DELTAS),-1,-1).to(device),tk.to(device)))
        for name,q,k in distributions:
            values=torch.randn((*k.shape[:-1],32),generator=generator).to(device)
            r,c=run_distribution(q,k,values,name,device); rows+=r; comm+=c
            print(json.dumps(dict(distribution=name,rows=len(r),commutation_rows=len(c))),flush=True)
        write_json(f'results/raw/{args.name}.json',dict(**metadata(vars(args)),rows=rows,commutation=comm,
            precision_note='FP32 reference; software INT4/FP4. Q/K quantized once. CPU FP64 angles reduced before FP32 tables.'))
    print(args.name+' PASS',flush=True)


if __name__=='__main__':main()
