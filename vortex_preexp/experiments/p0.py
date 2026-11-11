import argparse
import json
import math
import platform
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from src import vortex, spectral
from src.model import Decoder,ModelConfig
from src.common import setup,sync,autocast,write_json,Meter,ROOT,metadata


def math_checks(device):
    gen=torch.Generator().manual_seed(2026)
    x=torch.randint(-30,31,(2,1,4,128),generator=gen)
    checks=[]
    for p in [-1000000,-58,-29,-1,0,1,11,29,58,1000000]:
        a=vortex.apply(x,*vortex.indices([p]))
        checks.append(torch.equal(a,vortex.reference(x,p)))
    for m,n in [(-58,29),(11,13),(1000000,1001334),(-19,-11)]:
        a=vortex.apply(vortex.apply(x,*vortex.indices([n])),*vortex.indices([-m]))
        b=vortex.apply(x,*vortex.indices([n-m]))
        checks.append(torch.equal(a,b))
        q=torch.randint(-20,20,x.shape,generator=gen)
        lhs=(vortex.apply(x,*vortex.indices([m]))*vortex.apply(q,*vortex.indices([n]))).sum(-1)
        rhs=(x*vortex.apply(q,*vortex.indices([n-m]))).sum(-1)
        checks.append(torch.equal(lhs,rhs))
    start=0
    for length in vortex.RINGS:
        a=vortex.apply(x,*vortex.indices([length]))
        b=vortex.apply(x,*vortex.indices([2*length]))
        checks += [torch.equal(a[...,start:start+length],-x[...,start:start+length]),
                   torch.equal(b[...,start:start+length],x[...,start:start+length])]
        start+=length
    u=spectral.basis(); orth=(u.T@u-torch.eye(128)).abs().max().item()
    eye=torch.eye(128).reshape(128,1,1,128)
    maxerr=0
    for p in [-1000000,-58,-1,0,1,11,1334,1000000]:
        perm=vortex.apply(eye,*vortex.indices([p]))[:,0,0].T.double()
        rot=spectral.rotate(eye,spectral.rotation_tables([p],'spectral_rope'))[:,0,0].T.double()
        maxerr=max(maxerr,(perm-u@rot@u.T).abs().max().item())
    q=torch.randn(4,32,4,128,generator=gen).to(device)
    k=torch.randn(4,32,4,128,generator=gen).to(device)
    ud=u.float().to(device)
    qv=vortex.apply(q,*vortex.indices(torch.arange(32),device=device)).transpose(1,2)
    kv=vortex.apply(k,*vortex.indices(torch.arange(32),device=device)).transpose(1,2)
    qs=spectral.rotate(q@ud,spectral.rotation_tables(torch.arange(32),'spectral_rope',device)).transpose(1,2)
    ks=spectral.rotate(k@ud,spectral.rotation_tables(torch.arange(32),'spectral_rope',device)).transpose(1,2)
    av=qv@kv.transpose(-1,-2)/128**.5; ar=qs@ks.transpose(-1,-2)/128**.5
    err=(av-ar).abs()
    model=Decoder(ModelConfig(method='vortex')).to(device).eval()
    clone=Decoder(ModelConfig(method='spectral_rope')).to(device).eval()
    clone.load_state_dict(spectral.convert_checkpoint({k:v.cpu() for k,v in model.state_dict().items()}))
    inputs=torch.randint(0,128,(2,64),generator=gen).to(device)
    with torch.no_grad():
        a=model(inputs)[0]; b=clone(inputs)[0]
    result=dict(integer_checks=checks,all_integer_pass=all(checks),orthogonality_max_abs=orth,
        decomposition_max_abs=maxerr,device_fp32_logit_max_abs=err.max().item(),
        device_fp32_logit_relative_l2=(err.norm()/av.norm()).item(),
        full_checkpoint_fp32_max_abs=(a-b).abs().max().item(),
        full_checkpoint_relative_l2=((a-b).norm()/a.norm()).item())
    assert all(checks) and orth<1e-12 and maxerr<1e-5 and err.max()<1e-4 and (a-b).abs().max()<1e-4,result
    return result


def environment(device):
    import torch_musa
    result=dict(python=platform.python_version(),torch=torch.__version__,torch_musa=torch_musa.__version__,
                gpu=str(torch.musa.get_device_properties(0)),musa_available=torch.musa.is_available())
    for name in ['musaInfo','mthreads-gmi']:
        result[name]=subprocess.run([name],capture_output=True,text=True).stdout
    checks={}
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        try:
            a=torch.randn(4,128,device=device,dtype=dtype,requires_grad=True)
            b=a.gather(-1,torch.arange(128,device=device).expand(4,-1))
            c=torch.einsum('bi,ci->bc',b,b)
            y=F.layer_norm(c,(4,)).float().square().mean(); y.backward(); sync()
            checks[str(dtype)]={'device':str(c.device),'backward_finite':bool(torch.isfinite(a.grad).all().item())}
        except Exception as e: checks[str(dtype)]={'error':str(e)}
    backend='sdpa'
    try:
        q=torch.randn(2,4,256,128,device=device,dtype=torch.bfloat16,requires_grad=True)
        y=F.scaled_dot_product_attention(q,q,q,is_causal=True)
        y.float().square().mean().backward(); sync()
        checks['sdpa']={'forward_device':str(y.device),'backward_finite':bool(torch.isfinite(q.grad).all().item())}
    except Exception as e:
        checks['sdpa']={'error':str(e)}; backend='eager'
    result.update(checks=checks,backend=backend)
    return result


def pilot(device, backend, seconds):
    rows=[]
    for method in ['rope_safe','vortex','ring_plain']:
        setup(); cfg=ModelConfig(method=method,backend=backend)
        model=Decoder(cfg).to(device).train()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
        x=torch.randint(0,128,(16,256),device=device); labels=x.roll(-1,1)
        torch.musa.reset_peak_memory_stats()
        start=time.perf_counter(); times=[]; steps=0
        while time.perf_counter()-start<seconds or steps<12:
            sync(); t=time.perf_counter(); opt.zero_grad(set_to_none=True)
            with autocast(): _,loss=model(x,labels)
            loss.backward(); opt.step(); sync()
            times.append(time.perf_counter()-t); steps+=1
        stable=np.array(times[5:]); rows.append(dict(method=method,batch=16,sequence=256,layers=2,
            parameters=sum(p.numel() for p in model.parameters()),steps=steps,seconds=sum(times),
            median_step_s=float(np.median(stable)),p95_step_s=float(np.quantile(stable,.95)),
            tokens_per_second=float(16*256/np.median(stable)),
            peak_memory_bytes=torch.musa.max_memory_allocated(),loss=loss.item(),backend=backend))
        del model,opt; torch.musa.empty_cache()
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pilot-seconds',type=int,default=70); args=ap.parse_args()
    device=setup()
    with Meter('P0'):
        env=environment(device); write_json('results/raw/p0_environment.json',env)
        result=math_checks(device); write_json('results/raw/p0_math.json',result); print(json.dumps(result),flush=True)
        deltas=[0,1,2,11,58,1333,1334,1335,14674,25346]
        corr={str(d):vortex.trace_correlation(d) for d in deltas}
        write_json('results/raw/p0_correlation.json',dict(selected=corr,curve=[vortex.trace_correlation(d) for d in range(2049)],
            ring_lengths=vortex.RINGS,ring_periods=[2*x for x in vortex.RINGS],joint_period=2*math.prod(vortex.RINGS)))
        rows=pilot(device,env['backend'],args.pilot_seconds)
        write_json('results/raw/p0_pilot.json',dict(**metadata(vars(args)),rows=rows)); print(json.dumps(rows),flush=True)
    print('P0 PASS',flush=True)


if __name__=='__main__':main()
