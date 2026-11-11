import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from src import vortex,spectral
from src.common import setup,sync,write_json,metadata,Meter


def benchmark(fn,warmup,repeats):
    for _ in range(warmup):fn()
    sync();torch.musa.reset_peak_memory_stats();wall=[];events=[]
    event_available=hasattr(torch.musa,'Event')
    for _ in range(repeats):
        a=torch.musa.Event(enable_timing=True) if event_available else None
        b=torch.musa.Event(enable_timing=True) if event_available else None
        sync();t=time.perf_counter()
        if a:a.record()
        result=fn()
        if b:b.record()
        sync();wall.append((time.perf_counter()-t)*1000)
        if a:events.append(a.elapsed_time(b))
    return dict(median_ms=float(np.median(wall)),p95_ms=float(np.quantile(wall,.95)),
        gpu_event_median_ms=float(np.median(events)) if events else None,
        gpu_event_p95_ms=float(np.quantile(events,.95)) if events else None,
        peak_memory_bytes=torch.musa.max_memory_allocated(),wall_samples_ms=wall,event_samples_ms=events)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--repeats',type=int,default=60);ap.add_argument('--warmup',type=int,default=15)
    args=ap.parse_args();device=setup();rows=[]
    with Meter('P4'):
        for length in [128,512,2048]:
            q,k,v=[torch.randn(args.batch,length,4,128,device=device,dtype=torch.bfloat16) for _ in range(3)]
            for method in ['rope_safe','vortex']:
                tables=(vortex.indices(torch.arange(length),device=device) if method=='vortex' else
                        spectral.rotation_tables(torch.arange(length),method,device))
                def pos(x):return vortex.apply(x,*tables) if method=='vortex' else spectral.rotate(x,tables)
                def op():return pos(q),pos(k)
                def prefill():
                    a,b=op()
                    return F.scaled_dot_product_attention(a.transpose(1,2).contiguous(),b.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),is_causal=True)
                for path,fn in [('position_qk',op),('prefill_attention',prefill)]:
                    row=dict(method=method,path=path,batch=args.batch,sequence=length,dtype='bfloat16',backend='sdpa',**benchmark(fn,args.warmup,args.repeats))
                    rows.append(row);print(json.dumps({k:v for k,v in row.items() if not k.endswith('samples_ms')}),flush=True)
                if length in [512,2048]:
                    cache=pos(k).transpose(1,2).contiguous();vcache=v.transpose(1,2).contiguous();query=q[:,-1:]
                    qt=(vortex.indices([length-1],device=device) if method=='vortex' else spectral.rotation_tables([length-1],method,device))
                    def decode():
                        a=vortex.apply(query,*qt) if method=='vortex' else spectral.rotate(query,qt)
                        return F.scaled_dot_product_attention(a.transpose(1,2).contiguous(),cache,vcache,is_causal=False)
                    rows.append(dict(method=method,path='decode_attention',batch=args.batch,sequence=length,dtype='bfloat16',backend='sdpa',cache_prepared_outside_timing=True,**benchmark(decode,args.warmup,args.repeats)))
                del pos,op,prefill
            del q,k,v
        write_json('results/raw/p4.json',dict(**metadata(vars(args)),rows=rows,
            timing='MUSA Events and synchronized perf_counter; tables and K cache precomputed; no native low-bit kernels',
            compile='not attempted: no proven stable optimization path needed for this feasibility study'))
    print('P4 COMPLETE',flush=True)


if __name__=='__main__':main()
