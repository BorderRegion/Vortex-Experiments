"""Dispatch evidence supplement; run after all other GPU jobs."""
import json
import torch
from torch.nn import functional as F
from src.common import setup,sync,Meter,write_json


def main():
    device=setup()
    rows=[]
    with Meter('P0_backend_confirmation'):
        for length in [256,2048]:
            q=torch.randn(2,4,length,128,device=device,dtype=torch.bfloat16,requires_grad=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:
                y=F.scaled_dot_product_attention(q,q,q,is_causal=True)
                y.float().square().mean().backward();sync()
            operators=[x.key for x in prof.key_averages() if any(s in x.key.lower() for s in ['attention','sdpa','flash','musa'])]
            rows.append(dict(length=length,dtype=str(y.dtype),output_device=str(y.device),
                backward_finite=bool(q.grad.isfinite().all().item()),dispatch_operators=operators))
        write_json('results/raw/backend_dispatch.json',dict(rows=rows,note='CPU profiler records operator dispatch; MUSA output/gradient checked. Latency measured separately by MUSA Events.'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
