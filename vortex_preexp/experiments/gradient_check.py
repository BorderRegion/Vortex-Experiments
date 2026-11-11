"""Check MUSA sparse-supervision CE and SDPA gradients against CPU references."""
import torch
from torch.nn import functional as F
from src.common import setup,sync,Meter,write_json


def compare(a,b):
    a,b=a.float().cpu(),b.float().cpu()
    return dict(max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/a.norm().clamp_min(1e-12)).item(),
                norm_ratio=(b.norm()/a.norm().clamp_min(1e-12)).item())


def main():
    device=setup();rows=[];gen=torch.Generator().manual_seed(1701)
    with Meter('P0_sparse_gradient_confirmation'):
        for dtype in [torch.float32,torch.bfloat16]:
            for sparse in [False,True]:
                x=torch.randn(2,128,128,generator=gen).to(dtype)
                labels=torch.randint(0,128,(2,128),generator=gen)
                if sparse:labels[:,:-1]=-100
                a=x.clone().requires_grad_();b=x.to(device).requires_grad_()
                la=F.cross_entropy(a.float().reshape(-1,128),labels.reshape(-1),ignore_index=-100)
                lb=F.cross_entropy(b.float().reshape(-1,128),labels.to(device).reshape(-1),ignore_index=-100)
                la.backward();lb.backward();sync()
                r=dict(operator='cross_entropy',dtype=str(dtype),last_query_only=sparse,
                       cpu_loss=la.item(),musa_loss=lb.item(),gradient=compare(a.grad,b.grad))
                r['passed']=abs(la.item()-lb.item())<2e-5 and r['gradient']['relative_l2']<.005
                rows.append(r)
            for sparse in [False,True]:
                data=[torch.randn(2,2,128,128,generator=gen).to(dtype) for _ in range(3)]
                cpu=[x.clone().requires_grad_() for x in data];musa=[x.to(device).requires_grad_() for x in data]
                ca=F.scaled_dot_product_attention(*cpu,is_causal=True)
                ma=F.scaled_dot_product_attention(*musa,is_causal=True)
                ca_loss=ca[:,:,-1].float().square().mean() if sparse else ca.float().square().mean()
                ma_loss=ma[:,:,-1].float().square().mean() if sparse else ma.float().square().mean()
                ca_loss.backward();ma_loss.backward();sync()
                r=dict(operator='causal_sdpa',dtype=str(dtype),last_query_only=sparse,
                    output=compare(ca,ma),gradients={name:compare(x.grad,y.grad) for name,x,y in zip(['q','k','v'],cpu,musa)})
                tolerance=.035 if dtype==torch.bfloat16 else .0002
                r['passed']=r['output']['relative_l2']<tolerance and all(v['relative_l2']<tolerance for v in r['gradients'].values())
                rows.append(r)
        passed=all(r['passed'] for r in rows)
        write_json('results/raw/sparse_gradient_check.json',dict(passed=passed,rows=rows,
            note='CPU and MUSA receive identical FP32 or BF16-rounded inputs; no training checkpoint changed.'))
        assert passed,rows
    print('CPU/MUSA sparse CE and causal SDPA gradient agreement PASS',flush=True)


if __name__=='__main__':main()
