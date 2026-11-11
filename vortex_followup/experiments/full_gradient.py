"""Whole-model last-query gradients at the actual training lengths, CPU vs MUSA."""
from dataclasses import asdict
import torch
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import setup, autocast, Meter, write_json


def main():
    device=setup();rows=[]
    with Meter('followup_full_model_gradient_check'):
        for method in ('rope_safe','vortex'):
            for length in (256,512):
                for bf16 in (False,True):
                    cfg=ModelConfig(method=method,backend='eager')
                    cpu=Decoder(cfg);musa=Decoder(ModelConfig(**{**asdict(cfg),'backend':'sdpa'})).to(device)
                    musa.load_state_dict(cpu.state_dict())
                    x,y=associative(2,length,torch.Generator().manual_seed(49827),distance=32,evaluation=True)
                    with autocast('cpu',enabled=bf16):
                        a,la=cpu(x,y)
                    with autocast(enabled=bf16):
                        b,lb=musa(x.to(device),y.to(device))
                    la.backward();lb.backward()
                    gradient=[];reference=[];modules=[]
                    for (name,pa),(_,pb) in zip(cpu.named_parameters(),musa.named_parameters()):
                        u,v=pa.grad.float().flatten(),pb.grad.float().cpu().flatten()
                        gradient.append(v);reference.append(u)
                        if name.endswith('qkv.weight'):
                            modules.append(dict(name=name,relative_l2=((u-v).norm()/u.norm().clamp_min(1e-12)).item(),norm_ratio=(v.norm()/u.norm().clamp_min(1e-12)).item()))
                    u,v=torch.cat(reference),torch.cat(gradient)
                    rel=((u-v).norm()/u.norm()).item()
                    row=dict(method=method,length=length,bf16=bf16,cpu_loss=la.item(),musa_loss=lb.item(),
                             loss_abs_difference=abs(la.item()-lb.item()),gradient_relative_l2=rel,
                             gradient_norm_ratio=(v.norm()/u.norm()).item(),qkv_gradients=modules,
                             passed=rel<(.05 if bf16 else .01) and abs(la.item()-lb.item())<(.04 if bf16 else .005))
                    rows.append(row)
        write_json('results/raw/full_gradient.json',dict(rows=rows,passed=all(r['passed'] for r in rows),
                   reference='Identical initial weights and original last-query data. CPU eager attention versus MUSA SDPA; actual length256/512,2layer512wide models.'))
        assert all(r['passed'] for r in rows),rows
    print('Whole-model gradients at actual sequence lengths PASS',flush=True)


if __name__=='__main__':
    main()
