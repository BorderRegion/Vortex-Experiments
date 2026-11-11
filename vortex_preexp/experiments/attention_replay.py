"""Actual first-layer last-query attention, retaining content, positions and V.

The standard RoPE row is a same-content operator diagnostic, not a model swap
quality evaluation. Vortex and spectral use equivalent FP32 attention functions.
"""
from pathlib import Path
import torch
from src import spectral,vortex
from src.model import Decoder,ModelConfig
from src.synthetic import make_batch
from src.quant import quantize
from src.common import ROOT,setup,autocast,Meter,metadata,write_json


def position(x,pos,method):
    if method=='vortex':return vortex.apply(x,*vortex.indices(pos,device=x.device))
    return spectral.rotate(x,spectral.rotation_tables(pos,method,x.device))


def measures(score,reference,values):
    diff=score-reference
    p=reference.softmax(1);l=reference.log_softmax(1);lq=score.log_softmax(1)
    a=torch.einsum('bsh,bshd->bhd',score.softmax(1),values)
    b=torch.einsum('bsh,bshd->bhd',p,values)
    return dict(logit_mae=diff.abs().mean().item(),
        logit_nrmse=(diff.square().mean()/reference.square().mean().clamp_min(1e-16)).sqrt().item(),
        softmax_kl=(p*(l-lq)).sum(1).mean().item(),
        top1_change=(score.argmax(1)!=reference.argmax(1)).float().mean().item(),
        output_nrmse=((a-b).square().mean()/b.square().mean().clamp_min(1e-16)).sqrt().item())


@torch.no_grad()
def main():
    device=setup();rows=[];equivalences=[]
    sources=[('relative','p2_relative_vortex_s2026'),('associative','p2b_associative_vortex_s2026')]
    gate=ROOT/'results/raw/p2c_gate.json'
    if gate.exists():
        import json
        if json.loads(gate.read_text())['passed']:sources.append(('associative','p2c_associative_vortex_s2026'))
    with Meter('P1_actual_attention_replay'):
      for task,name in sources:
        ckpt=torch.load(ROOT/f'results/checkpoints/{name}_final.pt',map_location='cpu',weights_only=False)
        model=Decoder(ModelConfig(**ckpt['model_config'])).to(device)
        model.load_state_dict(ckpt['state']);model.eval()
        for length in [256,512,2048]:
            samples=64;gen=torch.Generator().manual_seed(771000+length)
            x,_=make_batch(task,samples,length,gen,distance=32,evaluation=True)
            with autocast():
                block=model.blocks[0]
                z=block.ln1(model.embedding(x.to(device)))
                q,k,v=block.attn.qkv(z).reshape(samples,length,3,4,128).unbind(2)
            q=q[:,-1:].float();k=k.float();v=v.float()
            pos=torch.arange(length);qp=torch.tensor([length-1]);refs={}
            u=spectral.basis().float().to(device)
            for method in ['rope_safe','vortex_post','vortex_pre','spectral_rope']:
                operator='vortex' if method.startswith('vortex') else method
                q0,k0=(q@u,k@u) if method=='spectral_rope' else (q,k)
                qa=position(q0,qp,operator);ka=position(k0,pos,operator)
                reference=(qa*ka).sum(-1)/128**.5;refs[method]=reference
                for group in [128,16]:
                  for scope in ['qk','k']:
                    if method=='vortex_pre':
                        qz=position(quantize(q0,group),qp,operator) if scope=='qk' else qa
                        kz=position(quantize(k0,group),pos,operator)
                    else:
                        qz=quantize(qa,group) if scope=='qk' else qa
                        kz=quantize(ka,group)
                    score=(qz*kz).sum(-1)/128**.5
                    rows.append(dict(source=name,task=task,length=length,distance=32,
                        samples=samples,head_queries=samples*4,layer=0,method=method,
                        kind='int4',group=group,scope=scope,**measures(score,reference,v)))
            diff=refs['vortex_post']-refs['spectral_rope']
            eq=dict(source=name,length=length,max_abs=diff.abs().max().item(),
                relative_l2=(diff.norm()/refs['vortex_post'].norm().clamp_min(1e-12)).item())
            assert eq['relative_l2']<2e-5,eq
            equivalences.append(eq)
        del model
      write_json('results/raw/attention_replay.json',dict(**metadata(dict(seed=2026)),rows=rows,
        fp32_equivalence=equivalences,description='First-layer BF16 projection inputs promoted to FP32, actual last-query Q, all preceding and current keys, actual V and actual positions; 64 fixed unseen sequences per condition. All keys are causal for the last query. Not a real-text or whole-model accuracy test.'))
    print('Actual causal attention replay complete',flush=True)


if __name__=='__main__':main()
