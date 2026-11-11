"""Ordinary inference, counterfactual queries, matched PTQ and mapped spectrum."""
import argparse
import json
import torch
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src import spectral
from src.ring_quant import GROUPS
from src.common import ROOT, setup, autocast, Meter, write_json
from .supervision import evaluate
from .query_swap import probe as query_swap


@torch.no_grad()
def fresh_test(model,device,samples=512):
    rows=[]; model.eval()
    for length in (256,512):
        for distance in (8,32,128):
            gen=torch.Generator().manual_seed(719031+length*17)
            correct=nll=0.
            for start in range(0,samples,16):
                n=min(16,samples-start)
                x,y=associative(n,length,gen,distance=distance,evaluation=True)
                with autocast(): logits,loss=model(x.to(device),y.to(device))
                correct+=(logits[:,-1].argmax(-1)==y[:,-1].to(device)).sum().item()
                nll+=loss.item()*n
            rows.append(dict(length=length,distance=distance,samples=samples,accuracy=correct/samples,nll=nll/samples))
    return rows


@torch.no_grad()
def equivalence(model,mapped,device):
    rows=[]
    for length in (256,512,2048):
        x,_=associative(16,length,torch.Generator().manual_seed(603791+length),distance=32,evaluation=True)
        x=x.to(device)
        model.set_quant(None);mapped.set_quant(None)
        a=model(x)[0];b=mapped(x)[0]
        error=(a-b).abs().max().item();rms=(a-b).square().mean().sqrt().item()
        assert torch.allclose(a,b,atol=.005,rtol=.001),(length,error)
        with autocast():
            model.set_quant(dict(group=GROUPS,scope='qk',order='pre'))
            pre=model(x)[0]
            model.set_quant(dict(group=GROUPS,scope='qk',order='post'))
            post=model(x)[0]
        commute=(pre-post).abs().max().item()
        assert commute==0,commute
        rows.append(dict(length=length,samples=16,fp32_mapped_max_abs=error,fp32_mapped_rms=rms,
                         fp32_mapped_argmax_agreement=(a[:,-1].argmax(-1)==b[:,-1].argmax(-1)).float().mean().item(),
                         ring_quant_pre_post_max_abs=commute))
    model.set_quant(None);mapped.set_quant(None)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pattern',default='routing_value_*_step12000.pt');args=ap.parse_args()
    device=setup(); outputs=[]
    with Meter('repair_final_ordinary_inference_validation'):
        for path in sorted((ROOT/'results/checkpoints').glob(args.pattern)):
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            model=Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']);model.eval()
            assert not model._forward_hooks
            assert all(not b.attn._forward_pre_hooks for b in model.blocks)
            row=dict(checkpoint=path.stem,fresh_test=fresh_test(model,device),query_swap=query_swap(model,device),ptq=[])
            for group in (128,16,GROUPS):
                model.set_quant(dict(group=group,scope='qk',order='post'))
                row['ptq'].append(dict(group=group,kind='simulated_INT4',rows=evaluate(model,device,256,True)))
            model.set_quant(None)
            if model.cfg.method=='vortex':
                mapped=Decoder(ModelConfig(**{**ckpt['model_config'],'method':'spectral_rope'})).to(device)
                mapped.load_state_dict(spectral.convert_checkpoint(ckpt['state']));mapped.eval()
                row['equivalence']=equivalence(model,mapped,device)
                row['mapped_spectral']=[dict(group=None,rows=evaluate(mapped,device,256,True))]
                for group in (128,16,GROUPS):
                    mapped.set_quant(dict(group=group,scope='qk',order='post'))
                    row['mapped_spectral'].append(dict(group=group,rows=evaluate(mapped,device,256,True)))
            outputs.append(row)
            write_json(f'results/raw/{path.stem}_validation.json',row)
            print(json.dumps(dict(checkpoint=path.stem,validation='complete',query_swap=row['query_swap'])),flush=True)
    write_json('results/raw/validation.json',dict(rows=outputs,
        boundary='Fresh-test seed719031+17L was not used for training or interim evaluation. All accuracy uses ordinary Decoder.forward; no training hooks or oracle positions. INT4 is fake quantization, not native acceleration.'))


if __name__=='__main__': main()
