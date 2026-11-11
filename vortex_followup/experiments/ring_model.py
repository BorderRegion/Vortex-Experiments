"""Verify variable-length quantization through the actual decoder inference path."""
from pathlib import Path
from dataclasses import asdict
import torch
from src.model import Decoder, ModelConfig
from src.ring_quant import GROUPS
from src import spectral
from src.synthetic import relative
from src.common import setup, autocast, Meter, write_json


@torch.no_grad()
def main():
    device=setup();rows=[]
    ckpt=torch.load(Path('/root/autodl-tmp/vortex_preexp/results/checkpoints/p2_relative_vortex_s2026_final.pt'),map_location='cpu',weights_only=False)
    model=Decoder(ModelConfig(**ckpt['model_config'])).to(device);model.load_state_dict(ckpt['state']);model.eval()
    clone=Decoder(ModelConfig(**{**ckpt['model_config'],'method':'spectral_rope'})).to(device)
    clone.load_state_dict(spectral.convert_checkpoint(ckpt['state']));clone.eval()
    with Meter('followup_ring_groups_whole_model'):
        for length in (256,512,2048):
            x,y=relative(64,length,torch.Generator().manual_seed(690000+length),distance=32,evaluation=True)
            x,y=x.to(device),y.to(device)
            model.set_quant(None)
            with autocast():
                baseline,loss=model(x,y)
            model.set_quant(dict(group=GROUPS,scope='qk',order='pre'))
            with autocast():
                pre,pre_loss=model(x,y)
            model.set_quant(dict(group=GROUPS,scope='qk',order='post'))
            clone.set_quant(dict(group=GROUPS,scope='qk',order='post'))
            with autocast():
                post,post_loss=model(x,y);mapped,mapped_loss=clone(x,y)
            diff=(pre-post).abs().max().item()
            assert diff==0.,diff
            rows.append(dict(length=length,distance=32,samples=64,pre_post_max_abs=diff,
                baseline_accuracy=(baseline[:,-1].argmax(-1)==y[:,-1]).float().mean().item(),
                ring_quant_accuracy=(post[:,-1].argmax(-1)==y[:,-1]).float().mean().item(),
                spectral_ring_quant_accuracy=(mapped[:,-1].argmax(-1)==y[:,-1]).float().mean().item(),
                baseline_nll=loss.item(),ring_quant_nll=post_loss.item(),spectral_ring_quant_nll=mapped_loss.item()))
    write_json('results/raw/ring_model.json',dict(rows=rows,groups=GROUPS,scales_per_head=7,
        limitation='Existing relative checkpoint; one distance and64 samples per length. This ceiling-prone task is a decoder integration/commutation check, not evidence of quantized retrieval or language quality.'))
    print('Whole-model ring-group pre/post logits exactly equal',flush=True)


if __name__=='__main__':
    main()
