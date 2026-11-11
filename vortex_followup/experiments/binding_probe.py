"""Separate local binding, query identity, representation matching and routing."""
import argparse
import torch
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import ROOT, setup, autocast, Meter, write_json
from .binding import binding_targets


@torch.no_grad()
def probe(model, device, length):
    gen=torch.Generator().manual_seed(493100+length)
    n=local_n=local_ok=query_ok=cosine_ok=0
    target_key_mass=target_value_mass=0.
    for _ in range(16):
        x,y=associative(16,length,gen,distance=32,evaluation=True)
        labels=binding_targets(x); valid=labels!=-100
        with autocast():
            h=model.blocks[0](model.embedding(x.to(device)))
            local_logits=model.output(model.norm(h[valid.to(device)])).float()
            query_logits=model.output(model.norm(h[:,-1])).float()
            attn=model.blocks[1].attn
            q,k,_=attn.qkv(model.blocks[1].ln1(h)).reshape(16,length,3,4,128).unbind(2)
            q,k=attn.position(q),attn.position(k)
        local_ok+=(local_logits.argmax(-1).cpu()==labels[valid]).sum().item()
        local_n+=int(valid.sum())
        # Restrict to key vocabulary to assess recoverability, separately from full-vocab output.
        query_ok+=(query_logits[:,4:36].argmax(-1).cpu()+4==x[:,-1]).sum().item()
        h=h.float(); normalized=h/h.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        scores=(normalized[:,-1:]*normalized).sum(-1).masked_fill(~valid.to(device),-float('inf'))
        cosine_ok+=(x.gather(1,scores.argmax(-1).cpu()[:,None]).squeeze(1)==y[:,-1]).sum().item()
        p=((q[:,-1].float().unsqueeze(2)@k.float().transpose(1,2).transpose(-1,-2))/128**.5).softmax(-1)
        target_key_mass+=p[:,:,:,length-33].sum().item()
        target_value_mass+=p[:,:,:,length-32].sum().item()
        n+=16
    return dict(length=length,distance=32,samples=n,first_layer_binding_accuracy=local_ok/local_n,
                first_layer_query_key_recovery=query_ok/n,first_layer_cosine_value_retrieval=cosine_ok/n,
                second_layer_target_key_mass=target_key_mass/(n*4),second_layer_target_value_mass=target_value_mass/(n*4))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pattern',default='binding_*_final.pt');ap.add_argument('--output',default='binding_probe');args=ap.parse_args()
    device=setup();rows=[]
    with Meter('followup_binding_path_probe'):
        for p in sorted((ROOT/'results/checkpoints').glob(args.pattern)):
            ckpt=torch.load(p,map_location='cpu',weights_only=False)
            model=Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']);model.eval()
            for length in (256,512):
                rows.append(dict(checkpoint=p.stem,**probe(model,device,length)))
    write_json(f'results/raw/{args.output}.json',dict(rows=rows,
        limitation='Shared decoder probes and raw cosine matching are descriptive; they do not establish absence of information under all possible decoders.'))
    print('Binding path probe complete',flush=True)


if __name__=='__main__':
    main()
