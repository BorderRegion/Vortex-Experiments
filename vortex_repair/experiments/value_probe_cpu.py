"""Heldout linear value recoverability is a probe, not retrieval performance."""
import json
import torch
from src.model import Decoder,ModelConfig
from src.synthetic import associative
from src.common import ROOT,write_json
from .routing import targets


def main():
    torch.set_num_threads(4)
    ckpt=torch.load(ROOT/'results/checkpoints/routing_rope_safe_s2026_step3000.pt',map_location='cpu',weights_only=False)
    model=Decoder(ModelConfig(**ckpt['model_config']));model.load_state_dict(ckpt['state']);model.eval()
    gen=torch.Generator().manual_seed(90711)
    xs=[];ys=[];mass=correct=value_hit=0.
    with torch.no_grad():
        for _ in range(8):
            x,y=associative(16,256,gen,distance=32,evaluation=True)
            h=model.blocks[0](model.embedding(x));block=model.blocks[1];attn=block.attn
            q,k,_=attn.qkv(block.ln1(h)).reshape(16,256,3,4,128).unbind(2)
            q,k=attn.position(q),attn.position(k)
            p=(q[:,-1].unsqueeze(2)@k.permute(0,2,3,1)/128**.5).softmax(-1)
            mass+=p[:,:,:,224].mean().item()
            logits=model(x)[0][:,-1];correct+=(logits.argmax(-1)==y[:,-1]).float().mean().item()
            local,_=targets(x);feat=h[local[0],local[1]];own=x[local[0],local[1]]
            value_hit+=(model.output(model.norm(feat)).argmax(-1)==own).float().mean().item()
            xs.append(block.ln1(feat));ys.append(own-36)
        features=torch.cat(xs);labels=torch.cat(ys);n=len(features)//2
        train=torch.cat((features[:n],torch.ones(n,1)),1)
        test=torch.cat((features[n:],torch.ones(len(features)-n,1)),1)
        coef=torch.linalg.solve(train.T@train+.01*torch.eye(513),train.T@torch.nn.functional.one_hot(labels[:n],32).float())
        row=dict(checkpoint_step=3000,cpu_ordinary_accuracy=correct/8,actual_target_attention_mass=mass/8,
                 first_layer_existing_value_readout=value_hit/8,
                 first_layer_value_ridge_heldout_accuracy=(test@coef).argmax(-1).eq(labels[n:]).float().mean().item(),
                 boundary='CPU FP32 diagnostic; ridge fits first half and evaluates heldout second half; not task accuracy.')
    write_json('results/raw/route3000_value_diagnostic_cpu.json',row)
    print(json.dumps(row),flush=True)


if __name__=='__main__': main()
