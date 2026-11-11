"""Oracle routing is diagnostic only; ordinary inference never uses locations."""
import json
import torch
from torch.nn import functional as F
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import ROOT, setup, autocast, Meter, write_json


@torch.no_grad()
def intervene(model, device, length, samples=256):
    modes = ('ordinary', 'oracle_value', 'wrong_value')
    stats = {mode: dict(correct=0, nll=0.) for mode in modes}
    gen = torch.Generator().manual_seed(864300+length)
    masses = []
    for start in range(0, samples, 16):
        n = min(16, samples-start)
        x, labels = associative(n, length, gen, distance=32, evaluation=True)
        target = length-32
        # A real distractor with a different value, not arbitrary filler.
        wrong = []
        for row, answer in zip(x, labels[:, -1]):
            eligible = ((row >= 36) & (row < 68) & (row != answer)).nonzero().flatten()
            assert len(eligible)
            wrong.append(eligible[0].item())
        wrong = torch.tensor(wrong, device=device)
        y = labels[:, -1].to(device)
        with autocast():
            h = model.blocks[0](model.embedding(x.to(device)))
            block = model.blocks[1]; attn = block.attn
            q,k,v = attn.qkv(block.ln1(h)).reshape(n,length,3,4,128).unbind(2)
            q,k = attn.position(q),attn.position(k)
            q,k,v = [t.transpose(1,2).contiguous() for t in (q,k,v)]
            p = ((q[:,:,-1:].float() @ k.float().transpose(-1,-2))/128**.5).softmax(-1)
            masses.append(p[:,:,:,target].mean().item())
            outputs = dict(ordinary=F.scaled_dot_product_attention(q[:,:,-1:],k,v),
                           oracle_value=v[:,:,target:target+1],
                           wrong_value=v[torch.arange(n,device=device),:,wrong].unsqueeze(2))
            for mode, out in outputs.items():
                z = h[:,-1:] + attn.proj(out.transpose(1,2).reshape(n,1,512))
                z = z+block.ffn(block.ln2(z))
                logits = model.output(model.norm(z))[:,0].float()
                stats[mode]['correct'] += (logits.argmax(-1)==y).sum().item()
                stats[mode]['nll'] += F.cross_entropy(logits,y,reduction='sum').item()
    return dict(length=length,distance=32,samples=samples,target_value_mass=sum(masses)/len(masses),
                modes={m:dict(accuracy=s['correct']/samples,nll=s['nll']/samples) for m,s in stats.items()})


def main():
    device=setup(); rows=[]
    checkpoints = [
        ROOT.parent/'vortex_followup/results/checkpoints/binding_rope_safe_aligned_s2026_final.pt',
        ROOT.parent/'vortex_followup/results/checkpoints/supervision_rope_safe_sparse_s2026_final.pt',
    ]
    with Meter('repair_causal_routing_intervention'):
        for path in checkpoints:
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            model=Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']); model.eval()
            for length in (256,512):
                row=dict(checkpoint=path.stem,**intervene(model,device,length))
                rows.append(row); print(json.dumps(row),flush=True)
    write_json('results/raw/intervention.json',dict(rows=rows,
        boundary='Oracle and wrong-value interventions replace final-query second-layer attention with one-hot routing. They are diagnostics, not deployable accuracy.'))


if __name__=='__main__': main()
