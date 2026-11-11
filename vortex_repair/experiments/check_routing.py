"""Check newly introduced training targets, masked CE and its gradients."""
import torch
from torch.nn import functional as F
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import setup, Meter, write_json
from .routing import targets, route_loss, auxiliary_weight


def main():
    device=setup(); rows=[]
    with Meter('repair_routing_implementation_checks'):
        gen=torch.Generator().manual_seed(774013)
        for length in (256,512,1024,2048):
            for distance in (8,32,128,512,1333,1334,1335):
                if distance>=length-1: continue
                x,y=associative(16,length,gen,distance=distance,evaluation=True)
                local,retrieval=targets(x)
                assert torch.all(x[local[0],local[1]-1]==x[local[0],local[2]])
                assert torch.all(retrieval==length-distance)
                assert torch.equal(x[torch.arange(16),retrieval],y[:,-1])
                rows.append(dict(check='input_only_targets',length=length,distance=distance,passed=True))
        for method in ('rope_safe','vortex','ring_plain'):
            model=Decoder(ModelConfig(method=method)).cpu()
            attn=model.blocks[0].attn
            z=torch.randn(2,32,512,requires_grad=True)
            batch=torch.tensor([0,0,1,1]); query=torch.tensor([7,13,10,21]); target=query-1
            sparse,_=route_loss(attn,z,batch,query,target)
            grad_sparse=torch.autograd.grad(sparse,(z,attn.qkv.weight))
            q,k,_=attn.qkv(z).reshape(2,32,3,4,128).unbind(2)
            q,k=attn.position(q),attn.position(k)
            score=(q.transpose(1,2).float()@k.transpose(1,2).float().transpose(-1,-2))/128**.5
            score=score.masked_fill(torch.ones(32,32,dtype=torch.bool).triu(1),-float('inf'))
            selected=score[batch,:,query]
            reference=F.cross_entropy(selected.reshape(-1,32),target[:,None].expand(-1,4).reshape(-1))
            grad_reference=torch.autograd.grad(reference,(z,attn.qkv.weight))
            errors=[(a-b).abs().max().item() for a,b in zip(grad_sparse,grad_reference)]
            assert torch.allclose(sparse,reference,atol=1e-6,rtol=1e-6)
            assert max(errors)<1e-6
            rows.append(dict(check='sparse_vs_full_causal_attention_CE_and_gradients',method=method,
                loss_abs_error=abs(sparse.item()-reference.item()),gradient_abs_errors=errors,passed=True))
        assert auxiliary_weight(6000)==auxiliary_weight(12000)==0
        assert auxiliary_weight(1)==auxiliary_weight(3000)==1
        assert all(auxiliary_weight(s+1)<=auxiliary_weight(s) for s in range(3000,6000))
        rows.append(dict(check='auxiliary_reaches_zero_for_last6001_updates',passed=True))
    write_json('results/raw/routing_checks.json',dict(rows=rows,passed=True))
    print('Routing target/loss/gradient/schedule checks passed',flush=True)


if __name__=='__main__': main()
