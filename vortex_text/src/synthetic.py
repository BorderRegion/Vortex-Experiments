"""Deterministic CPU generators. Symbols carry no answer statistics."""
import torch

OFFSETS=(1,2,8,32,128)
VERSION='relative-v1_assoc-latest-free-v1'


def relative(batch,length,generator,distance=None,evaluation=False):
    x=torch.randint(4,68,(batch,length),generator=generator)
    labels=torch.full_like(x,-100)
    if evaluation:
        ds=torch.full((batch,),distance,dtype=torch.long)
        labels[:,-1]=x[torch.arange(batch),length-1-ds]
        x[:,-1]=80+OFFSETS.index(distance)
    else:
        choices=torch.randint(0,len(OFFSETS),(batch,length),generator=generator)
        ds=torch.tensor(OFFSETS)[choices]
        source=torch.arange(length)[None,:]-ds
        mask=(torch.rand((batch,length),generator=generator)<.125)&(source>=0)
        source_query=mask.gather(1,source.clamp_min(0))
        mask &= ~source_query
        labels[mask]=x.gather(1,source.clamp_min(0))[mask]
        x[mask]=80+choices[mask]
    return x,labels


def associative(batch,length,generator,distance=None,evaluation=False):
    # Keys 4..35; values 36..67; random filler 96..111. Sixteen unique bindings,
    # plus repeats of non-target key/value patterns. Target occurs exactly once.
    x=torch.randint(96,112,(batch,length),generator=generator)
    labels=torch.full_like(x,-100)
    for b in range(batch):
        keys=torch.randperm(32,generator=generator)[:16]+4
        values=torch.randint(36,68,(16,),generator=generator)
        d=distance if distance is not None else int(torch.randint(2,length-1,(1,),generator=generator))
        target=length-1-d
        x[b,target:target+2]=torch.stack((keys[0],values[0]))
        allowed=torch.ones(length-2,dtype=torch.bool)
        allowed[max(0,target-1):min(length-2,target+2)]=False
        # Choose non-overlapping arbitrary positions, avoiding a token-grid parity confound.
        available=torch.randperm(length-2,generator=generator)
        used=1
        for p in available.tolist():
            if not allowed[p]:continue
            idx=1+(used-1)%15
            x[b,p:p+2]=torch.stack((keys[idx],values[idx]))
            allowed[max(0,p-1):min(length-2,p+2)]=False
            used+=1
            if used>=24:break
        x[b,-1]=keys[0]
        labels[b,-1]=values[0]
    return x,labels


def make_batch(task,batch,length,generator,**kwargs):
    return (relative if task=='relative' else associative)(batch,length,generator,**kwargs)
