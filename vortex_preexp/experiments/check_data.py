"""Independent label and paired-control oracle; CPU only."""
import torch
from src.synthetic import make_batch,OFFSETS
from src.common import write_json


def main():
    rows=[]
    for task in ['relative','associative']:
        for length in [256,512,2048]:
            paired=[]
            for distance in ([1,2,8,32,128] if task=='relative' else [8,128,1333,1334,1335]):
                if distance>=length-1:continue
                x,y=make_batch(task,64,length,torch.Generator().manual_seed(900000+length*17),distance=distance,evaluation=True)
                if task=='relative':assert torch.equal(y[:,-1],x[:,-1-distance])
                else:
                    for a,b in zip(x,y):
                        assert (a[:-1]==a[-1]).sum()==1
                        assert a[-1-distance]==a[-1] and a[-distance]==b[-1]
                    paired.append((distance,x[:,-1].clone(),y[:,-1].clone()))
                rows.append(dict(task=task,length=length,distance=distance,samples=64,oracle_pass=True))
            if task=='associative':
                for _,q,answer in paired:
                    assert torch.equal(q,paired[0][1]) and torch.equal(answer,paired[0][2])
    x,y=make_batch('relative',32,512,torch.Generator().manual_seed(55))
    for b,p in (y!=-100).nonzero().tolist():
        d=OFFSETS[x[b,p]-80];assert y[b,p]==x[b,p-d]
    write_json('results/raw/data_oracle.json',dict(passed=True,evaluation=rows,
        training_labels_checked=int((y!=-100).sum()),paired_associative_queries_and_answers_identical=True))
    print('Label/distance oracle and paired-control identity PASS')


if __name__=='__main__':main()
