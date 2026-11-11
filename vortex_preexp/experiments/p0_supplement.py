"""Exact CPU matrix checks supplement the batched/group and device P0 checks."""
import torch
from src import vortex
from src.common import write_json


def main():
    eye=torch.eye(128,dtype=torch.int64).reshape(128,1,1,128)
    identity=torch.eye(128,dtype=torch.int64)
    rows=[]
    positions=[-1000000,-58,-1,0,1,2,11,58,1333,1334,1335,14674,25346,1000000]
    for p in positions:
        matrix=vortex.apply(eye,*vortex.indices([p]))[:,0,0].T
        orthogonal=torch.equal(matrix.T@matrix,identity)
        tail_identity=torch.equal(matrix[-16:],identity[-16:])
        trace=matrix.diag().sum().item()/128
        assert orthogonal and tail_identity and trace==vortex.trace_correlation(p)
        rows.append(dict(position=p,exact_orthogonality=orthogonal,identity_tail=tail_identity,
            explicit_matrix_trace_correlation=trace,analytical_trace_correlation=vortex.trace_correlation(p)))
    write_json('results/raw/p0_math_supplement.json',dict(passed=True,rows=rows,device='cpu',dtype='int64'))
    print('Exact matrix orthogonality, identity tail, independent trace cross-check PASS')


if __name__=='__main__':main()
