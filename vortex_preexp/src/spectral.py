"""Real orthogonal spectral basis: G=URU^T. Angles prepared in CPU FP64."""
import math
import torch
from .vortex import RINGS, DIM


def basis():
    u = torch.zeros(DIM, DIM, dtype=torch.float64)
    start = 0
    for length in RINGS:
        j = torch.arange(length, dtype=torch.float64)
        for k in range((length - 1) // 2):
            theta = (2 * k + 1) * math.pi / length
            u[start:start+length, start+2*k] = (2/length)**0.5 * torch.cos(theta*j)
            u[start:start+length, start+2*k+1] = (2/length)**0.5 * torch.sin(theta*j)
        u[start:start+length, start+length-1] = (-1.0)**j / math.sqrt(length)
        start += length
    u[start:, start:] = torch.eye(DIM-start, dtype=torch.float64)
    return u


def rotation_tables(positions, method='rope_safe', device='cpu', dtype=torch.float32):
    p = torch.as_tensor(positions, dtype=torch.int64, device='cpu').reshape(-1, 1)
    if method == 'rope_safe':
        freq = 10000.0 ** (-torch.arange(0, DIM, 2, dtype=torch.float64)/DIM)
        angle = torch.remainder(p.double() * freq, 2 * math.pi)
        c = torch.repeat_interleave(angle.cos(), 2, -1)
        s = torch.repeat_interleave(angle.sin(), 2, -1)
        swap = torch.arange(DIM).reshape(-1, 2).flip(-1).reshape(-1)
        orientation = torch.tensor([-1., 1.]).repeat(DIM//2)
    else:
        c = torch.ones(len(p), DIM, dtype=torch.float64)
        s = torch.zeros_like(c)
        swap = torch.arange(DIM)
        orientation = torch.zeros(DIM)
        start = 0
        for length in RINGS:
            for k in range((length-1)//2):
                angle = torch.remainder(p[:, 0], 2*length).double() * ((2*k+1)*math.pi/length)
                c[:, start+2*k:start+2*k+2] = angle.cos()[:, None]
                s[:, start+2*k:start+2*k+2] = angle.sin()[:, None]
                swap[start+2*k:start+2*k+2] = torch.tensor([start+2*k+1, start+2*k])
                orientation[start+2*k:start+2*k+2] = torch.tensor([-1., 1.])
            c[:, start+length-1] = 1-2*p[:, 0].remainder(2)
            start += length
    return (c.to(device=device, dtype=dtype), s.to(device=device, dtype=dtype),
            swap.to(device), orientation.to(device=device, dtype=dtype))


def rotate(x, tables):
    c, s, swap, orientation = tables
    return (x.float()*c[None, :, None, :] +
            x.float().index_select(-1, swap)*s[None, :, None, :]*orientation).to(x.dtype)


def convert_checkpoint(state, heads=4):
    """Linear stores [output,input]; replace each Q/K head W by U^T W."""
    u = basis().float()
    result = {k:v.clone() for k,v in state.items()}
    for name, weight in state.items():
        if name.endswith('attn.qkv.weight'):
            parts = weight.float().reshape(3, heads, DIM, -1).clone()
            parts[:2] = torch.einsum('ij,ahjk->ahik', u.T, parts[:2])
            result[name] = parts.reshape_as(weight).to(weight.dtype)
    return result
