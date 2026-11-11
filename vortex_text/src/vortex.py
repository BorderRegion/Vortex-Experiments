"""Route A signed permutation; layout [B,S,H,D], D=128."""
import torch

RINGS = (11, 13, 17, 19, 23, 29)
DIM = 128


def indices(positions, signed=True, device='cpu'):
    p = torch.as_tensor(positions, dtype=torch.int64, device='cpu').reshape(-1, 1)
    sources, signs, start = [], [], 0
    for length in RINGS:
        raw = torch.arange(length)[None, :] - p
        sources.append(raw.remainder(length) + start)
        signs.append(1 - 2 * torch.div(raw, length, rounding_mode='floor').remainder(2)
                     if signed else torch.ones_like(raw))
        start += length
    sources.append(torch.arange(start, DIM)[None, :].expand(len(p), -1))
    signs.append(torch.ones(len(p), DIM - start, dtype=torch.int64))
    return torch.cat(sources, -1).to(device), torch.cat(signs, -1).to(device)


def apply(x, source, sign):
    source = source[None, :, None, :].expand(x.shape[0], -1, x.shape[2], -1)
    return x.gather(-1, source) * sign[None, :, None, :].to(x.dtype)


def reference(x, position, signed=True):
    """Independent integer reference, deliberately only for correctness checks."""
    out, start = [], 0
    for length in RINGS:
        block = x[..., start:start + length].clone()
        if position >= 0:
            for _ in range(position % (2 * length if signed else length)):
                block = torch.cat((-block[..., -1:] if signed else block[..., -1:], block[..., :-1]), -1)
        else:
            for _ in range((-position) % (2 * length if signed else length)):
                block = torch.cat((block[..., 1:], -block[..., :1] if signed else block[..., :1]), -1)
        out.append(block)
        start += length
    return torch.cat(out + [x[..., start:]], -1)


def trace_correlation(delta):
    trace = 16
    for length in RINGS:
        if delta % length == 0:
            trace += length * (-1 if (delta // length) % 2 else 1)
    return trace / DIM
