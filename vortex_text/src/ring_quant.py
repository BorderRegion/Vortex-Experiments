"""Variable-length scales aligned to the unchanged signed-permutation cycles.

This is simulated INT4/FP4 with seven FP32 scales/head, not a native group16 format.
"""
import torch
from .quant import quantize as uniform_quantize
from .vortex import RINGS, DIM

GROUPS = (*RINGS, DIM-sum(RINGS))


def quantize(x, group=GROUPS, kind='int4', return_codes=False):
    if isinstance(group, int):
        return uniform_quantize(x, group, kind, return_codes)
    assert sum(group) == x.shape[-1]
    parts = [uniform_quantize(block, n, kind, True)
             for block, n in zip(x.split(group, dim=-1), group)]
    values = torch.cat([p[0] for p in parts], -1)
    if not return_codes:
        return values
    return values, torch.cat([p[1] for p in parts], -1), torch.cat([p[2] for p in parts], -2)
