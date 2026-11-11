"""Software numerical models; no native 4-bit kernels."""
import torch


def quantize(x, group=128, kind='int4', return_codes=False):
    shape = x.shape
    z = x.float().reshape(*shape[:-1], shape[-1]//group, group)
    if kind == 'int4':
        scale = z.abs().amax(-1, keepdim=True).clamp_min(1e-12)/7
        codes = torch.round(z/scale).clamp(-7, 7)
        values = codes * scale
    elif kind == 'fp4':
        scale = z.abs().amax(-1, keepdim=True).clamp_min(1e-12)/6
        a = z.abs()/scale
        code = torch.zeros_like(a)
        levels = (0., .5, 1., 1.5, 2., 3., 4., 6.)
        for i, threshold in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
            code += (a > threshold).float()
        values = torch.zeros_like(a)
        for i, level in enumerate(levels):
            values += (code == i).float() * level
        codes = code*z.sign()
        values = values*z.sign()*scale
    else:
        raise ValueError(kind)
    values = values.reshape(shape).to(x.dtype)
    return (values, codes.reshape(shape).to(torch.int8), scale) if return_codes else values
