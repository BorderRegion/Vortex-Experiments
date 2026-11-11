"""Test the one previously proposed scale-layout revision; preserve the operator."""
from pathlib import Path
import torch
from src import spectral, vortex
from src.model import Decoder, ModelConfig
from src.ring_quant import quantize, GROUPS
from src.synthetic import make_batch
from src.common import setup, autocast, Meter, write_json, metadata

ORIGINAL = Path('/root/autodl-tmp/vortex_preexp')


def position(x, p, method):
    if method == 'vortex':
        return vortex.apply(x, *vortex.indices(p, device=x.device))
    return spectral.rotate(x, spectral.rotation_tables(p, method, x.device))


def measures(score, ref, v):
    delta = score-ref
    a = torch.einsum('bsh,bshd->bhd', score.softmax(1), v)
    b = torch.einsum('bsh,bshd->bhd', ref.softmax(1), v)
    return dict(logit_nrmse=(delta.square().mean()/ref.square().mean().clamp_min(1e-16)).sqrt().item(),
                output_nrmse=((a-b).square().mean()/b.square().mean().clamp_min(1e-16)).sqrt().item(),
                softmax_kl=(ref.softmax(1)*(ref.log_softmax(1)-score.log_softmax(1))).sum(1).mean().item(),
                top1_change=(score.argmax(1) != ref.argmax(1)).float().mean().item())


@torch.no_grad()
def main():
    device = setup(); commutation = []; rows = []; equivalences = []
    with Meter('followup_ring_aligned_scale_revision'):
        gen = torch.Generator().manual_seed(771234)
        x = torch.randn(8, 9, 4, 128, generator=gen).to(device)
        x[..., ::19] *= 9
        pos = torch.tensor([-1000000, -128, -1, 0, 1, 13, 128, 1334, 1000000])
        for group in (128, 16, GROUPS):
            for kind in ('int4', 'fp4'):
                before, codes, _ = quantize(x, group, kind, True)
                post, post_codes, _ = quantize(position(x, pos, 'vortex'), group, kind, True)
                pre = position(before, pos, 'vortex')
                code_pre = position(codes, pos, 'vortex')
                row = dict(group=group, kind=kind, scales_per_head=(128//group if isinstance(group, int) else len(group)),
                           code_match=(code_pre == post_codes).float().mean().item(), max_dequant_difference=(pre-post).abs().max().item())
                if group in (128, GROUPS):
                    assert row['code_match'] == 1. and row['max_dequant_difference'] == 0., row
                commutation.append(row)
        u = spectral.basis().float().to(device)
        for task, name in [('relative', 'p2_relative_vortex_s2026'), ('associative', 'p2b_associative_vortex_s2026')]:
            ckpt = torch.load(ORIGINAL/f'results/checkpoints/{name}_final.pt', map_location='cpu', weights_only=False)
            model = Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']); model.eval()
            for length in (256, 512, 2048):
                x, _ = make_batch(task, 64, length, torch.Generator().manual_seed(771000+length), distance=32, evaluation=True)
                with autocast():
                    block = model.blocks[0]
                    z = block.ln1(model.embedding(x.to(device)))
                    q, k, v = block.attn.qkv(z).reshape(64, length, 3, 4, 128).unbind(2)
                q, k, v = q[:, -1:].float(), k.float(), v.float()
                refs = {}
                for method in ('rope_safe', 'vortex', 'spectral_rope'):
                    q0, k0 = (q@u, k@u) if method == 'spectral_rope' else (q, k)
                    qa, ka = position(q0, [length-1], method), position(k0, torch.arange(length), method)
                    ref = (qa*ka).sum(-1)/128**.5; refs[method] = ref
                    for group in (128, 16, GROUPS):
                        qz, kz = quantize(qa, group), quantize(ka, group)
                        score = (qz*kz).sum(-1)/128**.5
                        rows.append(dict(task=task, source=name, length=length, samples=64, layer=0,
                                         method=method, group=group, kind='int4', scope='qk',
                                         scales_per_head=(128//group if isinstance(group,int) else len(group)),
                                         **measures(score, ref, v)))
                rel = ((refs['vortex']-refs['spectral_rope']).norm()/refs['vortex'].norm()).item()
                assert rel < 2e-5
                equivalences.append(dict(source=name, length=length, relative_l2=rel))
        write_json('results/raw/ring_groups.json', dict(**metadata(dict(seed=2026)), groups=GROUPS,
                   commutation=commutation, rows=rows, fp32_equivalence=equivalences,
                   limitations='Seven FP32 scales per head with variable-length groups. Same group lengths and scale budget for RoPE/spectral comparisons. Existing synthetic checkpoints, first layer only; no native kernel or text/whole-model quality claim.'))
    print('Ring-aligned scale revision complete', flush=True)


if __name__ == '__main__':
    main()
