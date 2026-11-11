"""Read-only probes of the completed associative controls; results stay separate."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import setup, autocast, Meter, write_json

ORIGINAL = Path('/root/autodl-tmp/vortex_preexp')


@torch.no_grad()
def probe(model, device, samples=256):
    gen = torch.Generator().manual_seed(381907)
    totals = {k: 0. for k in ('correct', 'nll', 'no_query_correct', 'no_query_nll',
                              'query_probability_change', 'bag_correct')}
    attention = []
    for start in range(0, samples, 16):
        x, y = associative(16, 256, gen, distance=32, evaluation=True)
        target = y[:, -1].to(device)
        captured = []
        hooks = []
        for block in model.blocks:
            def capture(module, args):
                z = args[0]
                q, k, _ = module.qkv(z).reshape(16, 256, 3, 4, 128).unbind(2)
                q, k = module.position(q), module.position(k)
                with autocast(enabled=False):
                    p = ((q[:, -1].float().unsqueeze(2)
                          @ k.float().transpose(1, 2).transpose(-1, -2)) / 128**.5).softmax(-1)
                captured.append(p.squeeze(2).detach())
            hooks.append(block.attn.register_forward_pre_hook(capture))
        with autocast():
            logits, _ = model(x.to(device))
        for h in hooks:
            h.remove()
        logits = logits[:, -1].float()
        totals['correct'] += (logits.argmax(-1) == target).sum().item()
        totals['nll'] += F.cross_entropy(logits, target, reduction='sum').item()
        changed = x.clone(); changed[:, -1] = 0
        with autocast():
            changed_logits = model(changed.to(device))[0][:, -1].float()
        totals['no_query_correct'] += (changed_logits.argmax(-1) == target).sum().item()
        totals['no_query_nll'] += F.cross_entropy(changed_logits, target, reduction='sum').item()
        totals['query_probability_change'] += (logits.softmax(-1)-changed_logits.softmax(-1)).abs().sum().item()/2
        counts = F.one_hot(x.clamp(36, 67)-36, 32).float()
        counts *= ((x >= 36) & (x < 68)).unsqueeze(-1)
        totals['bag_correct'] += (counts.sum(1).argmax(-1)+36 == y[:, -1]).sum().item()
        for layer, p in enumerate(captured):
            attention.append(dict(layer=layer, target_key_mass=p[:, :, 223].mean().item(),
                                  target_value_mass=p[:, :, 224].mean().item(),
                                  max_mass=p.max(-1).values.mean().item()))
    return dict(samples=samples, length=256, distance=32,
                **{k: v/samples for k, v in totals.items()},
                attention=[dict(layer=i, **{k: sum(r[k] for r in attention if r['layer']==i)/(samples//16)
                    for k in ('target_key_mass', 'target_value_mass', 'max_mass')}) for i in range(len(model.blocks))])


def main():
    device = setup()
    rows = []
    with Meter('followup_read_only_diagnosis'):
        for name in ('p2_associative_rope_safe_s2026', 'p2b_associative_rope_safe_s2026',
                     'p2b_associative_vortex_s2026', 'p2c_associative_rope_safe_s2026'):
            ckpt = torch.load(ORIGINAL/'results/checkpoints'/f'{name}_final.pt', map_location='cpu', weights_only=False)
            model = Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']); model.eval()
            row = dict(checkpoint=name, **probe(model, device))
            rows.append(row); print(json.dumps(row), flush=True)
    write_json('results/raw/diagnosis.json', dict(rows=rows, note='No weights updated; query erased only as an ablation.'))


if __name__ == '__main__':
    main()
