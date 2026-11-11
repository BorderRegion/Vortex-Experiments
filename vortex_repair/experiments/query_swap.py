"""Swap to another valid, explicitly bound key while holding the context fixed."""
from pathlib import Path
import argparse
import torch
from torch.nn import functional as F
from src.model import Decoder, ModelConfig
from src.synthetic import associative
from src.common import ROOT, setup, autocast, Meter, write_json


@torch.no_grad()
def probe(model, device):
    gen = torch.Generator().manual_seed(381907)
    count = 0; accuracy = changed_accuracy = unchanged = tv = 0.
    for _ in range(32):
        x, y = associative(16, 256, gen, distance=32, evaluation=True)
        z = x.clone(); other = []
        for b in range(16):
            # Require a different answer, so unchanged output cannot answer both queries.
            candidates = ((x[b, :-2] >= 4) & (x[b, :-2] < 36) &
                          (x[b, 1:-1] >= 36) & (x[b, 1:-1] < 68) &
                          (x[b, 1:-1] != y[b, -1])).nonzero().flatten()
            assert len(candidates)
            p = candidates[int(torch.randint(len(candidates), (1,), generator=gen))]
            z[b, -1] = x[b, p]; other.append(x[b, p+1])
        other = torch.stack(other).to(device)
        with autocast():
            a = model(x.to(device))[0][:, -1].float()
            b = model(z.to(device))[0][:, -1].float()
        accuracy += (a.argmax(-1) == y[:, -1].to(device)).sum().item()
        changed_accuracy += (b.argmax(-1) == other).sum().item()
        unchanged += (a.argmax(-1) == b.argmax(-1)).sum().item()
        tv += (a.softmax(-1)-b.softmax(-1)).abs().sum().item()/2
        count += 16
    return dict(samples=count, length=256, distance=32, original_accuracy=accuracy/count,
                changed_query_accuracy=changed_accuracy/count, unchanged_argmax=unchanged/count,
                query_probability_change=tv/count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binding-only', action='store_true')
    ap.add_argument('--output', default='query_swap')
    ap.add_argument('--pattern', default='*_final.pt')
    args = ap.parse_args()
    device = setup(); rows = []
    sources = [(Path('/root/autodl-tmp/vortex_preexp'), n) for n in (
        'p2_associative_rope_safe_s2026', 'p2b_associative_rope_safe_s2026', 'p2b_associative_vortex_s2026')]
    sources += [(ROOT, p.stem.removesuffix('_final')) for p in sorted((ROOT/'results/checkpoints').glob(args.pattern))]
    if args.binding_only:
        sources = [(base, name) for base, name in sources if name.startswith('binding_')]
    with Meter('followup_valid_query_swap'):
        for base, name in sources:
            ckpt = torch.load(base/f'results/checkpoints/{name}_final.pt', map_location='cpu', weights_only=False)
            model = Decoder(ModelConfig(**ckpt['model_config'])).to(device)
            model.load_state_dict(ckpt['state']); model.eval()
            rows.append(dict(checkpoint=name, **probe(model, device)))
    write_json(f'results/raw/{args.output}.json', dict(rows=rows,
        limitation='Counterfactual keys exist with unambiguous bindings and different values; some have repeated bindings. Original evaluation remains the main accuracy result.'))
    print('Valid query swap complete', flush=True)


if __name__ == '__main__':
    main()
