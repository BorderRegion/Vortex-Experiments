"""Check that every auxiliary answer has an earlier explicit binding, without leakage."""
import torch
from src.synthetic import associative
from src.common import write_json
from .supervision import add_queries


def main():
    answers = 0
    for length in (64, 128, 256, 512):
        gen = torch.Generator().manual_seed(12345)
        aux = torch.Generator().manual_seed(87654)
        for _ in range(16):
            x, y = associative(16, length, gen)
            original = x.clone()
            z, dense = add_queries(x, y, aux)
            assert torch.equal(x, original)
            assert torch.equal(dense[:, -1], y[:, -1])
            assert torch.equal(z[:, -1], x[:, -1])
            assert ((dense != -100).sum(-1) == 8).all()
            for b, p in (dense != -100).nonzero().tolist():
                key, value = z[b, p], dense[b, p]
                assert 4 <= key < 36 and 36 <= value < 68
                matches = ((original[b, :p-1] == key) & (original[b, 1:p] == value)).nonzero()
                assert len(matches) > 0
                if p != length-1:
                    assert 96 <= original[b, p] < 112
                answers += 1
    # Repeated paired conditions must consume exactly the same independent streams.
    streams = []
    for _ in range(2):
        g = torch.Generator().manual_seed(12026); a = torch.Generator().manual_seed(22026)
        streams.append([add_queries(*associative(16, 256, g), a) for _ in range(4)])
    assert all(torch.equal(x, u) and torch.equal(y, v) for (x,y),(u,v) in zip(*streams))
    write_json('results/raw/supervision_oracle.json', dict(passed=True, answers_checked=answers,
               queries_per_sequence=8, original_end_query_preserved=True, paired_inputs_deterministic=True))
    print(f'PASS: {answers} answers have earlier explicit bindings; no answer token inserted.', flush=True)


if __name__ == '__main__':
    main()
