import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def setup(seed=2026, device='musa'):
    if device == 'musa':
        import torch_musa
        assert torch.musa.is_available()
    torch.set_num_threads(8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'musa':
        torch.musa.manual_seed_all(seed)
    return torch.device(device)


def sync(device='musa'):
    if str(device).startswith('musa'):
        torch.musa.synchronize()


def autocast(device='musa', enabled=True):
    return torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def write_json(path, obj):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def append_jsonl(path, obj):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')


def code_hash():
    h = hashlib.sha256()
    for p in sorted(list((ROOT/'src').glob('*.py'))+list((ROOT/'experiments').glob('*.py'))):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


class Meter:
    """Conservative occupied wall time, including Python overhead in an active stage."""
    def __init__(self, stage):
        self.stage = stage
    def __enter__(self):
        sync(); self.start = time.time()
        self.prior = sum(json.loads(x)['seconds'] for x in (ROOT/'results/raw/gpu_ledger.jsonl').read_text().splitlines()) if (ROOT/'results/raw/gpu_ledger.jsonl').exists() else 0
        if self.prior >= 36*3600:
            raise RuntimeError('36 GPU-hour cap reached')
        return self
    def remaining(self):
        return 36*3600-self.prior-(time.time()-self.start)
    def __exit__(self, *args):
        sync()
        append_jsonl('results/raw/gpu_ledger.jsonl', {'stage': self.stage, 'start_unix': self.start,
            'end_unix': time.time(), 'seconds': time.time()-self.start, 'success': args[0] is None})


def metadata(config):
    return dict(config=config, code_sha256=code_hash(), torch=torch.__version__,
                device='musa', seed=config.get('seed',2026), timestamp=time.time())
