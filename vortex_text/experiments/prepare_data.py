"""Pinned WikiText-103 raw, train-only byte BPE, immutable packed token streams."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import requests
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FILES = {
    'train-00000-of-00002.parquet': '74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212',
    'train-00001-of-00002.parquet': 'ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720',
    'validation-00000-of-00001.parquet': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(split):
    for name in FILES:
        if name.startswith(split):
            for batch in pq.ParquetFile(DATA / 'raw' / name).iter_batches(batch_size=2048):
                yield from batch.column('text').to_pylist()


def main():
    start = time.time()
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['RAYON_NUM_THREADS'] = '8'
    (DATA / 'raw').mkdir(parents=True, exist_ok=True)
    base = 'https://hf-mirror.com'
    response = requests.get(base + '/api/datasets/Salesforce/wikitext', timeout=45)
    response.raise_for_status()
    revision = response.json()['sha']
    for name, expected in FILES.items():
        path = DATA / 'raw' / name
        if not path.exists() or digest(path) != expected:
            url = f'{base}/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-103-raw-v1/{name}?download=true'
            print('download', name, flush=True)
            with requests.get(url, stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                with open(str(path) + '.part', 'wb') as f:
                    for chunk in r.iter_content(4 << 20):
                        f.write(chunk)
            os.replace(str(path) + '.part', path)
        assert digest(path) == expected, name
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=16384, min_frequency=2,
        special_tokens=['<eos>', '<unk>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False)
    print('fit tokenizer on train split only', flush=True)
    tokenizer.train_from_iterator((s for s in rows('train') if s), trainer=trainer)
    tokenizer.save(str(DATA / 'tokenizer.json'))
    info = dict(dataset='Salesforce/wikitext', configuration='wikitext-103-raw-v1',
        revision=revision, download_host=base, original_host_network_unreachable=True,
        raw_sha256=FILES, tokenizer_sha256=digest(DATA / 'tokenizer.json'),
        tokenizer_fit_split='train', vocab_size=tokenizer.get_vocab_size(),
        tokenizer_type='byte-level BPE',
        packing='Original row order; preserve row text including newlines; EOS before each top-level article heading and at split end. No cross-split packing.',
        packages={p: importlib.metadata.version(p) for p in ['tokenizers','pyarrow','numpy','requests']},
        splits={})
    for split in ['train', 'validation']:
        print('encode', split, flush=True)
        count = 0
        heading_count = 0
        buf = []
        path = DATA / f'{split}.bin'
        with path.open('wb') as f:
            def emit(texts):
                nonlocal count, heading_count
                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                arrays = []
                for text, enc in zip(texts, encoded):
                    stripped = text.strip()
                    if stripped.startswith('= ') and not stripped.startswith('= =') and stripped.endswith(' ='):
                        arrays.append(np.array([0], dtype='<u2'))
                        count += 1
                        heading_count += 1
                    arrays.append(np.asarray(enc.ids, dtype='<u2'))
                    count += len(enc.ids)
                if arrays:
                    f.write(np.concatenate(arrays).tobytes())
            for text in rows(split):
                if text:
                    buf.append(text)
                if len(buf) == 2048:
                    emit(buf)
                    buf = []
            emit(buf)
            f.write(np.array([0], dtype='<u2').tobytes())
            count += 1
        info['splits'][split] = dict(tokens=count, dtype='<u2', sha256=digest(path), articles=heading_count)
        print(split, info['splits'][split], flush=True)
    info['preparation_wall_seconds'] = time.time() - start
    (DATA / 'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
