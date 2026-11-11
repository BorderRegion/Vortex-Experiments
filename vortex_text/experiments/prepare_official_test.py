"""Prepare the untouched official test once, using the frozen train-only tokenizer."""
import json
import os
import time
import numpy as np
import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer

from experiments.hadamard_holdout import ROOT, RAW, PLAN, digest, check_plan


def main():
    plan, provenance = check_plan()
    assert json.loads((RAW / 'validation/RUN_STATUS.json').read_text())['status'] == 'completed'
    assert json.loads((RAW / 'correctness.json').read_text())['passed']
    out = ROOT / 'data/official_test_manifest.json'
    target = ROOT / 'data/test.bin'
    assert not out.exists() and not target.exists(), 'Refuse to replace previously prepared official test'
    assert not (RAW / 'test').exists()
    assert digest(ROOT / 'data/tokenizer.json') == plan['tokenizer_sha256']
    spec = plan['official_test']
    name = spec['raw_filename']
    path = ROOT / 'data/raw' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/{spec['revision']}/wikitext-103-raw-v1/{name}?download=true"
    started = time.time()
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with path.with_suffix('.part').open('wb') as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    os.replace(path.with_suffix('.part'), path)
    assert digest(path) == spec['raw_sha256']
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    count, articles, rows = 0, 0, 0
    with target.open('wb') as handle:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            texts = [text for text in batch.column('text').to_pylist() if text]
            encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
            for text, encoding in zip(texts, encoded):
                stripped = text.strip()
                if stripped.startswith('= ') and not stripped.startswith('= =') and stripped.endswith(' ='):
                    handle.write(np.array([0], dtype='<u2').tobytes())
                    articles += 1
                    count += 1
                values = np.asarray(encoding.ids, dtype='<u2')
                handle.write(values.tobytes())
                count += len(values)
                rows += 1
        handle.write(np.array([0], dtype='<u2').tobytes())
        count += 1
    manifest = dict(dataset='Salesforce/wikitext', configuration='wikitext-103-raw-v1', split='test',
        revision=spec['revision'], raw_filename=name, raw_sha256=digest(path),
        tokenizer_sha256=digest(ROOT / 'data/tokenizer.json'), tokenizer_fit_split='train; unchanged',
        packing='Identical to original train/validation: row order/text retained; EOS before top-level article heading and at split end.',
        token_stream=dict(tokens=count, next_token_targets=count-1, dtype='<u2', sha256=digest(target)),
        articles=articles, nonempty_rows=rows, plan_sha256=digest(PLAN), preregistration=provenance,
        prepared_unix=time.time(), preparation_wall_seconds=time.time()-started)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
