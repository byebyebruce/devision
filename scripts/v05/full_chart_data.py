"""Prepare exp81's conditional scale dataset without changing the active exp80 data/code."""
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import random
import shutil
import sys

import pyarrow.parquet as pq

from chart_data import canonical, convert, digest, pixels, pool_key, rank, write_rows


def main():
    root, pilot, out = Path('data'), Path('data/v05/exp80v2'), Path('data/v05/exp81')
    if out.exists():
        raise SystemExit('Refusing to overwrite exp81')
    source = json.loads(Path('runs/v05-prep/chartqa_source.json').read_text())
    prior = json.loads((pilot / 'MANIFEST.json').read_text())
    source_dir = Path(source['snapshot']) / 'data'
    for p in source_dir.glob('*.parquet'):
        if digest(p.read_bytes()) != prior['source_parquets'][p.name]:
            raise ValueError('Changed source: ' + str(p))
    sys.path.insert(0, 'scripts/data/n2')
    from held import load
    held = load(str(root))
    if digest((root / 'n2/held.json').read_bytes()) != prior['exclusion_cache_sha256']:
        raise ValueError('Changed heldout definitions; replan required')
    tables, cache = {}, {}
    for split, pattern in [('train', 'train-*'), ('val', 'val-*')]:
        rows = []
        for p in sorted(source_dir.glob(pattern)):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    raw = r.pop('image')['bytes']
                    h = digest(raw)
                    if h not in cache:
                        sha, old_sha = pixels(raw)
                        image = 'v05/exp80/images/' + sha + '.png'
                        if not (root / image).is_file():
                            raise ValueError('Missing original decoded image: ' + image)
                        cache[h] = (sha, old_sha, image)
                    sha, old_sha, image = cache[h]
                    rows.append(dict(r, _pixel=sha, _old=old_sha, _image=image))
            print(split, p.name, len(rows), flush=True)
        tables[split] = rows
    val = {r['_pixel'] for r in tables['val']}
    eligible = [r for r in tables['train'] if r['_pixel'] not in val and r['_old'] not in held.pixels]
    calib = {r['_pixel'] for r in eligible if int(rank(r['_pixel'])[:8], 16) % 20 == 0}
    train_raw = [r for r in eligible if r['_pixel'] not in calib]
    pools, freq = defaultdict(dict), defaultdict(Counter)
    for r in train_raw:
        for answer in r['label']:
            k = pool_key(r['query'], answer)
            pools[k].setdefault(canonical(answer), answer.strip())
            freq[k][canonical(answer)] += 1
    chart = []
    for i, r in enumerate(train_raw):
        x = convert(r, pools, f'chartqa:train:{i}', r['_image'], 'chartqa:' + r['_pixel'], freq)
        if x:
            chart.append(x)
    chart.sort(key=lambda r: rank(r['id']))
    if len(chart) != prior['converted']['train'] or len(chart) < 24000:
        raise ValueError('Chart availability changed')
    chart = chart[:24000]
    pilot_rows = [json.loads(line) for line in (pilot / 'train.jsonl').open()]
    pc = {r['id']: r for r in pilot_rows if r['source'] == 'chartqa'}
    if {r['id']: r for r in chart[:8000]} != pc:
        raise ValueError('Chart conversion differs from pilot')
    with (root / 'n2/train_mix_fix.jsonl').open() as f:
        ordered = heapq.nsmallest(60000, (json.loads(line) for line in f), key=lambda r: rank(r['id']))
    replay, seen = [], set()
    excluded, pixel_cache = val | calib, {}
    for r in ordered:
        if r['id'] in seen:
            continue
        seen.add(r['id'])
        path = root / r['image']
        if not path.is_file():
            continue
        if r['image'] not in pixel_cache:
            pixel_cache[r['image']] = pixels(path.read_bytes())[0]
        if pixel_cache[r['image']] in excluded:
            continue
        replay.append(r)
        if len(replay) == 24000:
            break
    if len(replay) != 24000:
        raise ValueError('Insufficient replay')
    pr = {r['id']: r for r in pilot_rows if r['source'] != 'chartqa'}
    if {r['id']: r for r in replay[:8000]} != pr:
        raise ValueError('Replay differs from pilot')
    mixed = chart + replay
    random.Random(81).shuffle(mixed)
    out.mkdir()
    files = {'train': write_rows(out / 'train.jsonl', mixed)}
    for name in ['fit', 'calib', 'chart_dev', 'protect']:
        shutil.copyfile(pilot / (name + '.jsonl'), out / (name + '.jsonl'))
        files[name] = prior['files'][name]
    manifest = dict(prior, protocol='exp81-scale', parent_manifest_sha256=digest((pilot / 'MANIFEST.json').read_bytes()),
                    shuffle_seed=81, files=files, chart_questions=24000, replay_questions=24000,
                    pilot_prefix_exact_match=True, steps=6000)
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
