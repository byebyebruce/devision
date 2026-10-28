"""Build test_chartqa: ChartQA's official test split, converted with exp80/exp81's choice protocol.

Distractor pools and frequencies come from the same train rows exp81 used (train minus val pixels, minus held-out
pixels, minus the calib slice), so test questions get the same kind of options as chart_dev. Test pictures that are
pixel-equal to any train / val picture are dropped. Never overwrites an existing output.
"""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pyarrow.parquet as pq

from chart_data import canonical, convert, digest, pixels, pool_key, rank, write_rows


def load(paths, root, images, save):
    rows = []
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                raw = r.pop('image')['bytes']
                sha, old = pixels(raw)
                rel = images / (sha + '.png')
                if save and not (root / rel).is_file():
                    with Image.open(io.BytesIO(raw)) as im:
                        im.convert('RGB').save(root / rel)
                rows.append(dict(r, _pixel=sha, _old=old, _image=str(rel)))
        print(p.name, len(rows), flush=True)
    return rows


def main():
    root, out = Path('data'), Path('data/v05/chartqa_test')
    if (out / 'MANIFEST.json').exists():
        raise SystemExit('Refusing to overwrite ' + str(out))
    source = json.loads(Path('runs/v05-prep/chartqa_source.json').read_text())
    prior = json.loads(Path('data/v05/exp81/MANIFEST.json').read_text())
    data = Path(source['snapshot']) / 'data'
    for name, sha in prior['source_parquets'].items():
        if digest((data / name).read_bytes()) != sha:
            raise ValueError('Changed source: ' + name)
    sys.path.insert(0, 'scripts/data/n2')
    from held import load as load_held
    held = load_held(str(root))
    train = load(sorted(data.glob('train-*')), root, Path('v05/exp80/images'), False)
    val = load(sorted(data.glob('val-*')), root, Path('v05/exp80/images'), False)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    test_files = sorted(data.glob('test-*'))
    test = load(test_files, root, Path('v05/chartqa_test/images'), True)
    val_pics = {r['_pixel'] for r in val}
    eligible = [r for r in train if r['_pixel'] not in val_pics and r['_old'] not in held.pixels]
    if len(eligible) != prior['eligible']:
        raise ValueError('Eligible train rows differ from exp81')
    calib = {r['_pixel'] for r in eligible if int(rank(r['_pixel'])[:8], 16) % 20 == 0}
    pools, freq = defaultdict(dict), defaultdict(Counter)
    for r in eligible:
        if r['_pixel'] in calib:
            continue
        for answer in r['label']:
            k = pool_key(r['query'], answer)
            pools[k].setdefault(canonical(answer), answer.strip())
            freq[k][canonical(answer)] += 1
    seen = {r['_pixel'] for r in train} | val_pics
    kept = [r for r in test if r['_pixel'] not in seen]
    rows = [x for i, r in enumerate(kept)
            if (x := convert(r, pools, f'chartqa:test:{i}', r['_image'], 'chartqa:' + r['_pixel'], freq))]
    rows.sort(key=lambda r: rank(r['id']))
    positions = Counter()
    for r in rows:
        p = r['gold']['q']['probabilities']
        positions[list(p).index(max(p, key=p.get))] += 1
    manifest = {
        'protocol': 'chartqa-test-exp81-choices', 'source': source,
        'test_parquets': {f.name: digest(f.read_bytes()) for f in test_files},
        'raw_test': len(test), 'dropped_pixel_equal_to_train_or_val': len(test) - len(kept),
        'converted': len(rows), 'unconvertible': len(kept) - len(rows),
        'files': {'test_chartqa': write_rows(out / 'test_chartqa.jsonl', rows)},
        'by_kind': dict(Counter(r['kind'].split(':')[0] for r in rows)),
        'by_upstream': dict(Counter(r['upstream'] for r in rows)),
        'gold_positions': dict(positions),
        'limits': ['Locally constructed 4-way choices (yes/no as noul), not official relaxed-accuracy scoring',
                   'Pixel-equality exclusion only'],
    }
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
