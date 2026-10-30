"""Verify actual exp81 training images against its fixed held-out panels before launch."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from audit_exp80 import read
from chart_data import digest, pixels


def main():
    root, data = Path('data'), Path('data/v05/exp81')
    manifest = json.loads((data / 'MANIFEST.json').read_text())
    rows = {n: read(data / (n + '.jsonl')) for n in ['train', 'calib', 'chart_dev', 'protect']}
    rows.update(examples=read(root / 'v05/external/jev_examples.jsonl'),
                realworld=read(root / 'v05/external/realworld_native.jsonl'))
    for name, meta in manifest['files'].items():
        assert digest((data / (name + '.jsonl')).read_bytes()) == meta['sha256'], name
    files = sorted({r['image'] for rs in rows.values() for r in rs})
    def hashed(path):
        return path, pixels((root / path).read_bytes())[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(hashed, files))
    ids = {n: {hashes[r['image']] for r in rs} for n, rs in rows.items()}
    overlaps = {n: len(ids['train'] & s) for n, s in ids.items() if n != 'train'}
    prior = defaultdict(Counter)
    for r in rows['train']:
        q, gold = r['questions']['q'], r['gold']['q']['probabilities']
        assert abs(sum(gold.values()) - 1) < 1e-7 and all(0 <= x <= 1 for x in gold.values())
        assert set(gold) == (set(q['criteria']) if q['type'] == 'choice' else {'true', 'false'})
        if r['source'] == 'chartqa':
            prior[r['kind']][max(gold, key=gold.get)] += 1
    hits = 0
    for r in rows['chart_dev']:
        gold = r['gold']['q']['probabilities']
        prediction = max(sorted(gold), key=lambda o: prior[r['kind']][o])
        hits += gold[prediction]
    result = dict(manifest_sha256=digest((data / 'MANIFEST.json').read_bytes()), images_hashed=len(files),
                  train_questions=len(rows['train']), train_overlap_by_decoded_pixels=overlaps,
                  train_image_hashes_sha256=digest('\n'.join(sorted(ids['train'])).encode()),
                  label_prior_accuracy=hits/len(rows['chart_dev']),
                  limits=manifest['limits'])
    result['pass'] = len(rows['train']) == 48000 and not any(overlaps.values())
    Path('runs/v05-prep/exp81_data_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['pass']:
        raise SystemExit('exp81 preflight failed')


if __name__ == '__main__':
    main()
