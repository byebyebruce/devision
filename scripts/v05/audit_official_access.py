"""Archive the official public access evidence; never infer missing data from names alone."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path('runs/exp107-official-access')

def fetch(name, url):
    with urlopen(Request(url, headers={'User-Agent': 'deVision-access-audit'}), timeout=30) as response:
        raw = response.read()
        meta = {'url': url, 'status': response.status, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    (OUT / (name + '.json')).write_bytes(raw)
    manifest[name] = meta
    (OUT / 'fetches.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return json.loads(raw)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    manifest = {'observed_at': datetime.now(timezone.utc).isoformat()}
    try:
        results = fetch('official-results', 'https://benchmarkheaven.com/api/decision-results/imagejevbench/v0.3.0/json')
        root = 'https://api.github.com/repos/fstandhartinger/model-market-comparison'
        commit = fetch('commit', root + '/commits/main')['sha']
        tree = fetch('tree', root + '/git/trees/' + commit + '?recursive=1')
        assert tree['truncated'] is False
        examples = fetch('examples', 'https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/' + commit + '/public/image-jev/examples/manifest.json')
        assert results['version'] == 'v0.3.0'
        paths = [x['path'] for x in tree['tree']]
        relevant = [p for p in paths if any(t in p.lower() for t in ('imagejev', 'image-jev', 'public/core', 'score'))]
        report = {'commit': commit, 'tree_truncated': False, 'paths_total': len(paths), 'relevant_paths': relevant,
                  'source_sha256': results['source_sha256'], 'built_at': results['built_at'],
                  'counts': results['method']['counts'], 'capability_method': results['method']['capability'],
                  'examples_count': examples['count'], 'examples_status': examples.get('status'),
                  'public_core_paths': [p for p in paths if 'public/core/' in p],
                  'interpretation': 'Public-channel evidence only; no full public300 downloaded or official evaluation executed.'}
        (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    except Exception as exc:
        (OUT / 'ERROR.txt').write_text(type(exc).__name__ + ': ' + str(exc) + '\n')
        (OUT / 'EXIT_CODE').write_text('1\n')
        raise
    (OUT / 'EXIT_CODE').write_text('0\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'relevant_paths'}, indent=2))
