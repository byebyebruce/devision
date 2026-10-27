"""Archive selected public contract sources at the exp107 immutable revision."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path('runs/exp108-official-contract')
PATHS = ['components/ImageJevV03Page.tsx', 'app/jev-models/api/page.tsx',
         'ops/priority-evaluation/pod_drivers/native_image.py',
         'ops/priority-evaluation/official_scoring.py',
         'lib/image-jev-public-examples.mjs']

def main():
    OUT.mkdir(exist_ok=False)
    manifest = {'commit': json.loads(Path('runs/exp107-official-access/commit.json').read_text())['sha'], 'files': []}
    try:
        for path in PATHS:
            url = f"https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/{manifest['commit']}/{path}"
            with urlopen(Request(url, headers={'User-Agent': 'deVision-contract-audit'}), timeout=30) as response:
                raw = response.read()
                status = response.status
            name = path.replace('/', '__')
            (OUT / name).write_bytes(raw)
            manifest['files'].append(dict(path=path, local=name, url=url, status=status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (OUT / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (OUT / 'ERROR.txt').write_text(type(exc).__name__ + ': ' + str(exc))
        (OUT / 'EXIT_CODE').write_text('1\n')
        raise
    print('Archived', len(manifest['files']), 'public source files at', manifest['commit'])

if __name__ == '__main__':
    main()
