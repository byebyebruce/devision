"""Verify actual sampled training inputs against reserved identities and available RGB hashes."""
import json,hashlib,sys
from pathlib import Path
pool=Path(sys.argv[1]);train_path=Path(sys.argv[2]);out=Path(sys.argv[3]);m=json.loads((pool/'MANIFEST.json').read_text());aliases={}
for p in ['data/v2/image_identity.json','data/v5/image_identity.json']:aliases.update(json.loads(Path(p).read_text()))
def canon(i):return aliases.get(i,i)
reserved_ids=set();reserved_paths=set();train_sources={'data/v4/scienceqa.jsonl','data/n2/science_mix.jsonl','data/v05/exp81/train.jsonl','data/v05/exp85/train.jsonl'}
for meta in m['inputs']:
    p=Path(meta['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    if str(p) in train_sources:continue
    for line in p.read_text().splitlines():
        s=json.loads(line)
        if s.get('image'):reserved_paths.add(s['image'])
        if s.get('image_id'):reserved_ids.add(canon(s['image_id']))
hashes=json.loads((pool/'pixel_hashes.json').read_text());rh={hashes[p] for p in reserved_paths if hashes.get(p)}
train=[json.loads(x) for x in train_path.read_text().splitlines()]
collisions=[s['id'] for s in train if canon(s.get('image_id')) in reserved_ids or s['image'] in reserved_paths or hashes[s['image']] in rh]
assert not collisions,collisions[:10]
r=dict(verified=True,actual_training_questions=len(train),actual_training_images=len({s['image'] for s in train}),reserved_paths=len(reserved_paths),canonical_reserved_ids=len(reserved_ids),identity_or_pixel_collisions=collisions,missing_reserved_pixels=sum(hashes.get(p) is None for p in reserved_paths),train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),limitations='exact local pixels and mapped IDs; missing reserved images and near-duplicates not proved absent')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
