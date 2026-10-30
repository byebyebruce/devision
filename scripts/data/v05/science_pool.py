"""Create exp110 real-science/replay pool; exclude every local evaluation identity and RGB hash."""
import hashlib,json,re,struct
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
OUT=Path('data/v05/exp110-retry2'); OUT.mkdir(exist_ok=False)
def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in Path('data').rglob('*.jsonl') if re.search(r'dev|test|bench|calib|monitor|public|protect|independent|val[_.]|external',str(p)) and 'details' not in p.name]
# Include actual past screening inputs, including unnamed real-world panels.
files+=list(Path('runs').glob('exp*/inputs.jsonl'))
reserved={}; ids=set(); missing=[]; missing_sources=set(); signatures=[]
for p in sorted(set(files)):
    signatures.append(dict(path=str(p),sha256=sha(p)))
    for s in rows(p):
        if s.get('image'):
            reserved[s['image']]=s.get('source','unknown')
            if s.get('image_id'): ids.add(s['image_id'])
for mapping in ['data/v2/image_identity.json','data/v5/image_identity.json']:
    aliases=json.loads(Path(mapping).read_text()); ids.update(aliases.get(i,i) for i in list(ids))
print('reserved',len(reserved),'files',len(files),flush=True)
pool={}
for p in map(Path,['data/v4/scienceqa.jsonl','data/n2/science_mix.jsonl','data/v05/exp81/train.jsonl','data/v05/exp85/train.jsonl']):
    signatures.append(dict(path=str(p),sha256=sha(p)))
    for s in rows(p):
        if s['source'] in {'scienceqa','ai2d','tqa'} or str(p).startswith('data/v05/'):
            pool.setdefault(s['id'],s)
# Hash decoded pixels with dimensions; fail closed if evaluation image unavailable.
def pixel(name):
    p=Path('data')/name
    if not p.is_file() and re.fullmatch(r'\d{12}\.jpg',Path(name).name):
        p=Path('data/coco/val2014')/('COCO_val2014_'+Path(name).name)
    if not p.is_file(): return name,None
    with Image.open(p) as im:
        im=im.convert('RGB'); h=hashlib.sha256(struct.pack('>II',*im.size)+im.tobytes()).hexdigest()
    return name,h
paths=sorted(set(reserved)|{s['image'] for s in pool.values()})
hashes={}
with ThreadPoolExecutor(max_workers=8) as ex:
    for i,(p,h) in enumerate(ex.map(pixel,paths)):
        hashes[p]=h
        if h is None:
            missing.append(p)
            if p in reserved: missing_sources.add(reserved[p])
        if i%2000==0: print('pixels',i,'/',len(paths),flush=True)
forbidden={hashes[p] for p in reserved if hashes[p] is not None}; kept=[]; excluded=Counter()
for s in pool.values():
    if hashes[s['image']] is None or s['source'] in missing_sources: excluded['missing_or_source_quarantine']+=1
    elif s['image'] in reserved or s.get('image_id') in ids: excluded['identity']+=1
    elif hashes[s['image']] in forbidden: excluded['pixels']+=1
    else: kept.append(s)
science=[s for s in kept if s['source'] in {'scienceqa','ai2d','tqa'}]
replay=[s for s in kept if s['source'] not in {'scienceqa','ai2d','tqa'}]
assert all(sum(s['source']==source for s in science)>100 for source in ['scienceqa','ai2d','tqa'])
for name,rs in [('science',science),('replay',replay)]:
    (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rs))
(OUT/'pixel_hashes.json').write_text(json.dumps(hashes))
manifest=dict(inputs=signatures,reserved_images=len(reserved),reserved_ids=len(ids),pixel_images=len(hashes),excluded=dict(excluded),missing_images=missing,quarantined_sources=sorted(missing_sources),science=dict(Counter(s['source'] for s in science)),replay=dict(Counter(s['source'] for s in replay)),outputs={n:sha(OUT/f'{n}.jsonl') for n in ['science','replay']},limitations='Exact decoded RGB exclusion; no near-duplicate proof; unknown external backbone pretraining. Evaluation identity IDs plus pixel matching covers cross-source exact images; no training/heldout claims about earlier base weights.')
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
