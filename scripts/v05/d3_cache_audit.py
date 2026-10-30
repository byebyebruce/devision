"""Prepare audited frozen D3 features; does not train or select a model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('D3_FAST', '0')
import torch
from safetensors.torch import save_file
from PIL import Image
from d3_image_processor import load_processor
from chart_data import pixels

WORK = Path('runs/exp93-d3-cache-audit-64')
REV = '03560c26f5199c3c04129a2ccb38960c913635e0'

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/'STARTED').open('x').close()
    start = time.monotonic()
    sources = ['data/v05/exp81/train.jsonl', 'data/v05/exp85/ui_train.jsonl']
    held_paths = ['data/v05/exp81/'+n+'.jsonl' for n in ['protect','chart_dev','calib']]
    held_paths += ['data/v05/exp85/'+n+'.jsonl' for n in ['ui_dev','calib']]
    held_paths += ['data/v05/exp91/independent.jsonl','data/v05/external/realworld_native.jsonl','data/v05/external/jev_examples.jsonl']
    aliases = json.loads(Path('data/v5/image_identity.json').read_text())
    def identity(s): return aliases.get(s['image_id'], s['image_id'])
    held = [s for p in held_paths for s in read(p)]
    forbidden = {identity(s) for s in held}
    forbidden_paths = {s['image'] for s in held}
    candidates = {k: [] for k in ['coco-size','coco-position','chartqa','screenqa_cf']}
    seen = set()
    for p in sources:
        for s in read(p):
            k = s['source']
            if k not in candidates or len(candidates[k]) >= 256 or s.get('flipped'):
                continue
            if s['questions']['q']['type'] != 'choice': continue
            if identity(s) in forbidden or s['image'] in forbidden_paths or s['image'] in seen: continue
            seen.add(s['image']); candidates[k].append(s)
    paths = sorted(forbidden_paths | {s['image'] for rows in candidates.values() for s in rows})
    def hashed(p):
        return p, pixels((Path('data')/p).read_bytes())[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(hashed, paths))
    blocked = {hashes[p] for p in forbidden_paths}
    rows = []; selected = set(); counts = {}
    for k, group in candidates.items():
        chosen = []
        for s in group:
            h = hashes[s['image']]
            if h in blocked or h in selected: continue
            chosen.append(s); selected.add(h)
            if len(chosen) == 64: break
        counts[k] = len(chosen); rows.extend(chosen)
    audit = dict(counts=counts, images_hashed=len(paths), train_pixel_overlap=len(selected & blocked),
                 source_sha256={p:sha(p) for p in sources+held_paths}, aliases_sha256=sha('data/v5/image_identity.json'),
                 limits='Exact decoded pixels and COCO/VG aliases only; near duplicates and upstream pretraining exposure unknown')
    (WORK/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert all(n == 64 for n in counts.values()), counts
    assert not selected & blocked
    (WORK/'inputs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
    print('audit_pass', json.dumps(audit), flush=True)
    path = json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name == REV
    sys.path.insert(0,path)
    import d3_runtime
    d3_runtime.load_processor = load_processor
    m = d3_runtime.D3.from_pretrained(path,device='cuda:0',batch_size=1,local_files_only=True)
    assert m.fast is None and m.encoder is not None and m.image_unavailable is None
    features = []; errors = {}; metadata = []
    for i,s in enumerate(rows):
        if time.monotonic()-start > 1200: raise TimeoutError('exp93 20-minute budget')
        q = s['questions']['q']; options = list(q['criteria'])
        gold = s['gold']['q']['probabilities']
        assert set(options) == set(gold) and abs(sum(gold.values())-1)<1e-6
        with Image.open(Path('data')/s['image']) as im: image=im.convert('RGB')
        text = m.image_text(s.get('state',''),q,1)
        x = m.processor(text=[text],images=[image],padding=True,return_tensors='pt')
        x = {k:v.to(m.device) for k,v in x.items()}
        with torch.no_grad(), d3_runtime.sdpa_backends(m.device):
            embeds,rope = m.image_embeddings(x)
            h = m.encoder(embeds,x['attention_mask'],rope)[:,-1].float()
            p = ((h @ m.readout[:len(options)].T)/m.temperature).softmax(-1)[0]
        assert torch.isfinite(h).all() and torch.isfinite(p).all()
        if s['source'] not in errors:
            native = m.system_one(state=s.get('state',''),questions={'q':q},images=[image])['answers']['q']['probabilities']
            error = max(abs(p[j].item()-native[o]) for j,o in enumerate(options))
            assert error < 1e-5, error
            errors[s['source']] = error
        features.append(h.cpu())
        metadata.append(dict(id=s['id'],source=s['source'],image_sha256=hashes[s['image']],identity=identity(s),options=options,gold=[gold[o] for o in options]))
        if (i+1)%32==0: print('cached',i+1,'/',len(rows),'seconds',round(time.monotonic()-start,2),flush=True)
    save_file({'features':torch.cat(features).contiguous(),'initial_readout':m.readout.detach().float().cpu().contiguous()},str(WORK/'features.safetensors'))
    (WORK/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    report = dict(n=len(rows),counts=counts,revision=REV,temperature=m.temperature,native_max_probability_errors=errors,seconds=time.monotonic()-start,script_sha256=sha(__file__),inputs_sha256=sha(WORK/'inputs.jsonl'),cache_sha256=sha(WORK/'features.safetensors'),peak_cuda_bytes=torch.cuda.max_memory_allocated(),decision='cache_ready_no_training',limitations=audit['limits'])
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__ == '__main__': main()
