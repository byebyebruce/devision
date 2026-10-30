"""Conservative FUNSD linked-entity derived decisions; original test untouched."""
import hashlib
import io
import json
from pathlib import Path
import random
import re
import zipfile
from collections import Counter
from PIL import Image

ROOT=Path('data/v05/funsd')

def norm(s):
    return re.sub(r'\W+', ' ', s.casefold()).strip()

def pixel(im):
    im=im.convert('RGB')
    return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def main():
    out=ROOT/'derived';out.mkdir(exist_ok=False)
    z=zipfile.ZipFile(ROOT/'raw/dataset.zip');docs=[];seen=set();stats=Counter()
    for name in sorted(z.namelist()):
        if not name.startswith('dataset/training_data/annotations/') or not name.endswith('.json'):continue
        doc=Path(name).stem;raw=z.read(f'dataset/training_data/images/{doc}.png')
        im=Image.open(io.BytesIO(raw));sha=pixel(im)
        if sha in seen:stats['duplicate_document']+=1;continue
        seen.add(sha);docs.append((sha,doc,raw,json.loads(z.read(name))))
    docs.sort();assert len(docs)>=100
    dev_hash={d[0] for d in docs[:30]}
    # Audit actual image paths in every pre-existing v05 JSONL; no GPU needed.
    prior=set()
    for p in Path('data/v05').rglob('*.jsonl'):
        if ROOT in p.parents:continue
        for line in p.read_text().splitlines():
            s=json.loads(line);v=s.get('image')
            if isinstance(v,str) and (Path('data')/v).is_file():prior.add(Path('data')/v)
    prior_pixels=set()
    for p in sorted(prior):
        with Image.open(p) as im:prior_pixels.add(pixel(im))
    assert not seen & prior_pixels,'existing v05 image leakage'
    outputs={'train':[],'dev':[]};manifest=[]
    for sha,doc,raw,ann in docs:
        split='dev' if sha in dev_hash else 'train'
        forms={e['id']:e for e in ann['form']}
        links={tuple(sorted(pair)) for e in forms.values() for pair in e['linking']}
        qa={i:set() for i in forms};aq={i:set() for i in forms}
        for a,b in links:
            if a not in forms or b not in forms:stats['broken_link']+=1;continue
            if forms[a]['label']=='answer':a,b=b,a
            if forms[a]['label']=='question' and forms[b]['label']=='answer':qa[a].add(b);aq[b].add(a)
        qcounts=Counter(norm(e['text']) for e in forms.values() if e['label']=='question')
        acounts=Counter(norm(e['text']) for e in forms.values() if e['label']=='answer')
        eligible=[e for e in forms.values() if e['label']=='answer' and len(aq[e['id']])==1 and norm(e['text']) and acounts[norm(e['text'])]==1]
        candidates=[]
        for q in forms.values():
            if q['label']!='question':continue
            stats['questions']+=1
            if len(qa[q['id']])!=1 or qcounts[norm(q['text'])]!=1 or not norm(q['text']):stats['nonunique_question']+=1;continue
            target=forms[next(iter(qa[q['id']]))]
            if target not in eligible:stats['nonunique_answer']+=1;continue
            rng=random.Random(f'98:{doc}:{q["id"]}')
            negatives=[e for e in eligible if e['id']!=target['id']]
            rng.shuffle(negatives);chosen=[target]
            for e in negatives:
                if all(norm(e['text']) not in norm(c['text']) and norm(c['text']) not in norm(e['text']) for c in chosen):chosen.append(e)
                if len(chosen)==4:break
            if len(chosen)!=4:stats['insufficient_distinct_answers']+=1;continue
            rng.shuffle(chosen);keys=list('ABCD');gold=keys[chosen.index(target)]
            row=dict(id=f'funsd:{doc}:{q["id"]}',image=f'v05/funsd/derived/images/{doc}.png',image_id=f'funsd:{doc}',source='funsd_binding',state='',questions={'q':dict(type='choice',instructions=f'Which value is associated with the field "{q["text"].strip()}" in this document?',criteria={k:e['text'].strip() for k,e in zip(keys,chosen)})},gold={'q':{'probabilities':{k:float(k==gold) for k in keys}}})
            candidates.append((row,dict(sample_id=row['id'],split=split,pixel_sha256=sha,question=q,candidates=chosen,gold_entity=target['id'])))
        for row,m in candidates[:4]:outputs[split].append(row);manifest.append(m)
        if candidates:
            (out/'images').mkdir(exist_ok=True);(out/'images'/f'{doc}.png').write_bytes(raw)
    for split,rows in outputs.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    report=dict(original_train_documents=len(docs),original_test_parsed=False,prior_images_audited=len(prior),prior_pixel_overlap=len(seen&prior_pixels),split_overlap=0,stats=dict(stats),splits={k:dict(n=len(v),documents=len({r['image_id'] for r in v}),gold_positions=dict(Counter(max(r['gold']['q']['probabilities'],key=r['gold']['q']['probabilities'].get) for r in v))) for k,v in outputs.items()},archive_sha256=hashlib.sha256((ROOT/'raw/dataset.zip').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_license='not present in original archive; research only; no redistribution')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert len(outputs['dev'])>=64 and len({r['image_id'] for r in outputs['dev']})>=16

if __name__=='__main__':main()
