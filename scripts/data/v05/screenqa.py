"""Conservative original-human ScreenQA choices; no gold crop or generated answers."""
from collections import Counter, defaultdict
import io,json,random,re,sys,os
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/v05'))
from chart_data import pixels,digest,canonical,write_rows


def norm(s):return ' '.join(s.casefold().split())
def rank(s):return digest(('82:'+s).encode())
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(a):return (a[2]-a[0])*(a[3]-a[1])
def iou(a,b):return overlap(a,b)/max(1,area(a)+area(b)-overlap(a,b))
def good(e,w,h):
 b=e['bounds'];t=e['text'].strip()
 return len(b)==4 and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h and 1<=len(t)<=60 and bool(re.search('[A-Za-z0-9]',t))


def main():
 os.chdir(ROOT)
 os_root=ROOT/'data';out=os_root/'v05/exp82v2'
 if (out/'MANIFEST.json').exists():raise RuntimeError('Refusing to overwrite completed exp82')
 out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
 source=json.loads((ROOT/'runs/v05-prep/screenqa_source.json').read_text())
 tables={};cache={};stats=Counter();size={}
 for split,pattern in [('train','train-*'),('dev','validation-*')]:
  groups=defaultdict(list)
  for p in sorted((Path(source['snapshot'])/'data').glob(pattern)):
   for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
    for row in batch.to_pylist():
     raw=row.pop('image')['bytes'];h=digest(raw)
     if h not in cache:
      sha,old=pixels(raw)
      with Image.open(io.BytesIO(raw)) as im:
       path=os_root/'v05/exp82/images'/(sha+'.png')
       if not path.exists():im.convert('RGB').save(path)
       size[sha]=im.size
      cache[h]=(sha,old,str(path.relative_to(os_root)))
     sha,old,path=cache[h]
     groups[sha].append(dict(row,pixel=sha,oldpixel=old,image=path))
  tables[split]=groups
 valpix=set(tables['dev']);calibpix={h for h in tables['train'] if int(rank(h)[:8],16)%10==0}
 sys.path.insert(0,str(ROOT/'scripts/data/n2'))
 from held import load
 held=load('data');converted=defaultdict(list);audit={}
 for split,groups in tables.items():
  for sha,rows in groups.items():
   if split=='train' and (sha in valpix or rows[0]['oldpixel'] in held.pixels):
    stats['excluded_held_rows']+=len(rows);continue
   actual='calib' if split=='train' and sha in calibpix else split
   w,h=size[sha]
   pool=[e for r in rows for g in r['ground_truth'] for e in g['ui_elements'] if good(e,w,h)]
   for index,r in enumerate(rows):
    stats[split+'_raw']+=1;gs=r['ground_truth'];reason=None
    if not gs or any(len(g['ui_elements'])!=1 for g in gs):reason='not_single_target'
    elif len({canonical(g['ui_elements'][0]['text']) for g in gs})!=1:reason='raters_disagree'
    else:
     targets=[g['ui_elements'][0] for g in gs];target=targets[0]
     if not all(good(e,w,h) for e in targets) or any(iou(target['bounds'],e['bounds'])<.6 for e in targets):reason='invalid_or_disagreeing_box'
     elif not all(norm(target['text']) in norm(g['full_answer']) for g in gs):reason='answer_not_text_span'
     elif re.match(r'^(is|are|can|do|does|did|will|would|should|has|have|was|were)\b|^how (many|much)\b',r['question'].lower()):reason='non_span_question'
     elif min(256/w,256/h)*(target['bounds'][3]-target['bounds'][1])<6:reason='target_box_below_6px'
     else:
      neg={}
      for e in pool:
       if canonical(e['text'])!=canonical(target['text']) and norm(e['text']) not in norm(target['text']) and norm(target['text']) not in norm(e['text']) and not any(overlap(e['bounds'],t['bounds'])>0 for t in targets):
        neg.setdefault(canonical(e['text']),e)
      if len(neg)<3:reason='fewer_than_three_same_screen_negatives'
      else:
       sid='screenqa:'+actual+':'+sha+':'+str(index)
       rng=random.Random(int(rank(sid)[:16],16));selected=[target]+rng.sample([neg[k] for k in sorted(neg)],3);rng.shuffle(selected)
       opts=[e['text'].strip() for e in selected];gold=target['text'].strip()
       x=dict(id=sid,source='screenqa',image=r['image'],image_id='screenqa:'+sha,
              questions={'q':dict(type='choice',instructions=r['question'],criteria=dict.fromkeys(opts))},
              gold={'q':{'probabilities':{t:float(t==gold) for t in opts}}})
       converted[actual].append(x)
       audit[sid]=dict(screen_id=r['screen_id'],question=r['question'],full_answers=[g['full_answer'] for g in gs],
                       target=target,candidates=selected,target_box_height_at256=min(256/w,256/h)*(target['bounds'][3]-target['bounds'][1]))
    if reason:stats[split+'_'+reason]+=1
 for rows in converted.values():rows.sort(key=lambda r:rank(r['id']))
 availability={k:len(v) for k,v in converted.items()}
 (out/'AVAILABILITY.json').write_text(json.dumps(dict(source=source,counts=availability,filters=dict(stats)),indent=2)+'\n')
 print(availability,dict(stats),flush=True)
 if len(converted['train'])<512 or len(converted['dev'])<128 or len(converted['calib'])<32:
  raise RuntimeError('Insufficient eligible data; do not train or pad')
 train=converted['train'][:512];dev=converted['dev'][:256];calib=converted['calib'][:64]
 replay=[json.loads(line) for line in (os_root/'v05/exp81/train.jsonl').open()]
 chart=sorted([r for r in replay if r['source']=='chartqa'],key=lambda r:rank(r['id']))[:2048]
 old=sorted([r for r in replay if r['source']!='chartqa'],key=lambda r:rank(r['id']))[:4096]
 # Actual full mixed training image isolation is audited separately before launch.
 mixed=train*4+chart+old;random.Random(82).shuffle(mixed)
 files={name:write_rows(out/(name+'.jsonl'),rows) for name,rows in dict(train=mixed,ui_train=train,ui_dev=dev,calib=calib,fit=train[:64]).items()}
 sample=train[:25]+dev[:25]
 (out/'AUDIT50.json').write_text(json.dumps([dict(sample=r,annotation=audit[r['id']]) for r in sample],indent=2)+'\n')
 for page in range(5):
  canvas=Image.new('RGB',(1400,1550),'white');d=ImageDraw.Draw(canvas)
  for j,r in enumerate(sample[page*10:(page+1)*10]):
   import textwrap
   x=(j%2)*700;y=(j//2)*310
   with Image.open(os_root/r['image']) as im:
    im=im.convert('RGB');im.thumbnail((256,256));canvas.paste(im,(x,y))
   text=f'{page*10+j+1}. '+r['questions']['q']['instructions']+'\n'+'\n'.join(f'{i+1}: {s}' for i,s in enumerate(r['questions']['q']['criteria']))+'\nGOLD: '+max(r['gold']['q']['probabilities'],key=r['gold']['q']['probabilities'].get)
   lines=[]
   for line in text.splitlines():lines+=textwrap.wrap(line,49)
   d.multiline_text((x+260,y+3),'\n'.join(lines),fill='black',spacing=4)
  canvas.save(out/f'audit50-{page}.png')
 manifest=dict(protocol='exp82-screenqa-human-text-choices',source=source,files=files,availability=availability,filters=dict(stats),
               seed=82,train_unique_ui=512,ui_presentations=2048,steps=1024,
               limits=['derived choices, not official open QA score','unannotated acceptable alternatives may remain','pixel isolation audit pending','base exposure and near duplicates unknown'])
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Built exp82; manual audit and full image isolation required before training.',flush=True)


if __name__=='__main__':main()
