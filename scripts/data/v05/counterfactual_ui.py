"""Native human same-question four-screen tuples: identical text, balanced true answers."""
import os,io,json,random,sys
from collections import defaultdict,Counter
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image,ImageDraw
import textwrap
from screenqa import ROOT,norm,rank,good,iou
sys.path.insert(0,str(ROOT/'scripts/v05'))
from chart_data import digest,pixels,canonical,write_rows


def score(s):return digest(('85:'+s).encode())
def alias(a,b):return canonical(a)==canonical(b) or norm(a) in norm(b) or norm(b) in norm(a)


def main():
 os.chdir(ROOT);root=Path('data');out=root/'v05/exp85'
 if out.exists():raise RuntimeError('Refusing overwrite exp85')
 out.mkdir(parents=True)
 source=json.loads(Path('runs/v05-prep/screenqa_source.json').read_text());tables={};cache={};stats=Counter()
 import re
 for split,pattern in [('train','train-*'),('dev','validation-*')]:
  rows=[]
  for p in sorted((Path(source['snapshot'])/'data').glob(pattern)):
   for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
    for r in batch.to_pylist():
     raw=r.pop('image')['bytes'];h=digest(raw)
     if h not in cache:
      pix,old=pixels(raw)
      with Image.open(io.BytesIO(raw)) as im:w,height=im.size
      path='v05/exp82/images/'+pix+'.png'
      if not (root/path).exists():raise RuntimeError('Expected decoded source missing')
      cache[h]=(pix,old,path,w,height)
     pix,old,path,w,height=cache[h]
     r.update(pixel=pix,oldpixel=old,image=path,width=w,height=height);rows.append(r)
  tables[split]=rows;print(split,'decoded',len(rows),flush=True)
 sys.path.insert(0,'scripts/data/n2');from held import load
 held=load('data');valpix={r['pixel'] for r in tables['dev']}
 groups={}
 for split,rows in tables.items():
  byq=defaultdict(list)
  for r in rows:
   gs=r['ground_truth'];q=norm(r['question']);stats[split+'_raw']+=1
   if split=='train' and (r['pixel'] in valpix or r['oldpixel'] in held.pixels):continue
   if not gs or any(len(g['ui_elements'])!=1 for g in gs):continue
   es=[g['ui_elements'][0] for g in gs];t=es[0];w,h=r['width'],r['height']
   if len({canonical(e['text']) for e in es})!=1 or not all(good(e,w,h) for e in es):continue
   if any(iou(t['bounds'],e['bounds'])<.6 for e in es):continue
   if not all(norm(t['text']) in norm(g['full_answer']) for g in gs):continue
   if re.match(r'^(is|are|can|do|does|did|will|would|should|has|have|was|were)\b|^how (many|much)\b',q):continue
   if min(256/w,256/h)*(t['bounds'][3]-t['bounds'][1])<6:continue
   r['answer']=t['text'].strip();byq[q].append(r)
  used=set();tuples=[]
  for q in sorted(byq,key=score):
   remaining=sorted(byq[q],key=lambda r:score(r['pixel']+r['answer']))
   while True:
    chosen=[]
    for r in remaining:
     if r['pixel'] in used or r['pixel'] in {x['pixel'] for x in chosen}:continue
     if any(alias(r['answer'],x['answer']) for x in chosen):continue
     chosen.append(r)
     if len(chosen)==4:break
    if len(chosen)<4:break
    used.update(r['pixel'] for r in chosen)
    gid='uiquad:'+score(q+':'+':'.join(r['pixel'] for r in chosen))
    opts=[r['answer'] for r in chosen];random.Random(int(score(gid)[:16],16)).shuffle(opts)
    # Only whitespace/case equivalent original questions are grouped; use one exact prompt for all four.
    prompt=chosen[0]['question']
    records=[]
    for i,r in enumerate(chosen):
     sid=gid+':'+str(i)
     records.append(dict(id=sid,source='screenqa_cf',image=r['image'],image_id='screenqa:'+r['pixel'],counterfactual_group=gid,
                         questions={'q':dict(type='choice',instructions=prompt,criteria=dict.fromkeys(opts))},
                         gold={'q':{'probabilities':{v:float(v==r['answer']) for v in opts}}}))
    tuples.append(dict(id=gid,records=records,annotations=[dict(screen_id=r['screen_id'],original_question=r['question'],gold=r['answer'],raters=r['ground_truth']) for r in chosen]))
  groups[split]=sorted(tuples,key=lambda g:score(g['id']));print(split,'eligible tuples',len(tuples),flush=True)
 availability={k:len(v) for k,v in groups.items()}
 (out/'AVAILABILITY.json').write_text(json.dumps(availability,indent=2)+'\n')
 if len(groups['train'])<72 or len(groups['dev'])<16:raise RuntimeError('Insufficient independent tuples')
 selected={'calib':groups['train'][:8],'ui_train':groups['train'][8:72],'ui_dev':groups['dev'][:64]}
 selected['fit']=selected['ui_train'][:8]
 rows={k:[r for g in gs for r in g['records']] for k,gs in selected.items()}
 replay=[json.loads(l) for l in Path('data/v05/exp81/train.jsonl').open()]
 chart=sorted([r for r in replay if r['source']=='chartqa'],key=lambda r:score(r['id']))[:2048]
 old=sorted([r for r in replay if r['source']!='chartqa'],key=lambda r:score(r['id']))[:4096]
 rows['train']=rows['ui_train']*8+chart+old;random.Random(85).shuffle(rows['train'])
 files={n:write_rows(out/(n+'.jsonl'),rs) for n,rs in rows.items()}
 for name in ['ui_train','ui_dev','calib']:
  assert len({r['image_id'] for r in rows[name]})==len(rows[name])
  for g in selected[name]:
   rs=g['records'];assert len({json.dumps(r['questions'],sort_keys=False) for r in rs})==1
   assert {max(r['gold']['q']['probabilities'],key=r['gold']['q']['probabilities'].get) for r in rs}==set(rs[0]['questions']['q']['criteria'])
 allgroups={n:gs for n,gs in selected.items() if n!='fit'}
 (out/'GROUPS.json').write_text(json.dumps(allgroups,indent=2)+'\n')
 audit=selected['ui_train'][:6]+selected['ui_dev'][:6]
 (out/'AUDIT48.json').write_text(json.dumps(audit,indent=2)+'\n')
 for page in range(3):
  canvas=Image.new('RGB',(1600,1400),'white');draw=ImageDraw.Draw(canvas)
  for j,g in enumerate(audit[page*4:(page+1)*4]):
   y=j*350;draw.text((5,y+2),f'{page*4+j+1}: '+g['records'][0]['questions']['q']['instructions'],fill='black')
   for k,r in enumerate(g['records']):
    x=k*400
    with Image.open(root/r['image']) as im:
     im=im.convert('RGB');im.thumbnail((240,256));canvas.paste(im,(x,y+25))
    opts=list(r['questions']['q']['criteria']);gold=max(r['gold']['q']['probabilities'],key=r['gold']['q']['probabilities'].get)
    text='\n'.join(textwrap.wrap('Options: '+' | '.join(opts),65)+textwrap.wrap('Gold: '+gold,65))
    draw.multiline_text((x,y+285),text,fill='black')
  canvas.save(out/f'audit48-{page}.png')
 (out/'MANIFEST.json').write_text(json.dumps(dict(protocol='exp85-native-ui-balanced-quadruples',source=source,files=files,availability=availability,
       seed=85,steps=1024,ui_unique_questions=256,ui_presentations=2048,question_only_accuracy_bound=.25,
       limits=['native human text choices, not official open QA','development sources previously used for decision','pixel audit and manual review pending','base exposure unknown']),indent=2)+'\n')
 print('Built balanced UI tuples; audit required before GPU launch.',flush=True)


if __name__=='__main__':main()
