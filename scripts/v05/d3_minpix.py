"""exp113: D3 accuracy on the ScienceQA failure families with the default vs a 4x larger minimum image size."""
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('D3_FAST','0')
import torch
import d3_image_processor
WORK=Path('runs/exp113-d3-minpix')
REV='03560c26f5199c3c04129a2ccb38960c913635e0'

def family(s):
    s=s.lower()
    if 'magnet' in s:return 'magnets'
    if 'weather or the climate' in s:return 'weather'
    if 'highlighted' in s or 'capital' in s or 'which country' in s or 'which state' in s:return 'map'
    return None

def main():
    WORK.mkdir(exist_ok=False);start=time.monotonic()
    samples=[json.loads(x) for x in Path('runs/exp99-d3-scienceqa/inputs.jsonl').read_text().splitlines()]
    samples=[s for s in samples if family(s['questions']['q']['instructions'])]
    path=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name==REV
    sys.path.insert(0,path)
    import d3_runtime
    summary={}
    for arm,minpix in (('default',65536),('min4x',262144)):
        def load(root,minpix=minpix):
            p=d3_image_processor.load_processor(root);p.image_processor.size={'shortest_edge':minpix,'longest_edge':1638400};return p
        d3_runtime.load_processor=load
        model=d3_runtime.D3.from_pretrained(path,device='cuda:0',batch_size=1,local_files_only=True)
        assert model.image_unavailable is None and model.processor.image_processor.size['shortest_edge']==minpix
        with (WORK/f'{arm}.details.jsonl').open('x') as out:
            for s in samples:
                if time.monotonic()-start>600:raise TimeoutError('600s budget')
                q=s['questions']['q']
                result=model.system_one(state=s.get('state_text',''),questions={'q':q},images=[str(Path('data')/s['image'])])
                probs=result['answers']['q']['probabilities'];gold=s['gold']['q']['probabilities']
                assert set(probs)==set(gold) and all(math.isfinite(p) for p in probs.values())
                target=max(gold,key=gold.get);pred=max(probs,key=probs.get)
                r=dict(sample_id=s['id'],image_id=s['image_id'],family=family(q['instructions']),gold=gold,probabilities=probs,prediction=pred,correct=pred==target,nll=-math.log(max(probs[target],1e-30)))
                out.write(json.dumps(r)+'\n');out.flush()
                summary.setdefault(arm,{}).setdefault(r['family'],[]).append(r['correct'])
        del model;torch.cuda.empty_cache()
    report={arm:{f:dict(n=len(v),correct=sum(v)) for f,v in fams.items()} for arm,fams in summary.items()}
    report['seconds']=time.monotonic()-start
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
