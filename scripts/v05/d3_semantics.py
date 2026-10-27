"""Native D3 follow-up panels, fixed sources, no training."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('D3_FAST','0')
import torch
from huggingface_hub import snapshot_download
from d3_image_processor import load_processor
REV='03560c26f5199c3c04129a2ccb38960c913635e0'
WORK=Path('runs/exp90-d3-semantics')

def main():
    screening=json.loads(Path('runs/exp87-d3-ui-pil/native_report.json').read_text())
    assert screening['decision']=='advance_to_other_panels',screening['decision']
    (WORK/'STARTED').open('x').close();start=time.monotonic()
    path=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name==REV
    sys.path.insert(0,path)
    import d3_runtime
    d3_runtime.load_processor=load_processor
    model=d3_runtime.D3.from_pretrained(path,device='cuda:0',batch_size=1,local_files_only=True)
    assert model.image_unavailable is None,model.image_unavailable
    identity=dict(repo='vllm-sr/d3-lite',revision=REV,runtime=model.runtime_info(),image_contract=model.image_contract(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest())
    (WORK/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
    panels=[('protect','data/v05/exp81/protect.jsonl')]
    for panel,source in panels:
        source=Path(source);samples=[json.loads(x) for x in source.read_text().splitlines()]
        samples=[s for s in samples if s['source'] in ('coco-position','coco-size')]
        shift=next(k for k in range(1,len(samples)) if all(s['image']!=samples[(i+k)%len(samples)]['image'] for i,s in enumerate(samples)))
        for control in ['semantic']:
            rows=[]
            with (WORK/f'{panel}.{control}.details.jsonl').open('x') as out:
                for i,s in enumerate(samples):
                    if time.monotonic()-start>300:raise TimeoutError('exp90 5-minute budget')
                    image=s['image']
                    if control=='mismatched':
                        assert image!=s['image'],'adjacent rotation shares image; revise before running this panel'
                    q=dict(s['questions']['q'])
                    q['instructions'] += (' Compare only the visible pixel area occupied by each named object in this image, excluding occluded parts. Do not compare their real-world sizes.' if s['source']=='coco-size' else ' Use image coordinates as viewed: left/right refer to the left/right half of this picture, and top/bottom to its upper/lower half. Judge the center of the named object.')
                    torch.cuda.synchronize();tick=time.perf_counter()
                    result=model.system_one(state=s.get('state',''),questions={'q':q},images=[str(Path('data')/image)])
                    torch.cuda.synchronize();latency=1000*(time.perf_counter()-tick)
                    answer=result['answers']['q']
                    probs=answer['probabilities'] if q['type']=='choice' else {'false':1-answer['noul'],'true':answer['noul']}
                    gold=s['gold']['q']['probabilities']
                    assert set(probs)==set(gold)
                    assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5
                    target=max(gold,key=gold.get);pred=max(probs,key=probs.get)
                    row=dict(sample_id=s['id'],qid='q',image_id=s['image_id'],image_used=image,source=s['source'],type=q['type'],
                        gold=gold,gold_answer=target,probabilities=probs,prediction=pred,correct=pred==target,
                        nll=-sum(v*math.log(max(probs[k],1e-30)) for k,v in gold.items()),
                        brier=sum((probs[k]-v)**2 for k,v in gold.items()),control=control,latency_ms=latency,usage=result['usage'])
                    out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
                    if i%32==0:print(panel,control,i+1,'accuracy',sum(x['correct'] for x in rows)/len(rows),flush=True)
            summary=dict(n=len(rows),accuracy=sum(x['correct'] for x in rows)/len(rows),nll=sum(x['nll'] for x in rows)/len(rows),mismatch_rotation=shift,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            (WORK/f'{panel}.{control}.json').write_text(json.dumps(summary,indent=2)+'\n')
    (WORK/'STATUS.json').write_text(json.dumps(dict(state='completed'))+'\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        (WORK/'STATUS.json').write_text(json.dumps(dict(state='execution_failed',error=repr(exc)))+'\n');raise
