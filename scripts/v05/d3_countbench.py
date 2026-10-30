"""Full CountBench-derived development screening with original count choices."""
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
from d3_image_processor import load_processor
from devision.train.compare import paired_metric
WORK=Path('runs/exp100-d3-countbench-retry1')
REV='03560c26f5199c3c04129a2ccb38960c913635e0'

def main():
    WORK.mkdir(exist_ok=False);start=time.monotonic()
    source=Path('data/r13/bench/countbench.jsonl')
    samples=sorted([json.loads(x) for x in source.read_text().splitlines()],key=lambda s:hashlib.sha256(('exp100:'+s['image_id']).encode()).hexdigest())[:491]
    assert len(samples)==491 and len({s['image_id'] for s in samples})==491
    assert all((Path('data')/s['image']).is_file() for s in samples)
    (WORK/'inputs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    path=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name==REV
    sys.path.insert(0,path)
    import d3_runtime
    d3_runtime.load_processor=load_processor
    model=d3_runtime.D3.from_pretrained(path,device='cuda:0',batch_size=1,local_files_only=True)
    assert model.image_unavailable is None,model.image_unavailable
    (WORK/'IDENTITY.json').write_text(json.dumps(dict(revision=REV,snapshot=path,runtime=model.runtime_info(),image_contract=model.image_contract(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selected_sha256=hashlib.sha256((WORK/'inputs.jsonl').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest()),indent=2)+'\n')
    results={}
    for control in ('real','noimage','reversed'):
        rows=[]
        with (WORK/f'{control}.details.jsonl').open('x') as out:
            for i,s in enumerate(samples):
                if time.monotonic()-start>900:raise TimeoutError('900s screening budget')
                q=dict(s['questions']['q'])
                if control=='reversed':q['criteria']=dict(reversed(list(q['criteria'].items())))
                args=dict(state=s.get('state_text',''),questions={'q':q})
                if control!='noimage':args['images']=[str(Path('data')/s['image'])]
                tick=time.monotonic();result=model.system_one(**args)
                probs=result['answers']['q']['probabilities'];gold=s['gold']['q']['probabilities']
                assert set(probs)==set(gold)
                assert all(math.isfinite(p) and 0<=p<=1 for p in probs.values()) and abs(sum(probs.values())-1)<1e-5
                target=max(gold,key=gold.get);pred=max(probs,key=probs.get)
                r=dict(sample_id=s['id'],qid='q',image_id=s['image_id'],source=s['source'],type='choice',gold=gold,gold_answer=target,probabilities=probs,prediction=pred,correct=pred==target,nll=-math.log(max(probs[target],1e-30)),brier=sum((probs[k]-gold[k])**2 for k in gold),control=control,latency_ms=1000*(time.monotonic()-tick))
                out.write(json.dumps(r)+'\n');out.flush();rows.append(r)
                if (i+1)%16==0:print(control,i+1,len(samples),'accuracy',sum(r['correct'] for r in rows)/len(rows),flush=True)
        results[control]=rows
    summary={k:dict(n=len(v),accuracy=sum(r['correct'] for r in v)/len(v),nll=sum(r['nll'] for r in v)/len(v),brier=sum(r['brier'] for r in v)/len(v)) for k,v in results.items()}
    gain=paired_metric(results['noimage'],results['real'],metric='accuracy',resamples=5000,seed=100)
    changed=sum(a['prediction']!=b['prediction'] for a,b in zip(results['real'],results['reversed']))/len(samples)
    baseline_path=Path('runs/r21-soft/eval/bench_countbench.details.jsonl')
    baseline={r['sample_id']:r for r in map(json.loads,baseline_path.read_text().splitlines())}
    reference=[baseline[s['id']] for s in samples]
    assert all(a['gold']==b['gold'] for a,b in zip(reference,results['real']))
    comparison=paired_metric(reference,results['real'],metric='accuracy',resamples=5000,seed=100)
    (WORK/'v04.details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reference))
    passed=comparison['difference']>=.05 and comparison['ci95'][0]>0 and gain['difference']>=.05 and gain['ci95'][0]>0 and changed<=.2
    report=dict(summary=summary,v04_comparison=comparison,baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),visual_gain=gain,semantic_change_rate=changed,screening_pass=passed,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),limitations='historically monitored CountBench-derived choices, no LAION captions; unknown pretraining exposure; not ImageJevBench; no calibration or release acceptance')
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
