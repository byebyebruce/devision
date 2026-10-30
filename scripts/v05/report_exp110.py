"""Independent CPU verification of the full fixed before/after science adaptation panel."""
import json,math,random,hashlib
from pathlib import Path
from collections import defaultdict,Counter
root=Path('runs/exp110-d3-science/train')
def read(n):return [json.loads(x) for x in (root/n).read_text().splitlines()]
before=read('before.details.jsonl');after=read('after.details.jsonl');assert len(before)==len(after)==640
for a,b in zip(before,after):
    assert all(a[k]==b[k] for k in ['sample_id','image_id','gold','options','source'])
for r in before+after:
    p=r['probabilities'];g=r['gold'];assert set(p)==set(g)
    assert all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
    target=max(g,key=g.get);assert r['correct']==(max(p,key=p.get)==target)
    assert abs(r['nll']+math.log(max(p[target],1e-30)))<1e-10
rng=random.Random(110);summary={}
for source in sorted({r['source'] for r in before}):
    b=[r for r in before if r['source']==source];a=[r for r in after if r['source']==source];groups=defaultdict(list)
    for x,y in zip(b,a):groups[x['image_id']].append(int(y['correct'])-int(x['correct']))
    gs=list(groups.values());boot=[]
    for _ in range(5000):
        sampled=[rng.choice(gs) for _ in gs];boot.append(sum(map(sum,sampled))/sum(map(len,sampled)))
    boot.sort();summary[source]=dict(n=len(a),images=len(gs),before=sum(r['correct'] for r in b)/len(b),after=sum(r['correct'] for r in a)/len(a),difference=sum(int(y['correct'])-int(x['correct']) for x,y in zip(b,a))/len(a),ci95=[boot[125],boot[4874]],before_nll=sum(r['nll'] for r in b)/len(b),after_nll=sum(r['nll'] for r in a)/len(a))
controls=read('science.controls.jsonl');assert len(controls)==768
control_summary={}
for control in ['real_reload','noimage','reversed']:
    rs=[r for r in controls if r['control']==control];assert len(rs)==256
    assert [r['sample_id'] for r in rs]==[r['sample_id'] for r in after[:256]]
    for r in rs:
        p=r['probabilities'];g=r['gold'];assert set(p)==set(g) and abs(sum(p.values())-1)<1e-5 and all(math.isfinite(v) and 0<=v<=1 for v in p.values())
        assert r['correct']==(max(p,key=p.get)==max(g,key=g.get))
    control_summary[control]=dict(accuracy=sum(r['correct'] for r in rs)/256,nll=sum(r['nll'] for r in rs)/256,semantic_changes=sum(max(a['probabilities'],key=a['probabilities'].get)!=max(b['probabilities'],key=b['probabilities'].get) for a,b in zip(after[:256],rs))/256)
identity=json.loads((root/'IDENTITY.json').read_text())
assert hashlib.sha256((root/'executed_script.py').read_bytes()).hexdigest()==identity['script_sha256']
reloaded=[r for r in controls if r['control']=='real_reload']
assert max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(after[:256],reloaded) for k in a['probabilities'])<1e-5
history=read('train.jsonl');assert [h['step'] for h in history]==list(range(1,257))
assert all(math.isfinite(h[k]) for h in history for k in ['ce','gradient_norm','seconds'])
from safetensors import safe_open
patches={}
for step in [0,64,128,192,256]:
    p=root/f'adapter-step{step}.safetensors'
    with safe_open(p,framework='pt',device='cpu') as f: assert len(f.keys())==32
    patches[step]=hashlib.sha256(p.read_bytes()).hexdigest();assert (root/f'optimizer-step{step}.pt').is_file()
s=summary['lv-scienceqa'];passed=s['difference']>=.03 and s['ci95'][0]>0 and s['after_nll']<=s['before_nll'] and all(summary[k]['difference']>=-.02 for k in ['chartqa','screenqa_cf'])
result=dict(verified=True,summary=summary,controls=control_summary,continuation_gate_pass=passed,training_steps=len(history),first32_ce=sum(h['ce'] for h in history[:32])/32,last32_ce=sum(h['ce'] for h in history[-32:])/32,patch_sha256=patches,limitations='monitoring only; no official benchmark, independent calibration, full regression or release acceptance; source-level missing-image gap and unknown backbone exposure')
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
