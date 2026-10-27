"""Bounded YAML-driven D3 adaptation pilot; exports base-dependent patches."""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['D3_FAST'] = '0'
import hashlib, json, random, sys, time
from pathlib import Path
import torch, yaml
from PIL import Image
from safetensors.torch import save_file, load_file
from d3_image_processor import load_processor

cfg=yaml.safe_load(Path(sys.argv[1]).read_text()); out=Path(cfg['out'])
out.mkdir(parents=True,exist_ok=True); (out/'STARTED').open('x').close()
(out/'config.yaml').write_text(Path(sys.argv[1]).read_text())
t0=time.monotonic(); torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
cache=Path(cfg['cache']); raw=(cache/'inputs.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==json.loads((cache/'report.json').read_text())['inputs_sha256']
rows=[json.loads(x) for x in raw.splitlines()]; assert len(rows)==256
random.Random(cfg['seed']).shuffle(rows)
(out/'inputs.jsonl').write_bytes(raw)
root=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
assert Path(root).name=='03560c26f5199c3c04129a2ccb38960c913635e0'
sys.path.insert(0,root)
import d3_runtime
d3_runtime.load_processor=load_processor
m=d3_runtime.D3.from_pretrained(root,device='cuda:0',batch_size=1,local_files_only=True)
class LowRank(torch.nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base; self.enabled=True
        self.a=torch.nn.Parameter(torch.randn(4,base.in_features,device=m.device)*.01)
        self.b=torch.nn.Parameter(torch.zeros(base.out_features,4,device=m.device))
    def forward(self,x):
        if not self.enabled: return self.base(x)
        return self.base(x)+((x.float()@self.a.T)@self.b.T).to(x.dtype)
mods={}; layer=m.encoder.language_model.layers[-1]
for name in ['q_proj','v_proj']:
    base=getattr(layer.self_attn,name); assert isinstance(base,torch.nn.Linear)
    mods[name]=LowRank(base); setattr(layer.self_attn,name,mods[name])
params=[p for mod in mods.values() for p in [mod.a,mod.b]]
assert sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)==26624
opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=0)
def check():
    if time.monotonic()-t0>cfg['budget_seconds']: raise TimeoutError('pilot budget')
def prepare(s):
    check(); q=s['questions']['q']; options=list(q['criteria']) if q['type']=='choice' else ['false','true']
    with Image.open(Path('data')/s['image']) as im: image=im.convert('RGB')
    x=m.processor(text=[m.image_text(s.get('state',''),q,1)],images=[image],padding=True,return_tensors='pt')
    x={k:v.to(m.device) for k,v in x.items()}
    with torch.no_grad(),d3_runtime.sdpa_backends(m.device): e,r=m.image_embeddings(x)
    return options,e.detach(),r,x['attention_mask']
def logits(e,r,mask,n):
    with d3_runtime.sdpa_backends(m.device): h=m.encoder(e,mask,r)[:,-1].float()
    return (h@m.readout[:n].T)[0]
def patch(): return {f'{n}.{k}':getattr(mod,k).detach().cpu().contiguous() for n,mod in mods.items() for k in ['a','b']}
history=[]; opt.zero_grad(); accumulated=0.
for i,s in enumerate(rows):
    options,e,r,mask=prepare(s); g=torch.tensor([s['gold']['q']['probabilities'][o] for o in options],device=m.device)
    value=-(g*logits(e,r,mask,len(options)).log_softmax(-1)).sum()
    assert torch.isfinite(value) and value.item()<20
    (value/cfg['batch_size']).backward(); accumulated+=value.item()/cfg['batch_size']
    if (i+1)%cfg['batch_size']==0:
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True).item()
        opt.step(); opt.zero_grad(); step=(i+1)//cfg['batch_size']
        rec=dict(step=step,ce=accumulated,gradient_norm=grad,seconds=time.monotonic()-t0)
        history.append(rec); accumulated=0.; print('train',json.dumps(rec),flush=True)
        if step%cfg['save_every']==0:
            save_file(patch(),str(out/f'lora-step{step}.safetensors'))
            torch.save(dict(step=step,optimizer=opt.state_dict(),config=cfg),out/f'optimizer-step{step}.pt')
loaded=load_file(str(out/'lora-step64.safetensors'))
assert all(torch.equal(v,loaded[k]) for k,v in patch().items())
# Explicitly re-inject the saved patch before evaluating.
with torch.no_grad():
    for name,mod in mods.items():
        for k in ['a','b']: getattr(mod,k).copy_(loaded[f'{name}.{k}'])
held=[]
protect=[json.loads(x) for x in Path('data/v05/exp81/protect.jsonl').read_text().splitlines()]
for source in ['coco-size','coco-position']: held.extend(s for s in protect if s['source']==source)
for p in ['data/v05/exp81/chart_dev.jsonl','data/v05/exp85/ui_dev.jsonl']:
    held.extend([json.loads(x) for x in Path(p).read_text().splitlines()][:128])
assert len(held)==384 and not {s['image'] for s in held}&{s['image'] for s in rows}
results=[]; export_error=None
with (out/'validation.details.jsonl').open('x') as f, torch.no_grad():
    for i,s in enumerate(held):
        options,e,r,mask=prepare(s)
        for mod in mods.values(): mod.enabled=False
        before=(logits(e,r,mask,len(options))/m.temperature).softmax(-1).cpu().tolist()
        for mod in mods.values(): mod.enabled=True
        after=(logits(e,r,mask,len(options))/m.temperature).softmax(-1).cpu().tolist()
        assert all(torch.isfinite(torch.tensor(p)).all() and abs(sum(p)-1)<1e-5 for p in [before,after])
        if i==0:
            native=m.system_one(state=s.get('state',''),questions=s['questions'],images=[str(Path('data')/s['image'])])['answers']['q']['probabilities']
            export_error=max(abs(native[o]-after[j]) for j,o in enumerate(options)); assert export_error<1e-5
        result=dict(id=s['id'],image=s['image'],source=s['source'],options=options,gold=[s['gold']['q']['probabilities'][o] for o in options],before=before,after=after)
        results.append(result); f.write(json.dumps(result)+'\n'); f.flush()
        if (i+1)%32==0: print('validation',i+1,'/',len(held),flush=True)
rng=random.Random(cfg['seed']); summary={}
for source in sorted({r['source'] for r in results}):
    group=[r for r in results if r['source']==source]; clusters={}; b=[]; a=[]
    for r in group:
        target=max(range(len(r['gold'])),key=r['gold'].__getitem__)
        bc=int(max(range(len(r['before'])),key=r['before'].__getitem__)==target)
        ac=int(max(range(len(r['after'])),key=r['after'].__getitem__)==target)
        b.append(bc); a.append(ac); clusters.setdefault(r['image'],[]).append(ac-bc)
    diffs=[]; groups=list(clusters.values())
    for _ in range(5000):
        sample=[rng.choice(groups) for _ in groups]; diffs.append(sum(map(sum,sample))/sum(map(len,sample)))
    diffs.sort(); summary[source]=dict(n=len(group),images=len(groups),before=sum(b)/len(b),after=sum(a)/len(a),delta=(sum(a)-sum(b))/len(a),ci95=[diffs[125],diffs[4874]])
passed=all(v['delta']>=(.03 if k in ['coco-size','coco-position'] else -.02) for k,v in summary.items())
report=dict(summary=summary,gate_pass=passed,train_history=history,export_probability_error=export_error,revision=Path(root).name,inputs_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-t0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),limitations='monitoring slices; base-dependent LoRA patch; no final test, calibration, performance or release acceptance')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
