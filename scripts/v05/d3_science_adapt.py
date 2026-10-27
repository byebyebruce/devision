"""YAML-driven bounded real-science adapter training, with native paired regression."""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com');os.environ['D3_FAST']='0'
import hashlib,json,random,sys,time,math
from pathlib import Path
from collections import Counter
import torch,yaml
from PIL import Image
from safetensors.torch import save_file,load_file
from d3_image_processor import load_processor
cfg=yaml.safe_load(Path(sys.argv[1]).read_text());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();torch.manual_seed(cfg['seed']);torch.set_num_threads(4);rng=random.Random(cfg['seed'])
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pool=Path(cfg['pool']); manifest=json.loads((pool/'MANIFEST.json').read_text())
for name,h in manifest['outputs'].items():assert sha(pool/f'{name}.jsonl')==h
science=read(pool/'science.jsonl');replay=read(pool/'replay.jsonl')
sources={k:[s for s in science if s['source']==k] for k in ['scienceqa','ai2d','tqa']}
assert replay and all(sources.values())
# Each update: 2 examples per science source + 2 replay. Fixed shuffled order per source.
replay_groups={
    'ui':[s for s in replay if s['source'].startswith('screenqa')],
    'chart':[s for s in replay if s['source']=='chartqa'],
    'old':[s for s in replay if s['source'] not in {'chartqa','sqa-synth'} and not s['source'].startswith('screenqa')],
}
assert all(replay_groups.values())
for rs in list(sources.values())+list(replay_groups.values()):rng.shuffle(rs)
train=[]
for local_i in range(cfg['steps']):
    i=local_i+cfg.get('data_offset_steps',0)
    group=[]
    for rs in sources.values():group += [rs[(i*2+j)%len(rs)] for j in range(2)]
    replay_keys=[('ui','chart'),('ui','old'),('chart','old')][i%3]
    group += [replay_groups[k][i%len(replay_groups[k])] for k in replay_keys];rng.shuffle(group);train+=group
assert cfg['batch_size']==8
(out/'train.inputs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in train));(out/'config.yaml').write_text(Path(sys.argv[1]).read_text())
root=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot'];assert Path(root).name=='03560c26f5199c3c04129a2ccb38960c913635e0'
sys.path.insert(0,root);import d3_runtime
d3_runtime.load_processor=load_processor
m=d3_runtime.D3.from_pretrained(root,device='cuda:0',batch_size=1,local_files_only=True)
class LowRank(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.enabled=True
        self.a=torch.nn.Parameter(torch.randn(cfg['rank'],base.in_features,device=m.device)*.01)
        self.b=torch.nn.Parameter(torch.zeros(base.out_features,cfg['rank'],device=m.device))
    def forward(self,x):
        y=self.base(x)
        return y+((x.float()@self.a.T)@self.b.T).to(y.dtype) if self.enabled else y
mods={};layers=m.encoder.language_model.layers
for i in range(len(layers)-cfg['last_layers'],len(layers)):
    layer=layers[i]
    for name,base in list(layer.named_modules()):
        # Keep recurrent kernels untouched; adapt MLP in all four layers and full attention projections.
        if isinstance(base,torch.nn.Linear) and (name.startswith('mlp.') or name.startswith('self_attn.')):
            parent=layer;bits=name.split('.')
            for b in bits[:-1]:parent=getattr(parent,b)
            mod=LowRank(base);setattr(parent,bits[-1],mod);mods[f'{i}.{name}']=mod
params=[p for mod in mods.values() for p in [mod.a,mod.b]];assert len(mods)>=12
opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01)
if cfg.get('init_patch'):
    saved=load_file(cfg['init_patch']);expected=patch_keys={f'{n}.{k}' for n in mods for k in ['a','b']}
    assert set(saved)==expected
    with torch.no_grad():
        for n,mod in mods.items():
            for k in ['a','b']:getattr(mod,k).copy_(saved[f'{n}.{k}'])
    state=torch.load(cfg['init_optimizer'],map_location=m.device,weights_only=False)
    assert state['step']==cfg['data_offset_steps'] and state['config']['seed']==cfg['seed']
    assert all(state['config'][k]==cfg[k] for k in ['lr','rank','last_layers','batch_size','pool'])
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
identity=dict(snapshot=root,revision=Path(root).name,modules=list(mods),trainable_parameters=sum(p.numel() for p in params),script_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('d3_image_processor.py')),train_sha256=sha(out/'train.inputs.jsonl'),pool_manifest_sha256=sha(pool/'MANIFEST.json'),config=cfg,base_precision=str(next(m.backbone.parameters()).dtype),temperature=m.temperature)
(out/'IDENTITY.json').write_text(json.dumps(identity,indent=2));print('identity',json.dumps(identity),flush=True)
def check():
    if time.monotonic()-start>cfg['budget_seconds']:raise TimeoutError('bounded science adaptation')
def prepare(s):
    check();q=s['questions']['q'];options=list(q['criteria']) if q['type']=='choice' else ['false','true']
    with Image.open(Path('data')/s['image']) as im:image=im.convert('RGB')
    x=m.processor(text=[m.image_text(s.get('state_text',s.get('state','')),q,1)],images=[image],padding=True,return_tensors='pt')
    vocab=m.backbone.language_model.embed_tokens.num_embeddings
    assert 0<=int(x['input_ids'].min()) and int(x['input_ids'].max())<vocab
    x={k:v.to(m.device) for k,v in x.items()}
    with torch.no_grad(),d3_runtime.sdpa_backends(m.device):e,r=m.image_embeddings(x)
    return options,e.detach(),r,x['attention_mask']
def logits(e,r,mask,n):
    with d3_runtime.sdpa_backends(m.device):h=m.encoder(e,mask,r)[:,-1].float()
    return (h@m.readout[:n].T)[0]
def patch():return {f'{n}.{k}':getattr(mod,k).detach().cpu().contiguous() for n,mod in mods.items() for k in ['a','b']}
def save(step):
    save_file(patch(),str(out/f'adapter-step{step}.safetensors'))
    torch.save(dict(step=step,optimizer=opt.state_dict(),config=cfg,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'optimizer-step{step}.pt')
held=read('runs/exp99-d3-scienceqa/inputs.jsonl')
held+=read('data/v05/exp81/chart_dev.jsonl')[:128]+read('data/v05/exp85/ui_dev.jsonl')[:128]
held += [s for s in read('data/v05/exp81/protect.jsonl') if s['source'] in ['coco-size','coco-position']]
assert len(held)==640
assert not {s['image'] for s in held}&{s['image'] for s in train}
(out/'validation.inputs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in held))
def evaluate(label):
    results=[]
    with (out/f'{label}.details.jsonl').open('x') as f,torch.no_grad():
        for i,s in enumerate(held):
            options,e,r,mask=prepare(s);p=(logits(e,r,mask,len(options))/m.temperature).softmax(-1).cpu().tolist()
            assert all(math.isfinite(v) for v in p) and abs(sum(p)-1)<1e-5
            gold=s['gold']['q']['probabilities'];target=max(range(len(options)),key=lambda j:gold[options[j]])
            rec=dict(sample_id=s['id'],image_id=s['image_id'],image=s['image'],source=s['source'],options=options,gold=gold,probabilities=dict(zip(options,p)),correct=max(range(len(p)),key=p.__getitem__)==target,nll=-math.log(max(p[target],1e-30)))
            f.write(json.dumps(rec)+'\n');f.flush();results.append(rec)
            if i==0:
                native=m.system_one(state=s.get('state_text',s.get('state','')),questions=s['questions'],images=[str(Path('data')/s['image'])])['answers']['q']['probabilities']
                err=max(abs(native[o]-p[j]) for j,o in enumerate(options));assert err<1e-5,err
            if (i+1)%64==0:print(label,i+1,'/',len(held),flush=True)
    return results
before=evaluate('before');save(0);opt.zero_grad();loss_sum=0
with (out/'train.jsonl').open('x') as log:
    for i,s in enumerate(train):
        options,e,r,mask=prepare(s);gold=torch.tensor([s['gold']['q']['probabilities'][o] for o in options],device=m.device)
        loss=-(gold*logits(e,r,mask,len(options)).log_softmax(-1)).sum();assert torch.isfinite(loss) and 0<=float(loss.detach())<20
        if float(loss.detach())==0:assert int(logits(e,r,mask,len(options)).detach().argmax())==int(gold.argmax())
        (loss/8).backward();loss_sum+=float(loss.detach())/8
        if (i+1)%8==0:
            step=(i+1)//8;grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            lr=cfg['lr']*min(1,(step+cfg.get('data_offset_steps',0))/cfg['warmup'])
            for g in opt.param_groups:g['lr']=lr
            opt.step();opt.zero_grad();rec=dict(step=step,ce=loss_sum,gradient_norm=grad,lr=lr,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated());loss_sum=0
            log.write(json.dumps(rec)+'\n');log.flush();print('train',json.dumps(rec),flush=True)
            if step%cfg['save_every']==0:save(step)
saved=load_file(str(out/f'adapter-step{cfg["steps"]}.safetensors'))
assert all(torch.equal(v,saved[k]) for k,v in patch().items())
# Reload persisted tensors before evaluating the exported candidate.
with torch.no_grad():
    for name,mod in mods.items():
        for k in ['a','b']:getattr(mod,k).copy_(saved[f'{name}.{k}'])
after=evaluate('after');summary={}
for source in sorted({r['source'] for r in before}):
    b=[r for r in before if r['source']==source];a=[r for r in after if r['source']==source]
    summary[source]=dict(n=len(a),before=sum(r['correct'] for r in b)/len(b),after=sum(r['correct'] for r in a)/len(a),before_nll=sum(r['nll'] for r in b)/len(b),after_nll=sum(r['nll'] for r in a)/len(a))
# Paired intervals by original image, preserving all fixed monitoring questions.
from devision.train.compare import paired_metric
paired={}
for source in summary:
    paired[source]=paired_metric([dict(r,qid='q') for r in before if r['source']==source],[dict(r,qid='q') for r in after if r['source']==source],metric='accuracy',resamples=5000,seed=110)
# Controls on the same 256 science questions, never fed into training.
with (out/'science.controls.jsonl').open('x') as f,torch.no_grad():
    for control in ['noimage','reversed']:
        for s in held[:256]:
            check();q=dict(s['questions']['q'])
            if control=='reversed':q['criteria']=dict(reversed(list(q['criteria'].items())))
            args=dict(state=s.get('state_text',s.get('state','')),questions={'q':q})
            if control=='reversed':args['images']=[str(Path('data')/s['image'])]
            p=m.system_one(**args)['answers']['q']['probabilities']
            gold=s['gold']['q']['probabilities'];assert set(p)==set(gold) and all(math.isfinite(v) for v in p.values()) and abs(sum(p.values())-1)<1e-5
            rec=dict(sample_id=s['id'],image_id=s['image_id'],source=s['source'],gold=gold,probabilities=p,correct=max(p,key=p.get)==max(gold,key=gold.get),nll=-math.log(max(p[max(gold,key=gold.get)],1e-30)),control=control)
            f.write(json.dumps(rec)+'\n');f.flush()
        print('control',control,'complete',flush=True)
report=dict(paired=paired,summary=summary,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),limitations='monitoring only; no calibration/official acceptance; adapter depends on pinned base; paired confidence intervals and visual controls required')
(out/'report.json').write_text(json.dumps(report,indent=2));print('report',json.dumps(report),flush=True)
