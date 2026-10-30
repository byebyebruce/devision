"""Reload the persisted exp110 adapter for remaining controls; never rerun training."""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com');os.environ['D3_FAST']='0'
import json,sys,time,math
from pathlib import Path
import torch
from safetensors.torch import load_file
from d3_image_processor import load_processor
out=Path('runs/exp110-d3-science/train');identity=json.loads((out/'IDENTITY.json').read_text());root=identity['snapshot'];sys.path.insert(0,root)
import d3_runtime
d3_runtime.load_processor=load_processor
torch.set_num_threads(4);start=time.monotonic()
m=d3_runtime.D3.from_pretrained(root,device='cuda:0',batch_size=1,local_files_only=True)
patch=load_file(str(out/'adapter-step256.safetensors'))
class Restored(torch.nn.Module):
    def __init__(self,base,name):
        super().__init__();self.base=base;self.a=torch.nn.Parameter(patch[f'{name}.a'].to(m.device),requires_grad=False);self.b=torch.nn.Parameter(patch[f'{name}.b'].to(m.device),requires_grad=False)
    def forward(self,x):
        y=self.base(x);return y+((x.float()@self.a.T)@self.b.T).to(y.dtype)
for name in identity['modules']:
    bits=name.split('.');parent=m.encoder.language_model.layers[int(bits[0])]
    for b in bits[1:-1]:parent=getattr(parent,b)
    setattr(parent,bits[-1],Restored(getattr(parent,bits[-1]),name))
held=[json.loads(x) for x in (out/'validation.inputs.jsonl').read_text().splitlines()][:256]
after=[json.loads(x) for x in (out/'after.details.jsonl').read_text().splitlines()];assert len(after)==640
with (out/'science.controls.jsonl').open('x') as f,torch.no_grad():
    for control in ['real_reload','noimage','reversed']:
        for i,s in enumerate(held):
            if time.monotonic()-start>600:raise TimeoutError('control budget')
            q=dict(s['questions']['q'])
            if control=='reversed':q['criteria']=dict(reversed(list(q['criteria'].items())))
            args=dict(state=s.get('state_text',s.get('state','')),questions={'q':q})
            if control!='noimage':args['images']=[str(Path('data')/s['image'])]
            p=m.system_one(**args)['answers']['q']['probabilities'];gold=s['gold']['q']['probabilities']
            assert set(p)==set(gold) and all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
            if control=='real_reload':assert max(abs(p[k]-after[i]['probabilities'][k]) for k in p)<1e-5
            rec=dict(sample_id=s['id'],qid='q',image_id=s['image_id'],source=s['source'],gold=gold,probabilities=p,correct=max(p,key=p.get)==max(gold,key=gold.get),nll=-math.log(max(p[max(gold,key=gold.get)],1e-30)),control=control)
            f.write(json.dumps(rec)+'\n');f.flush()
            if (i+1)%64==0:print(control,i+1,flush=True)
print('CONTROLS_COMPLETE',time.monotonic()-start,flush=True)
