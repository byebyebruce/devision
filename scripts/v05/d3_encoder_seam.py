"""Bounded last-layer LoRA engineering probe, not a release model."""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ['D3_FAST']='0'
import json, sys, time
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import save_file,load_file
from d3_image_processor import load_processor

out=Path('runs/exp95-d3-encoder-seam'); out.mkdir(parents=True,exist_ok=True)
(out/'STARTED').open('x').close()
t0=time.monotonic(); torch.manual_seed(95)
root=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
assert Path(root).name=='03560c26f5199c3c04129a2ccb38960c913635e0'
sys.path.insert(0,root)
import d3_runtime

d3_runtime.load_processor=load_processor
m=d3_runtime.D3.from_pretrained(root,device='cuda:0',batch_size=1,local_files_only=True)
layer=m.encoder.language_model.layers[-1]
class LowRank(torch.nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base
        self.a=torch.nn.Parameter(torch.randn(4,base.in_features,device=m.device)*0.01)
        self.b=torch.nn.Parameter(torch.zeros(base.out_features,4,device=m.device))
    def forward(self,x):
        return self.base(x)+((x.float()@self.a.T)@self.b.T).to(x.dtype)

rows=[]; seen=set()
for line in Path('runs/exp93-d3-cache-audit-64/inputs.jsonl').read_text().splitlines():
    s=json.loads(line)
    if s['source'] not in seen: rows.append(s); seen.add(s['source'])
assert len(rows)==4
inputs=[]
for s in rows:
    q=s['questions']['q']; opts=list(q['criteria']) if q['type']=='choice' else ['false','true']
    with Image.open(Path('data')/s['image']) as im: image=im.convert('RGB')
    x=m.processor(text=[m.image_text(s.get('state',''),q,1)],images=[image],padding=True,return_tensors='pt')
    x={k:v.to(m.device) for k,v in x.items()}
    with torch.no_grad(),d3_runtime.sdpa_backends(m.device):
        e,r=m.image_embeddings(x); h=m.encoder(e,x['attention_mask'],r)[:,-1].float()
    inputs.append((e.detach(),r,x['attention_mask'],torch.tensor([s['gold']['q']['probabilities'][o] for o in opts],device=m.device),h.detach()))
mods={}
for name in ['q_proj','v_proj']:
    base=getattr(layer.self_attn,name)
    assert isinstance(base,torch.nn.Linear)
    mods[name]=LowRank(base); setattr(layer.self_attn,name,mods[name])
params=[p for mod in mods.values() for p in [mod.a,mod.b]]
opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
def losses(backward=False):
    total=0.; error=0.
    for e,r,mask,g,old in inputs:
        assert time.monotonic()-t0<300
        with d3_runtime.sdpa_backends(m.device): h=m.encoder(e,mask,r)[:,-1].float()
        logits=h@m.readout[:len(g)].T
        value=-(g*logits.log_softmax(-1)[0]).sum()/len(inputs)
        error=max(error,(((logits/m.temperature).softmax(-1)-(old@m.readout[:len(g)].T/m.temperature).softmax(-1)).abs().max().item()))
        assert torch.isfinite(value)
        if backward: value.backward()
        total+=value.item()
    return total,error
before,error=losses(True); assert error<1e-5,error
grad=sum(p.grad.float().square().sum().item() for p in params if p.grad is not None)**0.5
assert grad>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
opt.step()
with torch.no_grad(): after,_=losses()
assert after<before,(before,after)
patch={f'{n}.{k}':getattr(mod,k).detach().cpu().contiguous() for n,mod in mods.items() for k in ['a','b']}
save_file(patch,str(out/'diagnostic-lora.safetensors'))
restored=load_file(str(out/'diagnostic-lora.safetensors'))
assert all(torch.equal(v,restored[k]) for k,v in patch.items())
report=dict(revision=Path(root).name,layer_index=len(m.encoder.language_model.layers)-1,targets=list(mods),rank=4,trainable_parameters=sum(p.numel() for p in params),sources=list(seen),ce_before=before,ce_after=after,gradient_norm=grad,zero_patch_probability_error=error,seconds=time.monotonic()-t0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),decision='engineering_seam_pass_only',limitations='four training images; no generalization, calibration, full export or CPU acceptance')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
