"""Fixed full real-photo monitoring panel on FP32 deployment path; no training."""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('D3_FAST','0')
import hashlib,json,math,sys,time
from pathlib import Path
import torch
from d3_image_processor import load_processor
W=Path('runs/exp104-d3-fp32-realworld-retry1')
def main():
 start=time.monotonic();torch.set_num_threads(4)
 source=Path('data/v05/external/realworld_native.jsonl')
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
 assert digest==json.loads(Path('runs/exp88-d3-panels-local/realworld.real.json').read_text())['source_sha256']
 samples=[json.loads(x) for x in source.read_text().splitlines()];assert len(samples)==587
 root=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
 assert Path(root).name=='03560c26f5199c3c04129a2ccb38960c913635e0'
 sys.path.insert(0,root);import d3_runtime as rt
 rt.load_processor=load_processor
 m=rt.D3.from_pretrained(root,device='cuda:0',batch_size=1,local_files_only=True);m.backbone.to(dtype=torch.float32)
 assert m.image_unavailable is None
 (W/'IDENTITY.json').write_text(json.dumps(dict(revision=Path(root).name,source_sha256=digest,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest(),runtime=m.runtime_info(),precision='FP32 converted from BF16',role='monitoring'),indent=2,default=str)+'\n')
 rows=[]
 with (W/'details.jsonl').open('x') as out:
  for i,s in enumerate(samples):
   assert time.monotonic()-start<540,'wall budget exceeded'
   answer=m.system_one(state=s.get('state',''),questions=s['questions'],images=[str(Path('data')/s['image'])])['answers']['q']
   p=answer['probabilities'] if s['questions']['q']['type']=='choice' else {'false':1-answer['noul'],'true':answer['noul']};g=s['gold']['q']['probabilities']
   assert set(p)==set(g) and all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
   row=dict(sample_id=s['id'],image_id=s['image_id'],gold=g,probabilities=p,prediction=max(p,key=p.get),correct=max(p,key=p.get)==max(g,key=g.get),nll=-sum(v*math.log(max(p[k],1e-30)) for k,v in g.items()))
   out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
   if i%32==0:print(i+1,len(samples),flush=True)
 report=dict(n=len(rows),accuracy=sum(r['correct'] for r in rows)/len(rows),nll=sum(r['nll'] for r in rows)/len(rows),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
 (W/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
