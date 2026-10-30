"""exp102 fixed-input numerical diagnostic; no training or production changes."""
import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('D3_FAST','0')
import hashlib,json,sys,time,gc
from pathlib import Path
import torch
from PIL import Image
from d3_image_processor import load_processor
W=Path('runs/exp102-d3-device-numerics')
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
    start=time.monotonic();torch.set_num_threads(4)
    cpu={r['sample_id']:r for r in read('runs/exp101-d3-deployment-cost/cpu/details.jsonl') if r['phase']=='measured'}
    cuda={r['sample_id']:r for r in read('runs/exp101-d3-deployment-cost/cuda-0/details.jsonl') if r['phase']=='measured'}
    delta=lambda a,b:max(abs(a[k]-b[k]) for k in a)
    ids=sorted(cpu,key=lambda i:delta(cpu[i]['probabilities'],cuda[i]['probabilities']),reverse=True)[:4]
    assert delta(cpu[ids[0]]['probabilities'],cuda[ids[0]]['probabilities'])>.005
    samples={r['id']:r for r in read('data/v05/exp85/ui_dev.jsonl')}
    root=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot'];sys.path.insert(0,root)
    import d3_runtime as rt
    rt.load_processor=load_processor
    inputs=[];rows=[];native_errors=[]
    with (W/'details.jsonl').open('x') as out:
        for dev in ['cpu','cuda:0']:
            m=rt.D3.from_pretrained(root,device=dev,batch_size=1,local_files_only=True)
            if dev=='cpu':
                for sid in ids:
                    s=samples[sid];im=Image.open(Path('data')/s['image']).convert('RGB')
                    x=m.processor(text=[m.image_text(s.get('state',''),s['questions']['q'],1)],images=[im],padding=True,return_tensors='pt')
                    inputs.append(dict(x))
                torch.save(inputs,W/'inputs.pt')
            else: inputs=torch.load(W/'inputs.pt',weights_only=True)
            for precision,backend in [('bf16','default'),('fp32','default')]+([('bf16','math')] if dev!='cpu' else []):
                m.backbone.to(dtype=torch.float32 if precision=='fp32' else torch.bfloat16)
                for sid,x in zip(ids,inputs):
                    if time.monotonic()-start>900:raise TimeoutError('exp102 budget')
                    s=samples[sid];opts=list(s['questions']['q']['criteria']);y={k:v.to(dev) for k,v in x.items()}
                    context=rt.sdpa_backends(m.device) if backend=='default' else torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                    tick=time.monotonic()
                    with torch.inference_mode(),context:
                        e,r=m.image_embeddings(y);h=m.encoder(e,y['attention_mask'],r)[:,-1].float()
                        p=((h@m.readout[:len(opts)].T)/m.temperature).softmax(-1)[0].cpu().tolist()
                    probs=dict(zip(opts,p));assert all(torch.isfinite(torch.tensor(p))) and abs(sum(p)-1)<1e-5
                    row=dict(sample_id=sid,device=dev,precision=precision,backend=backend,probabilities=probs,seconds=time.monotonic()-tick)
                    if precision=='bf16' and backend=='default':
                        native=m.system_one(state=s.get('state',''),questions=s['questions'],images=[str(Path('data')/s['image'])])['answers']['q']['probabilities']
                        err=delta(native,probs);assert err<1e-5,err
                        row['native_error']=err;row['exp101_error']=delta(probs,(cpu if dev=='cpu' else cuda)[sid]['probabilities']);native_errors.append(err)
                    out.write(json.dumps(row)+'\n');out.flush();rows.append(row);print(dev,precision,backend,sid,probs,flush=True)
            del m;gc.collect()
            if dev!='cpu':torch.cuda.empty_cache()
    by={(r['device'],r['precision'],r['backend'],r['sample_id']):r['probabilities'] for r in rows}
    comparisons={}
    for precision,backend in [('bf16','default'),('fp32','default'),('bf16','math')]:
        comparisons[precision+'_'+backend]=max(delta(by[('cpu',precision,'default',sid)],by[('cuda:0',precision,backend,sid)]) for sid in ids)
    report=dict(n=4,selected_ids=ids,comparisons=comparisons,native_max_error=max(native_errors),max_exp101_reproduction_error=max(r.get('exp101_error',0) for r in rows),seconds=time.monotonic()-start,revision=Path(root).name,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=hashlib.sha256((W/'inputs.pt').read_bytes()).hexdigest(),scope='selected worst four UI cases; diagnostic only; no calibration or release acceptance')
    (W/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
