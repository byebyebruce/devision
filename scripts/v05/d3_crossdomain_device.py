"""Bounded batch-one local system_one device comparison; not official admission."""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('D3_FAST', '0')
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import torch
from PIL import Image
from d3_image_processor import load_processor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',required=True);a=ap.parse_args()
    work=Path('runs/exp105-d3-crossdomain-device')/a.device.replace(':','-');work.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    path=json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name=='03560c26f5199c3c04129a2ccb38960c913635e0'
    sys.path.insert(0,path)
    import d3_runtime
    d3_runtime.load_processor=load_processor
    tick=time.perf_counter();model=d3_runtime.D3.from_pretrained(path,device=a.device,batch_size=1,local_files_only=True)
    model.backbone.to(dtype=torch.float32)
    load_s=time.perf_counter()-tick
    started=time.monotonic()
    assert model.image_unavailable is None,model.image_unavailable
    sources=[('chart','data/v05/exp81/chart_dev.jsonl','choice'),('document','data/v05/funsd/derived/dev.jsonl','choice'),('photo','data/v05/external/realworld_native.jsonl','choice'),('noul','data/v05/external/realworld_native.jsonl','noul')]
    samples=[];source_hashes={}
    for domain,filename,kind in sources:
        source=Path(filename);source_hashes[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
        pool=[x for x in map(json.loads,source.read_text().splitlines()) if x['questions']['q']['type']==kind]
        selected=sorted(pool,key=lambda x:hashlib.sha256(str(x['id']).encode()).hexdigest())[:3]
        assert len(selected)==3
        for x in selected:x['_domain']=domain
        samples.extend(selected)
    assert len(samples)==12
    identity=dict(snapshot=path,device=a.device,threads=torch.get_num_threads(),cpu=platform.processor(),torch=torch.__version__,runtime=model.runtime_info(),image_contract=model.image_contract(),source_sha256=source_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest(),load_s=load_s)
    (work/'IDENTITY.json').write_text(json.dumps(identity,indent=2,default=str)+'\n')
    def sync():
        if a.device.startswith('cuda'):torch.cuda.synchronize()
    rows=[]
    with (work/'details.jsonl').open('x') as out:
        for i,s in enumerate(samples+samples):
            assert time.monotonic()-started<800, 'device wall budget exceeded'
            image=str(Path('data')/s['image']);sync();tick=time.perf_counter()
            result=model.system_one(state=s.get('state',''),questions=s['questions'],images=[image]);sync();ms=(time.perf_counter()-tick)*1000
            answer=result['answers']['q'];probs=answer['probabilities'] if s['questions']['q']['type']=='choice' else {'false':1-answer['noul'],'true':answer['noul']};criteria=s['gold']['q']['probabilities']
            assert set(probs)==set(criteria) and all(math.isfinite(p) and 0<=p<=1 for p in probs.values()) and abs(sum(probs.values())-1)<1e-5
            with Image.open(image) as im:size=im.size
            row=dict(sample_id=s['id'],domain=s['_domain'],phase='measured',repeat=i//12,latency_ms=ms,probabilities=probs,prediction=max(probs,key=probs.get),image_size=size,usage=result.get('usage'))
            out.write(json.dumps(row)+'\n');out.flush();rows.append(row);print(a.device,i,round(ms,2),flush=True)
    times=torch.tensor([r['latency_ms'] for r in rows],dtype=torch.float64)
    report=dict(n=24,unique_n=12,precision="fp32 converted from loaded bf16",load_s=load_s,cold_ms=rows[0]['latency_ms'],p50_ms=times.quantile(.5).item(),p95_ms=times.quantile(.95).item(),decisions_per_s=1000/times.mean().item(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,scope='local system_one batch1, no warmups; no HTTP; chart/document/photo/noul, two repeated passes, FP32; not official admission')
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
