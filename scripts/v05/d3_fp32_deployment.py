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
    work=Path('runs/exp103-d3-fp32-deployment')/a.device.replace(':','-');work.mkdir(parents=True,exist_ok=False)
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
    source=Path('data/v05/exp85/ui_dev.jsonl');samples=list(map(json.loads,source.read_text().splitlines()))[:32]
    identity=dict(snapshot=path,device=a.device,threads=torch.get_num_threads(),cpu=platform.processor(),torch=torch.__version__,runtime=model.runtime_info(),image_contract=model.image_contract(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest(),load_s=load_s)
    (work/'IDENTITY.json').write_text(json.dumps(identity,indent=2,default=str)+'\n')
    def sync():
        if a.device.startswith('cuda'):torch.cuda.synchronize()
    rows=[]
    with (work/'details.jsonl').open('x') as out:
        for i,s in enumerate(samples[:3]+samples+samples):
            assert time.monotonic()-started<800, 'device wall budget exceeded'
            image=str(Path('data')/s['image']);sync();tick=time.perf_counter()
            result=model.system_one(state=s.get('state',''),questions=s['questions'],images=[image]);sync();ms=(time.perf_counter()-tick)*1000
            probs=result['answers']['q']['probabilities'];criteria=s['questions']['q']['criteria']
            assert set(probs)==set(criteria) and all(math.isfinite(p) and 0<=p<=1 for p in probs.values()) and abs(sum(probs.values())-1)<1e-5
            with Image.open(image) as im:size=im.size
            row=dict(sample_id=s['id'],phase='warmup' if i<3 else 'measured',repeat=None if i<3 else (i-3)//32,latency_ms=ms,probabilities=probs,prediction=max(probs,key=probs.get),image_size=size,usage=result.get('usage'))
            out.write(json.dumps(row)+'\n');out.flush();rows.append(row);print(a.device,i,round(ms,2),flush=True)
    times=torch.tensor([r['latency_ms'] for r in rows[3:]],dtype=torch.float64)
    report=dict(n=64,unique_n=32,precision="fp32 converted from loaded bf16",load_s=load_s,cold_ms=rows[0]['latency_ms'],p50_ms=times.quantile(.5).item(),p95_ms=times.quantile(.95).item(),decisions_per_s=1000/times.mean().item(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,scope='local system_one batch1, 3 warmups; no HTTP; UI only, two repeated passes, FP32; not official admission')
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
