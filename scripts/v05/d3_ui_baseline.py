"""Pinned external native-decision baseline; no production integration or training."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('D3_FAST', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from huggingface_hub import snapshot_download
WORK = Path('runs/exp87-d3-ui-pil')
REV = '03560c26f5199c3c04129a2ccb38960c913635e0'

def main():
    (WORK / 'STARTED').open('x').close()
    started = time.monotonic()
    print('Downloading pinned D3 weights', REV, flush=True)
    path = snapshot_download('vllm-sr/d3-lite', revision=REV,
        allow_patterns=['*.json', '*.py', '*.txt', '*.jinja', '*.safetensors', 'README.md', 'LICENSE*'], max_workers=2)
    sys.path.insert(0, path)
    import d3_runtime
    from d3_image_processor import load_processor
    d3_runtime.load_processor = load_processor
    from d3_runtime import D3
    print('Loading native runtime', path, flush=True)
    model = D3.from_pretrained(path, device='cuda:0', batch_size=1, local_files_only=True)
    assert model.image_unavailable is None, model.image_unavailable
    identity = dict(repo='vllm-sr/d3-lite', revision=REV, snapshot=path,
        runtime=model.runtime_info(), image_contract=model.image_contract(),
        decision_config=model.config, adapter_sha256=hashlib.sha256(Path(__file__).with_name('d3_image_processor.py').read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (WORK/'IDENTITY.json').write_text(json.dumps(identity, indent=2, default=str)+'\n')
    for control in ['real', 'mismatched', 'reversed']:
        source=Path('data/v05/exp85')/('ui_dev_rotated.jsonl' if control=='mismatched' else 'ui_dev.jsonl')
        samples=[json.loads(x) for x in source.read_text().splitlines()]
        rows=[]
        with (WORK/f'ui.{control}.details.jsonl').open('x') as out:
            for i,s in enumerate(samples):
                if time.monotonic()-started>1800:
                    raise TimeoutError('exp87 wall budget')
                q=dict(s['questions']['q'])
                if control=='reversed':
                    q['criteria']=dict(reversed(list(q['criteria'].items())))
                torch.cuda.synchronize(); tick=time.perf_counter()
                result=model.system_one(state=s.get('state',''), questions={'q':q}, images=[str(Path('data')/s['image'])])
                torch.cuda.synchronize(); elapsed=1000*(time.perf_counter()-tick)
                answer=result['answers']['q']; probs=answer['probabilities']
                assert set(probs)==set(q['criteria'])
                assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5
                gold=s['gold']['q']['probabilities']; target=max(gold,key=gold.get)
                prediction=max(probs,key=probs.get)
                row=dict(sample_id=s['id'],qid='q',image_id=s['image_id'],image_used=s['image'],
                    source=s['source'],type='choice',options=list(q['criteria']),gold=gold,gold_answer=target,
                    probabilities=probs,prediction=prediction,correct=prediction==target,
                    nll=-math.log(max(probs[target],1e-30)),brier=sum((probs[o]-gold[o])**2 for o in probs),
                    control=control,latency_ms=elapsed,usage=result['usage'])
                out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
                if i%16==0: print(control,i+1,'accuracy',sum(x['correct'] for x in rows)/len(rows),flush=True)
        summary=dict(n=len(rows),accuracy=sum(x['correct'] for x in rows)/len(rows),
            nll=sum(x['nll'] for x in rows)/len(rows),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            timing_scope='native system_one end-to-end; includes cold call; not formal deployment benchmark')
        (WORK/f'ui.{control}.json').write_text(json.dumps(summary,indent=2)+'\n')
    (WORK/'STATUS.json').write_text(json.dumps(dict(state='completed'))+'\n')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        (WORK/'STATUS.json').write_text(json.dumps(dict(state='execution_failed',error=repr(exc)))+'\n')
        raise
