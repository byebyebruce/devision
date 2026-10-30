"""One-update engineering probe; not a model-quality training run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('D3_FAST', '0')
import torch
from safetensors.torch import load_file, save_file
from PIL import Image
from d3_image_processor import load_processor

WORK = Path('runs/exp92-d3-training-interface')
REV = '03560c26f5199c3c04129a2ccb38960c913635e0'

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/'STARTED').open('x').close()
    started = time.monotonic()
    held = []
    for name in ['data/v05/exp81/protect.jsonl', 'data/v05/exp91/independent.jsonl']:
        held.extend(json.loads(x) for x in Path(name).read_text().splitlines())
    forbidden = {s[k] for s in held for k in ['image', 'image_id']}
    rows = []; counts = {'coco-size': 0, 'coco-position': 0}; seen = set()
    source = Path('data/v05/exp81/train.jsonl')
    for line in source.read_text().splitlines():
        s = json.loads(line); kind = s['source']
        if kind not in counts or counts[kind] >= 8 or s.get('flipped'):
            continue
        if s['image'] in forbidden or s['image_id'] in forbidden or s['image'] in seen:
            continue
        if s['questions']['q']['type'] != 'choice':
            continue
        rows.append(s); seen.add(s['image']); counts[kind] += 1
        if len(rows) == 16: break
    assert len(rows) == 16, counts
    (WORK/'inputs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
    path = json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name == REV
    sys.path.insert(0, path)
    import d3_runtime
    d3_runtime.load_processor = load_processor
    m = d3_runtime.D3.from_pretrained(path, device='cuda:0', batch_size=1, local_files_only=True)
    assert m.fast is None and m.encoder is not None
    assert m.image_unavailable is None
    torch.manual_seed(92)
    features = []; targets = []; native = []; max_error = 0.
    for i, s in enumerate(rows):
        if time.monotonic()-started > 540: raise TimeoutError('probe budget')
        q = s['questions']['q']; options = list(q['criteria'])
        assert len(options) == 2
        image = Image.open(Path('data')/s['image']).convert('RGB')
        answer = m.system_one(state=s.get('state',''), questions={'q':q}, images=[image])['answers']['q']['probabilities']
        text = m.image_text(s.get('state',''), q, 1)
        x = m.processor(text=[text], images=[image], padding=True, return_tensors='pt')
        x = {k:v.to(m.device) for k,v in x.items()}
        with torch.no_grad(), d3_runtime.sdpa_backends(m.device):
            embeds, rope = m.image_embeddings(x)
            h = m.encoder(embeds, x['attention_mask'], rope)[:, -1].float()
            p = ((h @ m.readout[:2].T)/m.temperature).softmax(-1)[0]
        expected = torch.tensor([answer[o] for o in options], device=m.device)
        error = (p-expected).abs().max().item(); max_error = max(max_error,error)
        assert error < 1e-5, error
        features.append(h); native.append(expected)
        gold = s['gold']['q']['probabilities']
        targets.append([gold[o] for o in options])
        print('feature', i+1, 'native_probability_error', error, flush=True)
    h = torch.cat(features); gold = torch.tensor(targets, device=m.device)
    w = torch.nn.Parameter(m.readout.detach().clone())
    optimizer = torch.optim.AdamW([w], lr=1e-5, weight_decay=0)
    def loss(): return -(gold * (h @ w[:2].T).log_softmax(-1)).sum(-1).mean()
    before = loss(); before.backward()
    grad = w.grad.norm().item()
    assert torch.isfinite(w.grad).all() and grad > 0
    optimizer.step(); after = loss().item()
    assert after < before.item(), (before.item(),after)
    save_file({'weight':w.detach().cpu().contiguous()}, str(WORK/'diagnostic-readout.safetensors'))
    restored = load_file(str(WORK/'diagnostic-readout.safetensors'))['weight'].to(m.device)
    assert torch.equal(w.detach(),restored)
    m.readout = restored
    s = rows[0]; q = s['questions']['q']; options = list(q['criteria'])
    answer = m.system_one(state=s.get('state',''),questions={'q':q},images=[str(Path('data')/s['image'])])['answers']['q']['probabilities']
    expected = ((h[:1] @ w[:2].T)/m.temperature).softmax(-1)[0].detach()
    export_error = max(abs(answer[o]-expected[j].item()) for j,o in enumerate(options))
    assert export_error < 1e-5, export_error
    report = dict(revision=REV, n=len(rows), counts=counts, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=hashlib.sha256((WORK/'inputs.jsonl').read_bytes()).hexdigest(), trainable_parameters=w.numel(), backbone_trainable_parameters=sum(p.numel() for p in m.backbone.parameters() if p.requires_grad), temperature=m.temperature, ce_before=before.item(), ce_after=after, gradient_norm=grad, native_max_probability_error=max_error, export_max_probability_error=export_error, seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(), decision='interface_pass_only', limitations='16 training examples; no generalization, full export, backbone backward or calibration validation')
    (WORK/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

def pilot(config_path):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    (out/'STARTED').open('x').close()
    (out/'config.yaml').write_text(Path(config_path).read_text())
    start = time.monotonic(); torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    cache = Path(cfg['cache']); report = json.loads((cache/'report.json').read_text())
    assert hashlib.sha256((cache/'features.safetensors').read_bytes()).hexdigest() == report['cache_sha256']
    tensors = load_file(str(cache/'features.safetensors'))
    meta = json.loads((cache/'metadata.json').read_text())
    h = tensors['features']; initial = tensors['initial_readout']; w = torch.nn.Parameter(initial.clone())
    assert len(h) == len(meta) == 256
    optimizer = torch.optim.AdamW([w], lr=cfg['lr'], weight_decay=0)
    def loss():
        terms = []
        for n in sorted({len(s['gold']) for s in meta}):
            ix = [i for i,s in enumerate(meta) if len(s['gold']) == n]
            gold = torch.tensor([meta[i]['gold'] for i in ix])
            terms.append(-(gold*(h[ix]@w[:n].T).log_softmax(-1)).sum())
        return sum(terms)/len(meta)
    baseline = loss().item(); history = []
    if cfg.get('trained_patch'):
        with torch.no_grad(): w.copy_(load_file(cfg['trained_patch'])['weight'])
    for step in range(1, (0 if cfg.get('trained_patch') else cfg['steps'])+1):
        if time.monotonic()-start > cfg['budget_seconds']: raise TimeoutError('pilot budget')
        optimizer.zero_grad(); value = loss()
        assert torch.isfinite(value) and value.item() <= 2*baseline
        value.backward(); grad = w.grad.norm().item()
        assert torch.isfinite(w.grad).all()
        optimizer.step()
        if step % cfg['save_every'] == 0:
            save_file({'weight':w.detach().contiguous()}, str(out/f'readout-step{step}.safetensors'))
            history.append(dict(step=step,ce=value.item(),gradient_norm=grad))
            print('train',history[-1],flush=True)
    path = json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(path).name == REV; sys.path.insert(0,path)
    import d3_runtime
    d3_runtime.load_processor = load_processor
    m = d3_runtime.D3.from_pretrained(path,device='cuda:0',batch_size=1,local_files_only=True)
    assert torch.equal(initial,m.readout.detach().float().cpu())
    rows = []
    protect = [json.loads(x) for x in Path('data/v05/exp81/protect.jsonl').read_text().splitlines()]
    for source in ['coco-size','coco-position']:
        selected = [s for s in protect if s['source']==source]
        assert len(selected)==64
        rows.extend(selected)
    for p in ['data/v05/exp81/chart_dev.jsonl','data/v05/exp85/ui_dev.jsonl']:
        rows.extend([json.loads(x) for x in Path(p).read_text().splitlines()][:128])
    assert not {s['image'] for s in rows} & {json.loads(x)['image'] for x in (cache/'inputs.jsonl').read_text().splitlines()}
    wi = initial.to(m.device); wt = w.detach().to(m.device); results = []
    with (out/'validation.details.jsonl').open('x') as f:
        for i,s in enumerate(rows):
            if time.monotonic()-start > cfg['budget_seconds']: raise TimeoutError('pilot budget')
            q = s['questions']['q']; options=list(q['criteria']) if q['type']=='choice' else ['false','true']; n=len(options)
            with Image.open(Path('data')/s['image']) as im: image=im.convert('RGB')
            x=m.processor(text=[m.image_text(s.get('state',''),q,1)],images=[image],padding=True,return_tensors='pt')
            x={k:v.to(m.device) for k,v in x.items()}
            with torch.no_grad(),d3_runtime.sdpa_backends(m.device):
                embeds,rope=m.image_embeddings(x)
                v=m.encoder(embeds,x['attention_mask'],rope)[:,-1].float()
                before=((v@wi[:n].T)/m.temperature).softmax(-1)[0].cpu().tolist()
                after=((v@wt[:n].T)/m.temperature).softmax(-1)[0].cpu().tolist()
            gold=[s['gold']['q']['probabilities'][o] for o in options]
            r=dict(id=s['id'],image=s['image'],source=s['source'],options=options,gold=gold,before=before,after=after)
            results.append(r); f.write(json.dumps(r)+'\n'); f.flush()
            if (i+1)%32==0: print('validation',i+1,'/',len(rows),flush=True)
    import random
    rng=random.Random(cfg['seed']); summary={}
    for source in sorted({r['source'] for r in results}):
        group=[r for r in results if r['source']==source]; clustered={}; b=[]; a=[]
        for r in group:
            target=max(range(len(r['gold'])),key=r['gold'].__getitem__)
            bc=int(max(range(len(r['before'])),key=r['before'].__getitem__)==target)
            ac=int(max(range(len(r['after'])),key=r['after'].__getitem__)==target)
            b.append(bc); a.append(ac); clustered.setdefault(r['image'],[]).append(ac-bc)
        clusters=list(clustered.values()); diffs=[]
        for _ in range(5000):
            sample=[rng.choice(clusters) for _ in clusters]
            diffs.append(sum(sum(x) for x in sample)/sum(len(x) for x in sample))
        diffs.sort(); summary[source]=dict(n=len(group),images=len(clusters),before=sum(b)/len(b),after=sum(a)/len(a),delta=(sum(a)-sum(b))/len(a),ci95=[diffs[125],diffs[4874]])
    passed=all(v['delta'] >= (0.03 if k in ['coco-size','coco-position'] else -0.02) for k,v in summary.items())
    result=dict(summary=summary,gate_pass=passed,train_history=history,ce_initial=baseline,ce_final=loss().item(),seconds=time.monotonic()-start,revision=REV,cache_sha256=report['cache_sha256'],config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='readout patch only; monitoring validation; no calibration or release qualification')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--config':
        pilot(sys.argv[2])
    else:
        main()
