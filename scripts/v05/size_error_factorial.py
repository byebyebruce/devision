"""Read-only four-arm diagnosis of the existing size monitoring slice."""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['D3_FAST'] = '0'
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from d3_image_processor import load_processor

OUT = Path('runs/exp97-size-error-factorial')
SUFFIX = ' Compare only the visible pixel area occupied by each named object in this image, excluding occluded parts. Do not compare their real-world sizes.'


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main():
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    paths = ['data/v05/exp81/protect.jsonl',
             'runs/exp90-d3-semantics/protect.semantic.details.jsonl',
             'runs/exp96-d3-lora/validation.details.jsonl',
             'runs/exp96-d3-lora/lora-step64.safetensors']
    identity = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths + [__file__]}
    (OUT/'IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')
    old = {r['id']: r for r in read(paths[2]) if r['source'] in ('coco-size', 'coco-position')}
    semantic = {r['sample_id']: r for r in read(paths[1])}
    assert old.keys() == semantic.keys() and len(old) == 128
    cpu = {}
    for source in ('coco-size', 'coco-position'):
        fixed = []; repaired = []; damaged = []; shared = []
        for key, r in old.items():
            if r['source'] != source:
                continue
            target = max(range(len(r['gold'])), key=r['gold'].__getitem__)
            native_ok = max(range(len(r['before'])), key=r['before'].__getitem__) == target
            lora_ok = max(range(len(r['after'])), key=r['after'].__getitem__) == target
            s = semantic[key]
            assert s['gold'] == dict(zip(r['options'], r['gold']))
            if not native_ok and s['correct']:
                fixed.append(key)
            if not native_ok and lora_ok:
                repaired.append(key)
            if native_ok and not lora_ok:
                damaged.append(key)
            if not s['correct'] and not lora_ok:
                shared.append(key)
        cpu[source] = dict(semantic_repairs=fixed, lora_repairs=repaired,
                           lora_damage=damaged, both_still_wrong=shared,
                           repair_overlap=sorted(set(fixed) & set(repaired)))
    (OUT/'error-intersections.json').write_text(json.dumps(cpu, indent=2)+'\n')
    print('CPU error intersections', json.dumps(cpu), flush=True)
    root = json.loads(Path('runs/exp87-d3-ui-pil/IDENTITY.json').read_text())['snapshot']
    assert Path(root).name == '03560c26f5199c3c04129a2ccb38960c913635e0'
    sys.path.insert(0, root)
    import d3_runtime
    d3_runtime.load_processor = load_processor
    m = d3_runtime.D3.from_pretrained(root, device='cuda:0', batch_size=1, local_files_only=True)
    patch = load_file(paths[3])

    class LowRank(torch.nn.Module):
        def __init__(self, base, name):
            super().__init__()
            self.base = base
            self.a = patch[name+'.a'].to(m.device)
            self.b = patch[name+'.b'].to(m.device)
            self.enabled = False

        def forward(self, x):
            if not self.enabled:
                return self.base(x)
            return self.base(x) + ((x.float() @ self.a.T) @ self.b.T).to(x.dtype)

    mods = []
    layer = m.encoder.language_model.layers[-1]
    for name in ('q_proj', 'v_proj'):
        mod = LowRank(getattr(layer.self_attn, name), name)
        setattr(layer.self_attn, name, mod)
        mods.append(mod)
    samples = [s for s in read(paths[0]) if s['source'] == 'coco-size']
    assert len(samples) == 64
    rows = []; max_error = 0.
    with (OUT/'combined.details.jsonl').open('x') as f, torch.no_grad():
        for i, s in enumerate(samples):
            if time.monotonic()-start > 300:
                raise TimeoutError('exp97 budget')
            q = dict(s['questions']['q']); q['instructions'] += SUFFIX
            ref = old[s['id']]; options = ref['options']
            assert list(q['criteria']) == options and s['gold']['q']['probabilities'] == dict(zip(options, ref['gold']))
            outputs = []
            for enabled in (False, True):
                for mod in mods:
                    mod.enabled = enabled
                answer = m.system_one(state=s.get('state', ''), questions={'q': q}, images=[str(Path('data')/s['image'])])['answers']['q']['probabilities']
                p = [answer[o] for o in options]
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-5
                outputs.append(p)
            err = max(abs(outputs[0][j]-semantic[s['id']]['probabilities'][o]) for j, o in enumerate(options))
            max_error = max(max_error, err); assert err < 1e-5, (s['id'], err)
            row = dict(ref, semantic=outputs[0], combined=outputs[1])
            rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
            if (i+1) % 16 == 0:
                print('validation', i+1, '/64', flush=True)
    def correct(r, arm):
        return int(max(range(len(r[arm])), key=r[arm].__getitem__) == max(range(len(r['gold'])), key=r['gold'].__getitem__))
    summary = {arm: dict(accuracy=sum(correct(r, arm) for r in rows)/64,
                        nll=-sum(sum(g*math.log(max(p, 1e-30)) for g, p in zip(r['gold'], r[arm])) for r in rows)/64)
               for arm in ('before', 'semantic', 'after', 'combined')}
    clusters = {}
    for r in rows:
        clusters.setdefault(r['image'], []).append(correct(r, 'combined')-correct(r, 'semantic'))
    groups = list(clusters.values()); rng = random.Random(97); diffs = []
    for _ in range(5000):
        draw = [rng.choice(groups) for _ in groups]
        diffs.append(sum(map(sum, draw))/sum(map(len, draw)))
    diffs.sort()
    delta = summary['combined']['accuracy']-summary['semantic']['accuracy']
    passed = delta >= .03 and diffs[125] > 0 and summary['combined']['nll'] <= summary['semantic']['nll']
    report = dict(summary=summary, combined_minus_semantic=delta, ci95=[diffs[125], diffs[4874]],
                  gate_pass=passed, semantic_reproduction_max_error=max_error, n=64, images=len(groups),
                  seconds=time.monotonic()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  limitations='monitoring size slice only; no training, final tests, calibration or release acceptance')
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
