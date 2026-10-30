"""Full actual mixed-image isolation and label/prior audit before exp82."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from audit_exp80 import read
from chart_data import digest,pixels


def main():
 root=Path('data');data=root/'v05/exp82v2'
 manifest=json.loads((data/'MANIFEST.json').read_text())
 sets={n:read(data/(n+'.jsonl')) for n in ['train','ui_train','ui_dev','calib']}
 for n,path in dict(chart='v05/exp80v2/chart_dev.jsonl',protect='v05/exp80v2/protect.jsonl',realworld='v05/external/realworld_native.jsonl',examples='v05/external/jev_examples.jsonl').items():sets[n]=read(root/path)
 for n,meta in manifest['files'].items():assert digest((data/(n+'.jsonl')).read_bytes())==meta['sha256']
 paths=sorted({r['image'] for rs in sets.values() for r in rs})
 def hashed(p):return p,pixels((root/p).read_bytes())[0]
 with ThreadPoolExecutor(max_workers=4) as pool:h=dict(pool.map(hashed,paths))
 ids={n:{h[r['image']] for r in rs} for n,rs in sets.items()}
 overlaps={n:len(ids['train']&s) for n,s in ids.items() if n not in ['train','ui_train']}
 labels=Counter()
 for r in sets['ui_train']:
  gold=r['gold']['q']['probabilities'];q=r['questions']['q']
  assert len(gold)==4 and sum(gold.values())==1 and set(gold)==set(q['criteria'])
  labels[max(gold,key=gold.get)]+=1
 hits=sum(r['gold']['q']['probabilities'][max(sorted(r['gold']['q']['probabilities']),key=lambda x:labels[x])] for r in sets['ui_dev'])
 prior=hits/len(sets['ui_dev'])
 result=dict(manifest_sha256=digest((data/'MANIFEST.json').read_bytes()),images_hashed=len(paths),overlaps=overlaps,
             label_prior_accuracy=prior,random_accuracy=.25,train_presentations=len(sets['train']),
             dev_questions=len(sets['ui_dev']),ui_unique_questions=len(sets['ui_train']))
 result['pass']=not any(overlaps.values()) and prior<=.35 and len(sets['train'])==8192 and len(sets['ui_train'])==512
 Path('runs/v05-prep/exp82_data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 if not result['pass']:raise RuntimeError('exp82 data audit failed; do not train')


if __name__=='__main__':main()
