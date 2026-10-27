"""Frozen exp88 panel gates; picture-clustered comparisons on complete records."""
import json
from pathlib import Path
from devision.train.compare import load_records,paired_metric
from structure_report import occurrences
WORK=Path('runs/exp88-d3-panels-local')

def main():
    assert json.loads((WORK/'STATUS.json').read_text())['state']=='completed'
    def read(path):return occurrences(load_records(str(path)))
    def pair(a,b,metric='accuracy'):return paired_metric(a,b,metric=metric,resamples=5000,seed=88)
    panels={};harmful=[]
    for name in ['chart','protect','realworld']:
        a=read(Path('runs/exp81/raw')/f'{name}.real.details.jsonl')
        b=read(WORK/f'{name}.real.details.jsonl');wrong=read(WORK/f'{name}.mismatched.details.jsonl')
        gain=pair(a,b)
        if gain['difference']<=-.03 and gain['ci95'][1]<0:harmful.append(name)
        by_source={}
        if name=='protect':
            for source in sorted({r['source'] for r in a}):
                d=pair([r for r in a if r['source']==source],[r for r in b if r['source']==source])
                by_source[source]=d
                if d['difference']<=-.03 and d['ci95'][1]<0:harmful.append('protect:'+source)
        panels[name]=dict(accuracy=gain,nll_uncalibrated=pair(a,b,'nll'),visual_gain=pair(wrong,b),by_source=by_source)
    photo=pair(read(Path('runs/v05-prep/v04_realworld_raw.details.jsonl')),read(WORK/'realworld.real.details.jsonl'))
    gain=any(panels[n]['accuracy']['difference']>=.05 and panels[n]['accuracy']['ci95'][0]>0 for n in ['chart','realworld'])
    visual=all(panels[n]['visual_gain']['ci95'][0]>0 for n in ['chart','realworld'])
    result=dict(panels=panels,photo_vs_v04=photo,harmful=harmful,gain_pass=gain,visual_pass=visual,
        decision='design_training_pilot' if gain and visual and not harmful else 'do_not_expand_before_analysis',
        scope='route screening only; uncalibrated NLL; pretrained exposure unknown; no release qualification',seed=88,resamples=5000)
    (WORK/'report.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
