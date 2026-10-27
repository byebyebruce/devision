"""Apply the prewritten scale protocol to saved decide outputs; bootstrap seed81."""
import json
from pathlib import Path
from devision.train.compare import load_records, paired_metric
from structure_report import occurrences
from devision.train.evaluate import order_sensitivity


def rows(base,name,control='real'):
    return occurrences(load_records(f'{base}/{name}.{control}.details.jsonl'))


def pair(a,b,metric='accuracy'):
    return paired_metric(a,b,metric,seed=81)


def main():
    base='runs/exp80/baseline/raw'; new='runs/exp81/raw'
    chart=rows(new,'chart'); old=rows(base,'protect'); protect=rows(new,'protect')
    p1=pair(rows(base,'chart'),chart)
    p2=pair(rows(new,'chart','mismatched'),chart)
    p3=pair(old,protect,'nll')
    groups={s:pair([r for r in old if r['source']==s],[r for r in protect if r['source']==s]) for s in sorted({r['source'] for r in old})}
    harmful=[s for s,d in groups.items() if d['difference']<=-.03 and d['ci95'][1]<0]
    photo=pair(occurrences(load_records('runs/v05-prep/v04_realworld_raw.details.jsonl')),rows(new,'realworld'))
    gates=dict(P1=p1['difference']>=.03 and p1['ci95'][0]>0,P2=p2['ci95'][0]>0,
               P3=not harmful and p3['difference']<=.05,
               P4=not(photo['difference']<=-.03 and photo['ci95'][1]<0))
    out=dict(P1_accuracy=p1,P2_real_minus_mismatched=p2,P3_protect_nll=p3,protect_sources=groups,
             harmful_sources=harmful,realworld=photo,pass_conditions=gates,
             pilot_comparison=pair(rows('runs/exp80/c/raw','chart'),chart),
             adopt_eligible=all(gates.values()),bootstrap_seed=81,
             fit_comparison=pair(rows(base,'fit'),rows(new,'fit')),
             order_sensitivity=order_sensitivity(chart,rows(new,'chart','reversed')))
    Path('runs/exp81/report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='protect_sources'},indent=2))


if __name__=='__main__':main()
