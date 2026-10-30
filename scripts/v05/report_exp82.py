"""Fixed end-point UI pilot decision (prewritten exp82 criteria, picture bootstrap seed82)."""
import json
from pathlib import Path
from devision.train.compare import load_records,paired_metric
from devision.train.evaluate import order_sensitivity
from structure_report import occurrences


def records(path):return occurrences(load_records(path))
def panel(folder,name,control='real'):return records(f'{folder}/{name}.{control}.details.jsonl')
def pair(a,b,metric='accuracy'):return paired_metric(a,b,metric,seed=82)
def harmed(d):return d['difference']<=-.03 and d['ci95'][1]<0


def main():
 base='runs/exp82/baseline';end='runs/exp82/raw';init='runs/exp81/raw'
 ui=panel(end,'ui');p1=pair(panel(base,'ui'),ui);p2=pair(panel(end,'ui','mismatched'),ui)
 chart=pair(panel(init,'chart'),panel(end,'chart'))
 old,new=panel(init,'protect'),panel(end,'protect')
 source={s:pair([r for r in old if r['source']==s],[r for r in new if r['source']==s]) for s in sorted({r['source'] for r in old})}
 alarm=[s for s,d in source.items() if harmed(d)]
 nll=pair(old,new,'nll');photo=pair(panel(init,'realworld'),panel(end,'realworld'))
 gates=dict(P1=p1['difference']>=.05 and p1['ci95'][0]>0,
            P2=p2['difference']>=.03 and p2['ci95'][0]>0,
            P3=not harmed(chart) and not alarm and nll['difference']<=.05 and not harmed(photo))
 order=order_sensitivity(ui,panel(end,'ui','reversed'))
 if order['n']!=len(ui):raise ValueError('Incomplete option-order panel')
 out=dict(P1_ui=p1,P2_real_minus_mismatched=p2,chart_protection=chart,old_protect_nll=nll,
          protect_sources=source,harmful_sources=alarm,realworld=photo,
          fit=pair(panel(base,'fit'),panel(end,'fit')),order_sensitivity=order,
          pass_conditions=gates,expand=all(gates.values()),bootstrap_seed=82)
 Path('runs/exp82/report.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='protect_sources'},indent=2))


if __name__=='__main__':main()
