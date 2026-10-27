"""Independent paired verification of complete exp105 device outputs."""
import json, math
from pathlib import Path
w=Path('runs/exp105-d3-crossdomain-device')
rows={}
for device in ('cpu','cuda-0'):
    assert (w/device/'EXIT_CODE').read_text().strip()=='0'
    rr=[json.loads(x) for x in (w/device/'details.jsonl').read_text().splitlines()]
    assert len(rr)==24
    rows[device]={(x['domain'],x['sample_id'],x['repeat']):x for x in rr}
    assert len(rows[device])==24
    for x in rr:
        p=x['probabilities']
        assert all(math.isfinite(v) and 0<=v<=1 for v in p.values())
        assert abs(sum(p.values())-1)<1e-5 and x['prediction']==max(p,key=p.get)
assert rows['cpu'].keys()==rows['cuda-0'].keys()
report={}
for domain in ('chart','document','photo','noul'):
    keys=[k for k in rows['cpu'] if k[0]==domain]
    assert len(keys)==6
    cross=0.;repeat=0.;changes=0
    for key in keys:
        a,b=rows['cpu'][key],rows['cuda-0'][key]
        assert a['probabilities'].keys()==b['probabilities'].keys()
        cross=max(cross,max(abs(v-b['probabilities'][k]) for k,v in a['probabilities'].items()))
        changes+=a['prediction']!=b['prediction']
        for device in rows:
            c=rows[device][(key[0],key[1],1-key[2])]
            repeat=max(repeat,max(abs(v-c['probabilities'][k]) for k,v in rows[device][key]['probabilities'].items()))
    report[domain]=dict(n=3,cross_max=cross,repeat_max=repeat,semantic_changes=changes,passed=cross<=.005 and repeat<=1e-5 and changes==0)
result=dict(domains=report,passed=all(x['passed'] for x in report.values()),scope='12 fixed monitoring samples, FP32; not formal benchmark or calibration')
with (w/'verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
