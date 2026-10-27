"""Report native D3 UI screening, including order sensitivity."""
import json
from pathlib import Path
from report_exp86 import main as baseline_report

if __name__ == '__main__':
    import sys
    work=Path('runs/exp87-d3-ui-pil')
    sys.argv=[sys.argv[0],str(work)]
    baseline_report()
    report=json.loads((work/'report.json').read_text())
    def read(control):
        rows=[json.loads(x) for x in (work/f'ui.{control}.details.jsonl').read_text().splitlines()]
        assert len(rows)==256
        return {x['sample_id']:x for x in rows}
    real,reverse=read('real'),read('reversed')
    assert real.keys()==reverse.keys()
    changed=sum(real[k]['prediction']!=reverse[k]['prediction'] for k in real)/256
    report['accuracy']['d3_real']=report['accuracy'].pop('smol_real')
    report['accuracy']['d3_rotation']=report['accuracy'].pop('smol_rotation')
    report['accuracy']['d3_reversed']=sum(x['correct'] for x in reverse.values())/256
    report['semantic_order_change_rate']=changed
    if changed>.2:
        report['decision']='validate_order_sensitivity_before_training'
    (work/'native_report.json').open('x').write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
