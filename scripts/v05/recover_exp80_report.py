"""Record and repair the descriptive choice-only order metric after the frozen pilot finishes."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    work = Path('runs/exp80')
    status = json.loads((work/'STATUS.json').read_text())
    if status.get('command') != 'report' or status['state'] != 'execution_failed':
        raise RuntimeError('Only recover a finished pilot whose report command failed')
    if 'KeyError' not in (work/'report.log').read_text():
        raise RuntimeError('Unexpected report failure; diagnose separately')
    identity = json.loads((work/'IDENTITY.json').read_text())
    changed = [p for p,h in identity['sha256'].items() if sha(p) != h]
    if changed:
        raise RuntimeError('Original frozen inputs changed: ' + repr(changed))
    path=Path('scripts/v05/report_exp80.py')
    source=path.read_text()
    before='    flips = sum(r["prediction"] != reversed_rows[(r["sample_id"], r["qid"])]["prediction"] for r in new) / len(new)'
    after='''    choice = [r for r in new if r["type"] == "choice"]
    if {(r["sample_id"], r["qid"]) for r in choice} != set(reversed_rows):
        raise ValueError("Reversed choice panel is incomplete or mismatched")
    flips = sum(r["prediction"] != reversed_rows[(r["sample_id"], r["qid"])]["prediction"] for r in choice) / len(choice)'''
    if source.count(before)!=1:
        raise RuntimeError('Unexpected report source')
    (work/'report_exp80.original.py').write_text(source)
    path.write_text(source.replace(before,after))
    with (work/'report_recovered.log').open('w') as f:
        subprocess.run(['uv','run','--no-sync','python',str(path)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    recovery=dict(time=time.strftime('%F %T %z'),original_status=status,
                  original_report_sha256=identity['sha256'][str(path)],patched_report_sha256=sha(path),
                  change='Choice-only order sensitivity denominator/key coverage. P1/P2/P3, bootstrap and selection rules unchanged.',
                  original_inputs_unchanged_before_patch=True,training_and_raw_outputs_unchanged=True)
    (work/'analysis_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    new=dict(updated=time.strftime('%F %T %z'),state='completed',report='runs/exp80/report.json',analysis_recovery='runs/exp80/analysis_recovery.json')
    tmp=work/'STATUS.tmp';tmp.write_text(json.dumps(new,indent=2)+'\n');tmp.replace(work/'STATUS.json')
    print(json.dumps(new,indent=2))


if __name__=='__main__':
    main()
