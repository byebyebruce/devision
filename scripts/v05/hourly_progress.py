"""Local read-only hourly training inspection; records evidence without restarting jobs."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/v05-monitor'


def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'exit_code':p.returncode,'output':p.stdout.strip(),'error':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'exit_code':-1,'output':'','error':str(exc)}


def main():
    now=datetime.now().astimezone()
    OUT.mkdir(parents=True,exist_ok=True)
    ps=command(['ps','-eo','pid,ppid,etime,args'])
    active=[line for line in ps['output'].splitlines() if any(s in line for s in ['train_main','eval_main','scripts/v05/run_exp','scripts/v05/run_ui_queue.py','devision-eval','devision-pipeline','scripts/data/v05/']) and 'hourly_progress' not in line]
    statuses={}
    for p in sorted([*(ROOT/'runs').glob('exp8*/STATUS.json'),*(ROOT/'runs').glob('v05-cycle-*/STATUS.json')]):
        statuses[str(p.relative_to(ROOT))]=json.loads(p.read_text())
    logs={}
    for p in sorted((ROOT/'runs').glob('exp8*/logs/train.log')):
        lines=p.read_text(errors='replace').splitlines()
        steps=[s for s in lines if s.startswith('epoch ')][-2:]
        logs[str(p.relative_to(ROOT))]={'age_seconds':round(now.timestamp()-p.stat().st_mtime),'last_steps':steps,'tail':lines[-2:]}
    reports={}
    for p in sorted((ROOT/'runs').glob('exp8*/report.json')):
        d=json.loads(p.read_text())
        reports[str(p.relative_to(ROOT))]={k:d[k] for k in ['decision','adopt_eligible','pass_conditions','P1_accuracy','P1_ui','P2_real_minus_within_tuple_rotation','P2_real_minus_mismatched','realworld'] if k in d}
    notes=[]
    for name,d in statuses.items():
        if d.get('state')=='awaiting_analysis':notes.append(name+': training/evaluation ended; analysis needed')
        if d.get('state') in ['execution_failed','incomplete','interrupted']:notes.append(name+': '+d['state'])
        if d.get('state')=='running' and not active:notes.append(name+': running state has no matching training/evaluation/supervisor process; inspect before any restart')
    if not active and not (OUT/'GOAL_COMPLETE').exists():
        notes.append('No active experiment process; v0.5 goal remains pending. Schedule the next evidence-based experiment rather than treating idle as success.')
    result=dict(checked_at=now.isoformat(),active_processes=active,statuses=statuses,training_logs=logs,
                gpu=command([shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader']),
                reports=reports,attention=notes,scope='Read-only local monitoring; no automatic restart or chat notification')
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    (OUT/(now.strftime('%Y%m%d-%H%M%S')+'.json')).write_text(text)
    temp=OUT/'latest.tmp';temp.write_text(text);temp.replace(OUT/'latest.json')
    print(now.isoformat(),'active_processes='+str(len(active)),'attention='+json.dumps(notes,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
