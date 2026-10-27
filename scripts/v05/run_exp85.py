"""Bounded UI pilot; refuses launch until data isolation and human audit pass."""
import hashlib,json,os,signal,time,subprocess
from pathlib import Path
import run_exp80 as supervisor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 os.chdir(supervisor.ROOT);work=Path('runs/exp85');work.mkdir(exist_ok=True);supervisor.WORK=work
 # Wait for the verified specific preceding queue, without taking over its jobs.
 parent_pid=int(Path('runs/v05-prep/ui_queue_20261024.pid').read_text())
 waiting_until=time.monotonic()+3600
 while True:
  p=Path('runs/v05-cycle-20261024/STATUS.json')
  state=json.loads(p.read_text()) if p.exists() else {}
  if state.get('state')=='completed':break
  try:
   cmd=Path(f'/proc/{parent_pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
  except FileNotFoundError:
   raise RuntimeError('Parent queue handle missing before terminal status; do not restart it')
  if 'run_ui_queue.py' not in cmd:raise RuntimeError('Parent PID identity differs')
  if time.monotonic()>waiting_until:raise TimeoutError('Waiting for preceding queue exceeded60min')
  supervisor.status(state='waiting_for_preceding_queue',parent_pid=parent_pid,parent_arm=state.get('arm'))
  time.sleep(30)
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if active:raise RuntimeError('Other GPU process active after preceding queue; do not interfere')
 audit=json.loads(Path('runs/v05-prep/exp85_data_audit.json').read_text())
 manual=json.loads(Path('runs/v05-prep/exp85_manual_audit.json').read_text())
 manifest=Path('data/v05/exp85/MANIFEST.json');meta=json.loads(manifest.read_text())
 if not audit['pass'] or not manual['pass'] or audit['manifest_sha256']!=sha(manifest) or manual['manifest_sha256']!=sha(manifest):raise RuntimeError('Required audits missing or stale')
 if len(manual['reviewed_ids'])!=48 or len(manual['reviewed_groups'])!=12:raise RuntimeError('12 tuple manual audit required')
 for n,d in meta['files'].items():
  if sha(f'data/v05/exp85/{n}.jsonl')!=d['sha256']:raise RuntimeError('Data changed')
 if Path('runs/exp85-cf-ui').exists():raise RuntimeError('Refusing overwrite or silent resume')
 with (work/'STARTED').open('x') as f:f.write(time.strftime('%F %T %z')+'\n')
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,supervisor.interrupted)
 include=json.loads(Path('runs/exp81/IDENTITY.json').read_text())
 # The pilot's Python include is stable and verified by exp80; no package sync or reinstall.
 pyinclude=json.loads(Path('runs/exp80/IDENTITY.json').read_text())['python_include']
 os.environ.update(UV_NO_SYNC='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 os.environ['CPATH']=pyinclude+(':'+os.environ['CPATH'] if os.environ.get('CPATH') else '')
 frozen=[*Path('scripts/v05').glob('*.py'),Path('scripts/data/v05/counterfactual_ui.py'),Path('configs/exp85-cf-ui.yaml'),manifest,
         Path('runs/exp81-chart/train/model.safetensors'),Path('runs/exp81-chart/train/config.json'),
         Path('docs/experiments/2026-10-24-exp85-cf-ui-plan.md'),Path('runs/v05-prep/exp85_manual_audit.json'),
         Path('src/devision/train/rlcd.py'),Path('src/devision/infer.py')]
 (work/'IDENTITY.json').write_text(json.dumps({'recorded':time.strftime('%F %T %z'),'sha256':{str(p):sha(p) for p in frozen}},indent=2)+'\n')
 supervisor.run(['devision-pipeline','configs/exp85-cf-ui.yaml','--dry-run'],'preflight',120)
 deadline=time.monotonic()+1800
 def evaluate(checkpoint,folder,endpoint=False):
  dest=work/folder;dest.mkdir()
  panels={'ui':'data/v05/exp85/ui_dev.jsonl','fit':'data/v05/exp85/fit.jsonl'}
  if endpoint:panels.update(chart='data/v05/exp80v2/chart_dev.jsonl',protect='data/v05/exp80v2/protect.jsonl',realworld='data/v05/external/realworld_native.jsonl')
  for name,path in panels.items():
   for control in (['real','mismatched','reversed'] if name=='ui' else ['real']):
    args=['devision-eval','--checkpoint',checkpoint,'--data',path,'--data-root','data','--device','cuda','--no-temperature','--role','monitoring','--out',str(dest/f'{name}.{control}.json')]
    if name=='ui' and control=='mismatched':
     args[args.index('--data')+1]='data/v05/exp85/ui_dev_rotated.jsonl'
    elif control!='real':args+=['--control',control]
    supervisor.run(args,folder+'_'+name+'_'+control,deadline-time.monotonic())
 evaluate('runs/exp81-chart/train','baseline')
 supervisor.run(['devision-pipeline','configs/exp85-cf-ui.yaml'],'pipeline',deadline-time.monotonic())
 train=json.loads(Path('runs/exp85-cf-ui/train/train_report.json').read_text())
 if len(train['losses'])!=1024 or train['train_items']!=8192:raise RuntimeError('Actual budget differs')
 evaluate('runs/exp85-cf-ui/train','raw',True)
 supervisor.run(['python','scripts/v05/report_exp85.py'],'report',min(180,deadline-time.monotonic()))
 result=json.loads((work/'report.json').read_text())
 supervisor.status(state='completed',checkpoint='runs/exp85-cf-ui/train',decision=result['decision'],report='runs/exp85/report.json')


if __name__=='__main__':
 try:main()
 except Exception as exc:
  print(type(exc).__name__,str(exc),flush=True)
  p=supervisor.WORK/'STATUS.json'
  if not p.exists() or json.loads(p.read_text()).get('state')=='running':supervisor.status(state='execution_failed',reason=str(exc))
  raise
