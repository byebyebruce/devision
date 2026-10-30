"""Bounded UI pilot; refuses launch until data isolation and human audit pass."""
import hashlib,json,os,signal,time
from pathlib import Path
import run_exp80 as supervisor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 os.chdir(supervisor.ROOT);work=Path('runs/exp82');work.mkdir(exist_ok=True);supervisor.WORK=work
 audit=json.loads(Path('runs/v05-prep/exp82_data_audit.json').read_text())
 manual=json.loads(Path('runs/v05-prep/exp82_manual_audit.json').read_text())
 manifest=Path('data/v05/exp82v2/MANIFEST.json');meta=json.loads(manifest.read_text())
 if not audit['pass'] or not manual['pass'] or audit['manifest_sha256']!=sha(manifest) or manual['manifest_sha256']!=sha(manifest):raise RuntimeError('Required audits missing or stale')
 if len(manual['reviewed_ids'])!=50:raise RuntimeError('50 question manual audit required')
 for n,d in meta['files'].items():
  if sha(f'data/v05/exp82v2/{n}.jsonl')!=d['sha256']:raise RuntimeError('Data changed')
 if Path('runs/exp82-ui').exists():raise RuntimeError('Refusing overwrite or silent resume')
 with (work/'STARTED').open('x') as f:f.write(time.strftime('%F %T %z')+'\n')
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,supervisor.interrupted)
 include=json.loads(Path('runs/exp81/IDENTITY.json').read_text())
 # The pilot's Python include is stable and verified by exp80; no package sync or reinstall.
 pyinclude=json.loads(Path('runs/exp80/IDENTITY.json').read_text())['python_include']
 os.environ.update(UV_NO_SYNC='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 os.environ['CPATH']=pyinclude+(':'+os.environ['CPATH'] if os.environ.get('CPATH') else '')
 frozen=[*Path('scripts/v05').glob('*.py'),Path('scripts/data/v05/screenqa.py'),Path('configs/exp82-ui.yaml'),manifest,
         Path('runs/exp81-chart/train/model.safetensors'),Path('runs/exp81-chart/train/config.json'),
         Path('docs/experiments/2026-10-24-exp82-ui-plan.md'),Path('runs/v05-prep/exp82_manual_audit.json'),
         Path('src/devision/train/rlcd.py'),Path('src/devision/infer.py')]
 (work/'IDENTITY.json').write_text(json.dumps({'recorded':time.strftime('%F %T %z'),'sha256':{str(p):sha(p) for p in frozen}},indent=2)+'\n')
 supervisor.run(['devision-pipeline','configs/exp82-ui.yaml','--dry-run'],'preflight',120)
 deadline=time.monotonic()+1800
 def evaluate(checkpoint,folder,endpoint=False):
  dest=work/folder;dest.mkdir()
  panels={'ui':'data/v05/exp82v2/ui_dev.jsonl','fit':'data/v05/exp82v2/fit.jsonl'}
  if endpoint:panels.update(chart='data/v05/exp80v2/chart_dev.jsonl',protect='data/v05/exp80v2/protect.jsonl',realworld='data/v05/external/realworld_native.jsonl')
  for name,path in panels.items():
   for control in (['real','mismatched','reversed'] if name=='ui' else ['real']):
    args=['devision-eval','--checkpoint',checkpoint,'--data',path,'--data-root','data','--device','cuda','--no-temperature','--role','monitoring','--out',str(dest/f'{name}.{control}.json')]
    if control!='real':args+=['--control',control]
    supervisor.run(args,folder+'_'+name+'_'+control,deadline-time.monotonic())
 evaluate('runs/exp81-chart/train','baseline')
 supervisor.run(['devision-pipeline','configs/exp82-ui.yaml'],'pipeline',deadline-time.monotonic())
 train=json.loads(Path('runs/exp82-ui/train/train_report.json').read_text())
 if len(train['losses'])!=1024 or train['train_items']!=8192:raise RuntimeError('Actual budget differs')
 evaluate('runs/exp82-ui/train','raw',True)
 supervisor.status(state='awaiting_analysis',checkpoint='runs/exp82-ui/train')


if __name__=='__main__':
 try:main()
 except Exception as exc:
  print(type(exc).__name__,str(exc),flush=True)
  p=supervisor.WORK/'STATUS.json'
  if not p.exists() or json.loads(p.read_text()).get('state')=='running':supervisor.status(state='execution_failed',reason=str(exc))
  raise
