"""Run the prewritten exp83/84 queue sequentially, evaluate and decide after every arm."""
import hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
import run_exp80 as runner
from devision.train.compare import load_records,paired_metric
from devision.train.evaluate import order_sensitivity
from structure_report import occurrences

ROOT=runner.ROOT
QUEUE=ROOT/'runs/v05-cycle-20261024'
PANELS={'ui':'data/v05/exp82v2/ui_dev.jsonl','fit':'data/v05/exp82v2/fit.jsonl',
        'chart':'data/v05/exp80v2/chart_dev.jsonl','protect':'data/v05/exp80v2/protect.jsonl',
        'realworld':'data/v05/external/realworld_native.jsonl'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(folder,name,control='real'):
 return occurrences(load_records(str(folder/f'{name}.{control}.details.jsonl')))
def analyzed(work,seed):
 def pair(a,b,metric='accuracy'):return paired_metric(a,b,metric,seed=seed)
 def harm(d):return d['difference']<=-.03 and d['ci95'][1]<0
 base,end=work/'baseline',work/'raw';init=ROOT/'runs/exp81/raw'
 ui=rows(end,'ui');p1=pair(rows(base,'ui'),ui);p2=pair(rows(end,'ui','mismatched'),ui)
 chart=pair(rows(init,'chart'),rows(end,'chart'));old,new=rows(init,'protect'),rows(end,'protect')
 groups={s:pair([r for r in old if r['source']==s],[r for r in new if r['source']==s]) for s in sorted({r['source'] for r in old})}
 bad=[s for s,d in groups.items() if harm(d)];nll=pair(old,new,'nll');photo=pair(rows(init,'realworld'),rows(end,'realworld'))
 gates=dict(P1=p1['difference']>=.05 and p1['ci95'][0]>0,P2=p2['difference']>=.03 and p2['ci95'][0]>0,
            P3=not harm(chart) and not bad and nll['difference']<=.05 and not harm(photo))
 order=order_sensitivity(ui,rows(end,'ui','reversed'))
 assert order['n']==len(ui)
 result=dict(P1_ui=p1,P2_real_minus_mismatched=p2,chart_protection=chart,old_protect_nll=nll,
             protect_sources=groups,harmful_sources=bad,realworld=photo,order_sensitivity=order,
             fit=pair(rows(base,'fit'),rows(end,'fit')),pass_conditions=gates,
             versus_exp82=pair(rows(ROOT/'runs/exp82/raw','ui'),ui),
             decision='retain_candidate' if all(gates.values()) else 'reject',bootstrap_seed=seed)
 if seed==83:
  result['folded_start_vs_exp81']={n:pair(rows(init,n),rows(base,n),'nll' if n=='protect' else 'accuracy') for n in ['chart','protect','realworld']}
 (work/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 return result


def gpu_free():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return not p.stdout.strip()


def evaluate(checkpoint,dest,deadline,full=True):
 dest.mkdir(exist_ok=True)
 for name,path in PANELS.items():
  if not full and name not in ['ui','fit']:continue
  for control in (['real','mismatched','reversed'] if name=='ui' else ['real']):
   args=['devision-eval','--checkpoint',checkpoint,'--data',path,'--data-root','data','--device','cuda','--no-temperature','--role','monitoring','--out',str(dest/f'{name}.{control}.json')]
   if control!='real':args+=['--control',control]
   runner.run(args,dest.name+'_'+name+'_'+control,deadline-time.monotonic())


def main():
 os.chdir(ROOT);QUEUE.mkdir(exist_ok=True)
 with (QUEUE/'STARTED').open('x') as f:f.write(time.strftime('%F %T %z')+'\n')
 os.environ.update(UV_NO_SYNC='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 os.environ['CPATH']=json.loads(Path('runs/exp80/IDENTITY.json').read_text())['python_include']
 for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,runner.interrupted)
 audit=json.loads(Path('runs/v05-prep/exp82_data_audit.json').read_text());data=Path('data/v05/exp82v2')
 assert audit['pass'] and sha(data/'MANIFEST.json')==audit['manifest_sha256']
 for name,meta in json.loads((data/'MANIFEST.json').read_text())['files'].items():assert sha(data/(name+'.jsonl'))==meta['sha256']
 fixed=[Path('configs/exp83-ui-tok256.yaml'),Path('configs/exp84-ui-frozen.yaml'),Path('scripts/v05/run_ui_queue.py'),Path('scripts/exp20/fold.py'),Path('src/devision/train/rlcd.py'),Path('src/devision/infer.py'),Path('runs/exp81-chart/train/model.safetensors'),Path('runs/exp81-chart/train/config.json'),data/'MANIFEST.json',Path('docs/experiments/2026-10-24-exp83-84-plan.md')]
 identity={str(p):sha(p) for p in fixed}
 (QUEUE/'IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
 summary={}
 for number,name in [(83,'exp83-ui-tok256'),(84,'exp84-ui-frozen')]:
  work=ROOT/f'runs/exp{number}';work.mkdir(exist_ok=True);runner.WORK=work
  if (ROOT/f'runs/{name}').exists():raise RuntimeError('Output exists; refusing silent resume')
  (QUEUE/'STATUS.json').write_text(json.dumps(dict(updated=time.strftime('%F %T %z'),state='running',arm=number,completed=summary),indent=2)+'\n')
  if not gpu_free():
   runner.status(state='waiting_for_resource',reason='Another GPU process is active')
   raise RuntimeError('Other GPU workload; do not interfere')
  if any(sha(p)!=h for p,h in identity.items()):raise RuntimeError('Frozen inputs changed')
  deadline=time.monotonic()+1800
  try:
   if number==83:
    runner.run(['python','scripts/exp20/fold.py','runs/exp81-chart/train','runs/exp83/init-tok256'],'fold',deadline-time.monotonic())
    evaluate('runs/exp83/init-tok256',work/'baseline',deadline)
   else:
    base=work/'baseline';base.mkdir()
    # Exact exp81 checkpoint and UI panel have not changed: reuse the recorded exp82 starting-point outputs.
    old_id=json.loads(Path('runs/exp82/IDENTITY.json').read_text())['sha256']
    for p in ['runs/exp81-chart/train/model.safetensors','runs/exp81-chart/train/config.json']:
     assert sha(p)==old_id[p]
    for p in Path('runs/exp82/baseline').glob('*'):
     if p.is_file():shutil.copyfile(p,base/p.name)
   runner.run(['devision-pipeline',f'configs/{name}.yaml','--dry-run'],'preflight',min(120,deadline-time.monotonic()))
   runner.run(['devision-pipeline',f'configs/{name}.yaml'],'pipeline',deadline-time.monotonic())
   train=json.loads(Path(f'runs/{name}/train/train_report.json').read_text())
   assert len(train['losses'])==1024 and train['train_items']==8192
   evaluate(f'runs/{name}/train',work/'raw',deadline)
   result=analyzed(work,number)
   runner.status(state='completed',decision=result['decision'],report=str(work/'report.json'))
   summary[str(number)]={'state':'completed','decision':result['decision'],'ui_accuracy':result['P1_ui']['b'],'pass':result['pass_conditions']}
  except (Exception,AssertionError) as exc:
   summary[str(number)]={'state':'execution_failed','reason':str(exc)}
   print('ARM FAILED',number,str(exc),flush=True)
   runner.status(state='execution_failed',reason=str(exc))
   # Every subprocess is awaited or stopped by runner.run; never restart an expired observation handle.
  (QUEUE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 (QUEUE/'STATUS.json').write_text(json.dumps(dict(updated=time.strftime('%F %T %z'),state='completed',arms=summary),indent=2)+'\n')
 print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
