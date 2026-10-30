"""Start the prewritten scale arm only after a valid completed exp80 decision and data audit."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml
import run_exp80 as supervisor


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.chdir(supervisor.ROOT)
    work = Path('runs/exp81')
    work.mkdir(parents=True, exist_ok=True)
    supervisor.WORK = work
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, supervisor.interrupted)
    status80 = json.loads(Path('runs/exp80/STATUS.json').read_text())
    result = json.loads(Path('runs/exp80/report.json').read_text())
    if status80['state'] != 'completed' or not result['decision'].startswith('expand_'):
        raise RuntimeError('exp80 has not selected a valid arm')
    arm = result['decision'].split('_')[1]
    if arm not in ('r', 'c') or not result['arms'][arm]['expand']:
        raise RuntimeError('Invalid selected arm')
    if Path('runs/exp81-chart').exists():
        raise RuntimeError('Scale arm output exists; refusing overwrite or silent resume')
    audit = json.loads(Path('runs/v05-prep/exp81_data_audit.json').read_text())
    manifest_path = Path('data/v05/exp81/MANIFEST.json')
    manifest = json.loads(manifest_path.read_text())
    if not audit['pass'] or audit['manifest_sha256'] != sha(manifest_path):
        raise RuntimeError('Scale audit missing or stale')
    for name, meta in manifest['files'].items():
        if sha(f'data/v05/exp81/{name}.jsonl') != meta['sha256']:
            raise RuntimeError('Scale data changed: ' + name)
    # Shared raw evaluator points to exp80's panels; require byte equality to the prewritten copied panels.
    for name in ['chart_dev', 'calib', 'protect', 'fit']:
        if sha(f'data/v05/exp81/{name}.jsonl') != sha(f'data/v05/exp80v2/{name}.jsonl'):
            raise RuntimeError('Fixed panel changed: ' + name)
    if manifest['files']['train']['questions'] != 48000:
        raise RuntimeError('Unexpected update budget')
    cfg = yaml.safe_load(Path(f'configs/exp80-chart-{arm}.yaml').read_text())
    cfg['name'] = 'exp81-chart'
    cfg.pop('evaluate', None)
    stage = cfg['stages'][0]
    stage['data'], stage['val'] = 'data/v05/exp81/train.jsonl', 'data/v05/exp81/calib.jsonl'
    stage['eval'] = {n: f'data/v05/exp81/{f}.jsonl' for n, f in [('chart','chart_dev'),('protect','protect'),('fit','fit')]}
    stage['params'].pop('start_check', None)
    stage['params'].pop('start_tolerance', None)
    stage['params'].update(seed=81, warmup=300)
    cfgpath = Path('configs/exp81-chart.yaml')
    if cfgpath.exists():
        raise RuntimeError('Scale config already exists')
    cfgpath.write_text('# Generated from the completed exp80 decision; see exp81 plan.\n' + yaml.safe_dump(cfg, sort_keys=False))
    (work / 'STARTED').write_text(time.strftime('%F %T %z')+'\n')
    files = [*Path('scripts/v05').glob('*.py'), cfgpath, manifest_path,
             Path('runs/r21-soft/train/model.safetensors'), Path('runs/r21-soft/train/config.json'),
             Path('src/devision/train/rlcd.py'), Path('src/devision/infer.py')]
    (work / 'IDENTITY.json').write_text(json.dumps(dict(selected_pilot=arm, decision=result['decision'],
        exp80_report_sha256=sha('runs/exp80/report.json'),
        sha256={str(p):sha(p) for p in files},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True)),indent=2)+'\n')
    os.environ.update(UV_NO_SYNC='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    # Reuse the exact include verified by the completed pilot, rather than reinstalling Python or PyTorch.
    include = json.loads(Path('runs/exp80/IDENTITY.json').read_text())['python_include']
    os.environ['CPATH'] = include + (':' + os.environ['CPATH'] if os.environ.get('CPATH') else '')
    supervisor.run(['devision-pipeline',str(cfgpath),'--dry-run'], 'preflight', 120)
    deadline = time.monotonic() + 3600
    supervisor.run(['devision-pipeline',str(cfgpath)], 'pipeline', deadline-time.monotonic())
    train = json.loads(Path('runs/exp81-chart/train/train_report.json').read_text())
    if len(train['losses']) != 6000 or train['train_items'] != 48000:
        raise RuntimeError('Unexpected actual training budget')
    supervisor.evaluate('runs/exp81-chart/train',work/'raw',deadline)
    supervisor.run(['devision-eval','--checkpoint','runs/exp81-chart/train',
                    '--data','data/v05/external/realworld_native.jsonl','--data-root','data',
                    '--device','cuda','--no-temperature','--role','monitoring',
                    '--out',str(work/'raw/realworld.real.json')], 'realworld_real', deadline-time.monotonic())
    supervisor.status(state='awaiting_analysis', selected_pilot=arm, checkpoint='runs/exp81-chart/train')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__,str(exc),flush=True)
        p=supervisor.WORK/'STATUS.json'
        if not p.exists() or json.loads(p.read_text()).get('state')=='running':
            supervisor.status(state='execution_failed',reason=str(exc))
        raise
