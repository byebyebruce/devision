"""Run in a foreground tool session; persist child identity, log and actual return status."""
import json,os,subprocess,time,sys,signal
from pathlib import Path
import yaml
config=sys.argv[1];cfg=yaml.safe_load(Path(config).read_text());work=Path(cfg['out']).parent;work.mkdir(parents=True,exist_ok=True);start=time.time()
with (work/'training.log').open('x') as log:
    p=subprocess.Popen(['uv','run','--no-sync','python','scripts/v05/d3_science_adapt.py',config],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (work/'launch.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),child_pid=p.pid,start_epoch=start,command=p.args),indent=2))
    try:code=p.wait(timeout=2700)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        code=124
    (work/'EXIT_CODE').write_text(str(code)+'\n');print('EXIT_CODE',code,flush=True)
