"""Cron-triggered engineering turn, with a singleton lock and bounded agent runtime."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/v05-monitor/engineer'
CODEX = Path('/home/bruce/.nvm/versions/node/v24.22.0/bin/codex')
PROMPT = '''你是deVision模型训练工程师。负责人明确要求每小时检查并实际推进v0.5，保持有价值的训练/评测串行衔接。
先阅读AGENTS.md、CLAUDE.md、docs/experiments/2026-10-22-v05-goal.md和最新训练日志、研究策略；以实际进程、GPU、日志和checkpoint为准，不能只读STATUS。
本轮必须：1核对训练/评测进度与异常；2若有正常活跃GPU任务，不重复开训，利用CPU分析结果、准备后续独立配置/数据；3若空闲，分析上一实验并选择有证据的下一项，预写假设/预算/门禁，然后实际启动职责内本地任务。当前主线见docs/experiments/2026-10-24-next-training-options.md。
exp86 SmolVLM已有原始失败目录和PIL修复尝试目录，检查最新实际状态，保留原始证据。模型/包下载走国内镜像，uv run --no-sync，不重装现有CUDA。已失败的相同条件实验不重复；不为占GPU无意义空跑。需要新方向时先查历史。
所有任务留日志/checkpoint/退出条件，单卡串行，协调其他活跃执行者，禁止杀他人进程。不要覆盖唯一权重。保留exp81可靠开发控制组。
不得发布、外传数据、上传权重、远程git推送、发消息、购买算力。不得修改本调度脚本或递归创建agent/定时任务。若负责人已明确暂停则停止工作并记录。仅使用现有本地资源和已配置训练上报。
本轮最多50分钟，长GPU任务可独立后台继续但必须有硬预算及可恢复状态。离开前确认已启动任务的实际PID/日志，写清下一项及启动条件；本轮报告追加到docs/training-log.md，普通进度不做critic。
最终中文报告实际运行、证据、下一步、阻塞，不能只列计划或声称v0.5已完成。'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('skip: previous hourly engineer still running', flush=True)
            return
        if (OUT / 'PAUSE').exists():
            print('skip: PAUSE marker', flush=True)
            return
        now = datetime.datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
        env = dict(os.environ)
        env['PATH'] = str(CODEX.parent)+':/home/bruce/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib'
        env['UV_NO_SYNC'] = '1'
        args = [str(CODEX), 'exec', '-c', 'approval_policy="never"', '--sandbox',
                'danger-full-access', '--output-last-message', str(OUT / f'{now}.md'), '-']
        with (OUT / f'{now}.log').open('w') as log:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (OUT / 'latest.json').write_text(json.dumps(dict(started=now, pid=process.pid, state='running'))+'\n')
            try:
                process.communicate(PROMPT.encode(), timeout=3000)
                state = 'completed' if process.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                state = 'time_limit'
            (OUT / 'latest.json').write_text(json.dumps(dict(started=now, pid=process.pid, state=state, exit_code=process.returncode))+'\n')
            print(now, state, process.returncode, flush=True)


if __name__ == '__main__':
    main()
