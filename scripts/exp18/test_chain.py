"""exp18's chain (critic EXP18-I01) with stand-in commands, no model: a start mismatch is not retried and ends
non-zero, a crash is retried, and the readout's exit status is the chain's."""
import os
import subprocess
import textwrap

CHAIN = os.path.join(os.path.dirname(__file__), "chain.sh")
SETS = "val exist science ext bind_gqa count_narrow count_wide px_wide px_narrow px_exist flip vsr".split()


def chain(tmp_path, train_body, report_status=0):
    stage = tmp_path / "run" / "stage2"
    stage.mkdir(parents=True)
    calls = tmp_path / "calls"
    train = tmp_path / "train.sh"
    train.write_text("#!/usr/bin/env bash\necho x >> %s\nn=$(wc -l < %s)\nSTAGE=%s\n%s\n"
                     % (calls, calls, stage, textwrap.dedent(train_body)))
    report = tmp_path / "report.sh"
    report.write_text("#!/usr/bin/env bash\necho readout\nexit %d\n" % report_status)
    for f in (train, report):
        f.chmod(0o755)
    env = dict(os.environ, EXP18_TRAIN=str(train), EXP18_REPORT=str(report), EXP18_STAGE=str(stage), EXP18_POLL="0.2")
    p = subprocess.run(["bash", CHAIN], env=env, capture_output=True, text=True, timeout=60)
    n = len(calls.read_text().splitlines()) if calls.exists() else 0
    return p, n, stage


REACH_2000 = """
mkdir -p $STAGE/monitor/step2000
for s in %s; do touch $STAGE/monitor/step2000/$s.details.jsonl; done
sleep 30
""" % " ".join(SETS)


def test_a_start_mismatch_stops_without_retry(tmp_path):
    p, n, stage = chain(tmp_path, "echo 'val: differ' > $STAGE/START_MISMATCH\nexit 1")
    assert p.returncode == 3
    assert n == 1
    assert not (tmp_path / "run" / "report.txt").exists()


def test_a_crash_is_retried(tmp_path):
    p, n, stage = chain(tmp_path, "[ $n -ge 2 ] || exit 1\n" + REACH_2000)
    assert p.returncode == 0, p.stdout
    assert n == 2
    assert (stage / "DONE_AT_2000").exists()


def test_the_readout_status_is_the_chain_status(tmp_path):
    p, n, stage = chain(tmp_path, REACH_2000, report_status=2)
    assert p.returncode == 2
    assert (stage / "DONE_AT_2000").exists()
    assert "readout" in (tmp_path / "run" / "report.txt").read_text()


def test_another_stop_step(tmp_path):
    """EXP18_STEP (exp25): the run ends once that step's records are complete; step 2000 alone does not end it."""
    body = """
mkdir -p $STAGE/monitor/step2000 $STAGE/monitor/step5000
for s in %s; do touch $STAGE/monitor/step2000/$s.details.jsonl; done
sleep 1
for s in %s; do touch $STAGE/monitor/step5000/$s.details.jsonl; done
sleep 30
""" % (" ".join(SETS), " ".join(SETS))
    stage = tmp_path / "run" / "stage2"
    stage.mkdir(parents=True)
    train = tmp_path / "train.sh"
    train.write_text("#!/usr/bin/env bash\nSTAGE=%s\n%s\n" % (stage, textwrap.dedent(body)))
    report = tmp_path / "report.sh"
    report.write_text("#!/usr/bin/env bash\nexit 0\n")
    for f in (train, report):
        f.chmod(0o755)
    env = dict(os.environ, EXP18_TRAIN=str(train), EXP18_REPORT=str(report), EXP18_STAGE=str(stage),
               EXP18_POLL="0.2", EXP18_STEP="5000")
    p = subprocess.run(["bash", CHAIN], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and (stage / "DONE_AT_5000").exists() and not (stage / "DONE_AT_2000").exists()
    assert "reached step 5000" in p.stdout
