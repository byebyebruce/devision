"""exp56 runner without the real model: start refuses a selector / head that is not the registered one; the read
fidelity rule reproduces on EXP53's own strings and fails when 2% differ; the run finalizes inside its limit through the
real watch."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp56", os.path.join(HERE, "exp56.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
e54 = m.e54


def test_start_refuses_other_components(tmp_path, monkeypatch):
    fake = tmp_path / "selector.pt"
    fake.write_bytes(b"not the selector")
    monkeypatch.setattr(m, "SELECTOR", str(fake))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with pytest.raises(SystemExit, match="selector"):
            m.stage_start(0, out=str(tmp_path / "out"))
    finally:
        os.chdir(cwd)
    assert not (tmp_path / "out" / "START.json").exists()


def exp53_live(rows):
    qs = [json.loads(x) for x in open(e54.QUESTIONS)]
    pred, _, _ = e54.p49.read_details(os.path.join(m.E53, "details_auto.jsonl"), qs)
    live = {}
    for q, p in zip(qs, pred):
        if q["set"] == "rl_tune":
            d = e54.p49.decode(p)
            t = rows[q["id"]]["questions"][q["qid"]]["instructions"]
            live.setdefault(q["id"], {})[t] = {"question": t, "decoded": d["text"], "valid": d["valid"]}
    return live


@pytest.fixture(scope="module")
def rows():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return {r["id"]: r for r in map(json.loads, open(e54.TUNE))}
    finally:
        os.chdir(cwd)


def in_root(fn, *a, **k):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return fn(*a, **k)
    finally:
        os.chdir(cwd)


def test_read_fidelity_rule(rows):
    live = in_root(exp53_live, rows)
    ok = in_root(m.read_fidelity, rows, live)
    assert ok["questions"] == 1800 and ok["same_strings"] == 1.0 and ok["reproduced"] and len(ok["_same_pictures"]) == 600
    n = 0
    for v in live.values():                                            # 2% of the strings changed
        for r in v.values():
            if n < 36:
                r["decoded"], r["valid"] = "9999", True
                n += 1
    assert not in_root(m.read_fidelity, rows, live)["reproduced"]


def fake_run(tmp_path, rows, inputs_change=None):
    """A run directory as start / live leave it: START, pictures.json, bundle files, reads (EXP53's strings), inputs_P and
    details_P (EXP55's records with this run's identity), all registered."""
    from devision.reading import TEMPLATES, fmt, reads_for
    out = tmp_path / "exp56"
    (out / "bundle").mkdir(parents=True)
    for f in ("reading.json", "selector.pt", "head.pt"):
        (out / "bundle" / f).write_text(f)
    pics = {r["image"]: "0" * 64 for r in rows.values()}
    (out / "pictures.json").write_text(json.dumps(pics))
    st = {"files": {}, "model": e54.T_SHA, "bundle": {f: e54.sha(str(out / "bundle" / f)) for f in ("reading.json", "selector.pt", "head.pt")},
          "pictures": {"pictures.json": e54.sha(str(out / "pictures.json"))}, "questions_sha256": e54.sha(os.path.join(ROOT, e54.QUESTIONS)),
          "tune_sha256": e54.sha(os.path.join(ROOT, e54.TUNE))}
    (out / "START.json").write_text(json.dumps(st))
    live = in_root(exp53_live, rows)
    reads, pin = [], {}
    for rid in sorted(rows):
        rd = []
        for t in reads_for(rows[rid]["questions"]):
            x = dict(live[rid][t])
            x["value"] = fmt(x["decoded"]) if x["valid"] else "unreadable"
            rd.append(x)
        reads.append({"id": rid, "reads": rd, "read_seconds": 0.1, "request_seconds": 0.2})
    (out / "reads.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reads))
    m.merge_register(str(out), "reads", {"reads.jsonl": e54.sha(str(out / "reads.jsonl"))})
    old = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, m.E55, "inputs_P.jsonl")))}
    if inputs_change:
        inputs_change(old)
    (out / "inputs_P.jsonl").write_text("".join(json.dumps(old[k]) + "\n" for k in sorted(old)))
    m.merge_register(str(out), "inputs", {"inputs_P.jsonl": e54.sha(str(out / "inputs_P.jsonl"))})
    lines = [json.loads(x) for x in open(os.path.join(ROOT, m.E55, "details_P.jsonl"))]
    ident = dict(lines[0]["identity"], start_sha256=e54.sha(str(out / "START.json")), inputs_sha256=e54.sha(str(out / "inputs_P.jsonl")),
                 reading_bundle=st["bundle"]["reading.json"], pictures=st["pictures"]["pictures.json"])
    (out / "details_P.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [{"identity": ident}] + lines[1:]))
    m.register(str(out), "decide_P", {"details_P.jsonl": e54.sha(str(out / "details_P.jsonl"))})
    return str(out), st


def test_reads_integrity_and_decision_pairing(tmp_path, rows):
    out, st = fake_run(tmp_path, rows)
    live = in_root(m.check_reads, out, rows, st)
    fid = in_root(m.read_fidelity, rows, live)
    dec = in_root(m.decision_fidelity, out, rows, fid.pop("_same_pictures"))
    assert fid["reproduced"] and dec["reproduced"] and dec["requests_same_table"] == 600 and dec["max_probability_difference"] == 0.0


def rewrite_reads(out, change):
    p = os.path.join(out, "reads.jsonl")
    lines = [json.loads(x) for x in open(p)]
    change(lines)
    open(p, "w").write("".join(json.dumps(x) + "\n" for x in lines))
    m.merge_register(out, "reads", {"reads.jsonl": e54.sha(p)})                # re-registered: the content checks act


@pytest.mark.parametrize("kind", ["decoded", "duplicate", "missing_read", "unregistered", "bundle_in_details"])
def test_reads_refusals(tmp_path, rows, kind):
    out, st = fake_run(tmp_path, rows)
    if kind == "decoded":                                              # one legal decoded string changed
        rewrite_reads(out, lambda L: L[0]["reads"][0].update(decoded="12345" if L[0]["reads"][0]["decoded"] != "12345" else "54321"))
    elif kind == "duplicate":
        rewrite_reads(out, lambda L: L.append(L[0]))
    elif kind == "missing_read":
        rewrite_reads(out, lambda L: L[0]["reads"].pop())
    elif kind == "unregistered":
        open(os.path.join(out, "reads.jsonl"), "a").write("\n")
    else:
        st = dict(st, bundle=dict(st["bundle"], **{"reading.json": "0" * 64}))   # P details about another bundle
    with pytest.raises(SystemExit):
        in_root(m.check_reads, out, rows, st)


def test_no_pairs_or_inconsistent_serialisation_is_not_reproduced(tmp_path, rows):
    def header(old):
        for v in old.values():
            v["state"] = v["state"].replace("Values read from the picture", "Values read")
    out, st = fake_run(tmp_path / "a", rows, header)
    same = sorted(rows)
    d = in_root(m.decision_fidelity, out, rows, same)
    assert not d["reproduced"] and d["questions_paired"] == 0 and d["reason"] == "no paired request"

    def one(old):
        k = sorted(old)[0]
        old[k]["state"] = old[k]["state"] + " "
    out2, _ = fake_run(tmp_path / "b", rows, one)
    d2 = in_root(m.decision_fidelity, out2, rows, same)
    assert not d2["reproduced"] and d2["same_reading_other_state"] == 1


def test_combination_and_pictures_are_checked(tmp_path, rows):
    out, st = fake_run(tmp_path, rows)
    with pytest.raises(SystemExit, match="pictures differ"):
        in_root(m.check_pictures, out)                                   # the recorded hashes are not the files'
    open(os.path.join(out, "bundle", "head.pt"), "a").write("x")
    with pytest.raises(SystemExit, match="bundle head.pt"):
        m.check_combination(out, st, code=False)


STAGE = """import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'])
open('runs/exp56/child.pid', 'w').write(str(child.pid))
time.sleep(300)
"""
STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp56.py start"*) exit 0;;
  *"exp56.py live"*) exec python3 "${STAGE}";;
  *"exp56.py report --finalize"*) echo final > runs/exp56/report.txt; exit 0;;
esac
exit 9
"""


def test_run_stops_at_the_watch_deadline_and_finalizes(tmp_path):
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp56/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    (tmp_path / "stage.py").write_text(STAGE)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "STAGE": str(tmp_path / "stage.py"),
           "EXP56_LIMIT": "60", "EXP56_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp56/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 124 and time.time() - t < 60, p.stdout + p.stderr
    assert "report --finalize" in (tmp_path / "calls").read_text()
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "runs/exp56/child.pid").read_text()), 0)


def test_live_refuses_a_replaced_component_before_loading(tmp_path, rows, monkeypatch):
    out, st = fake_run(tmp_path, rows)
    open(os.path.join(out, "bundle", "selector.pt"), "a").write("x")   # a replaced component, START still the old one
    import devision.reading as R
    monkeypatch.setattr(R.Reading, "load", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(AssertionError("loaded"))))
    with pytest.raises(SystemExit, match="bundle selector.pt"):
        in_root(m.stage_live, out)
