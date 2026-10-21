# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""exp64 without a model: the 3,552 training records carry EXP62's registered new boxes and their own targets, no dev
record; the plan keeps the pairs whole (1,000 x 64, 20 exposures per pair); the head trains from the given state only;
the old-head reproduction comparison; the gates / branch (head validity first, missing never a pass); START binds the
data and EXP62's run; the run's stage order and bounded finalize."""
import importlib.util
import json
import os

import pytest
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("exp64", os.path.join(HERE, "exp64.py"))
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
has = os.path.exists(os.path.join(ROOT, m.D, "MANIFEST.json")) and os.path.exists(os.path.join(ROOT, m.E62, "START.json"))
real = pytest.mark.skipif(not has, reason="data / runs not here")


@pytest.fixture(autouse=True)
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@real
def test_training_records_and_plan():
    recs = m.jsonl(os.path.join(m.D, "train.jsonl"))
    boxes = m.b64.e62_boxes()
    assert len(recs) == 3552 and {r["set"] for r in recs} == {"fit_O", "fit_L", "train_V", "train_P"}
    assert all(r["box"] == boxes[r["key"]] for r in recs)
    plan = m.jsonl(os.path.join(m.D, "plan.jsonl"))
    assert len(plan) == 1000 and all(len(b) == 64 for b in plan)
    assert all(b[2 * i] == b[2 * i + 1].replace("train_V|", "train_P|", 1) for b in plan[:100] for i in range(24))
    keys = {r["key"] for r in recs}
    assert all(k in keys for b in plan for k in b)
    man = json.load(open(os.path.join(m.D, "MANIFEST.json")))
    assert man["plan"]["parent_exposure_range"] == [20, 20] and man["plan"]["split_pairs"] == 0


class FakeTokens:
    def __init__(self, n=8):
        torch.manual_seed(2)
        self.x = torch.randn(n, 6, 768)

    def batch(self, idx):
        return self.x[idx], torch.ones(len(idx), 6, dtype=torch.bool), torch.zeros(len(idx), 6, 2, dtype=torch.long), torch.rand(len(idx), 6, 4)


def test_head_trains_from_the_given_state_only():
    from devision.reading import CharHead
    torch.manual_seed(0)
    init = CharHead(768).state_dict()
    y = torch.tensor([m.p49.labels(t) for t in ("14", "1300", "2.5", "7", "44", "9", "120", "3")])
    tok = FakeTokens()

    def fresh():
        h = CharHead(768)
        h.load_state_dict(init)
        return h
    same, s0 = m.train_head(fresh(), tok, y, [])
    assert all(torch.equal(same.state_dict()[k], init[k]) for k in init) and s0["exposures"] == 0
    a, st = m.train_head(fresh(), tok, y, [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]])
    b, _ = m.train_head(fresh(), tok, y, [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]])
    assert st["steps"] == 3 and any(not torch.equal(a.state_dict()[k], init[k]) for k in init)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in init) and set(a.state_dict()) == set(init)


def test_old_head_reproduction_compares_strings_and_validity():
    ref = {"q1": {"decoded": "14", "valid": True}, "q2": {"decoded": "13", "valid": False}}
    assert m.same_strings([{"question": "q1", "decoded": "14", "valid": True}, {"question": "q2", "decoded": "13", "valid": False}], ref)
    assert not m.same_strings([{"question": "q1", "decoded": "14", "valid": True}, {"question": "q2", "decoded": "13", "valid": True}], ref)
    assert not m.same_strings([{"question": "q1", "decoded": "14", "valid": True}], ref)                 # a missing read


def stat(exact=0.85, f=0.8):
    return {"exact": exact, "both": 0.7, "twin": 0.7, "real_minus_donor": 0.8, "real_minus_donor_ci": [0.7, 0.9],
            **{"exact:" + x: {"value": f, "n": 600} for x in m.e61.FIELDS}}


def good():
    return {"fit_validity": True, "tune_V": {"auto_new": stat(), "new_minus_old_head": {"ci95": [0.01, 0.05]}},
            "rl_tune": {"new": stat(0.82), "baseline": stat(0.817)}, "dev24": {"speed_new": 19, "speed_new_minus_original": {"ci95": [0.5, 0.9]}}}


@pytest.mark.parametrize("edit,text", [
    (lambda r: None, "head adaptation passes"),
    (lambda r: r.update(fit_validity=False), "did not fit its own training data"),
    (lambda r: r["rl_tune"]["new"].update(exact=0.78), "old-layout regression"),
    (lambda r: r["tune_V"]["auto_new"].update(exact=0.79), "investment line"),
    (lambda r: r["tune_V"]["new_minus_old_head"].update(ci95=[0.0, 0.05]), "not attributed to the head"),
    (lambda r: r["dev24"].update(speed_new=17), "region / coverage gap"),
    (lambda r: r.pop("dev24"), "incomplete"),
    (lambda r: r["tune_V"]["auto_new"].update(twin=None), "incomplete"),
])
def test_gates_and_branch(edit, text):
    r = good()
    edit(r)
    assert text in m.branch(m.gates(r))


@real
def test_start_binds_data_and_exp62(tmp_path, monkeypatch):
    import shutil
    out = str(tmp_path / "out")
    d = tmp_path / "exp64"
    shutil.copytree(os.path.join(ROOT, m.D), d)
    monkeypatch.setattr(m, "D", str(d))
    m.stage_start(0, out=out)
    m.check_start(out, data_root="")
    lines = (d / "train.jsonl").read_text().splitlines()
    r = json.loads(lines[0])
    r["target"] = r["target"] + "1"
    lines[0] = json.dumps(r)
    (d / "train.jsonl").write_text("\n".join(lines) + "\n")
    man = json.load(open(d / "MANIFEST.json"))
    man["train_sha256"] = m.sha(str(d / "train.jsonl"))                           # the current MANIFEST updated too
    json.dump(man, open(d / "MANIFEST.json", "w"))
    with pytest.raises(SystemExit, match="differ from START"):
        m.check_start(out, data_root="")
    assert m.report(out)["branch"].startswith("invalid")
    monkeypatch.setattr(m, "E62_SHA", dict(m.E62_SHA, **{"selector.pt": "0" * 64}))
    with pytest.raises(SystemExit, match="EXP62"):
        m.bound_e62()


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"watch.py"*) shift 2; exec python3 "$@";;
  *"exp64.py start"*) mkdir -p runs/exp64; exit 0;;
  *"exp64.py features"*) exit 0;;
  *"exp64.py train"*) [ -n "${FAIL_TRAIN:-}" ] && exit 6; exit 0;;
  *"exp64.py reads"*) exit 0;;
  *"exp64.py report"*) echo report > runs/exp64/report.txt; exit 0;;
esac
exit 9
"""


def chain(tmp_path, extra):
    import shutil
    import stat as st_
    import subprocess
    root = tmp_path / "repo"
    for rel in ("scripts/exp64/run.sh", "scripts/exp53/watch.py", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | st_.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP64_LIMIT": "60", "EXP64_RESERVE": "4", **extra}
    p = subprocess.run(["bash", str(root / "scripts/exp64/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    return root, p, (tmp_path / "calls").read_text().splitlines()


def test_run_order_and_failed_stage_finalize(tmp_path):
    root, p, calls = chain(tmp_path / "a", {})
    assert p.returncode == 0, p.stdout + p.stderr
    first = lambda s: next(i for i, c in enumerate(calls) if s in c)  # noqa: E731
    assert first("exp64.py start") < first("exp64.py features") < first("exp64.py train") < first("exp64.py reads") < first("exp64.py report")
    root, p, calls = chain(tmp_path / "b", {"FAIL_TRAIN": "1"})
    joined = "\n".join(calls)
    assert p.returncode == 6 and "exp64.py reads" not in joined and "report --finalize" in joined


# ---------------------------------------------------------------- EXP64-R1-01 / 02

def fake_run(tmp_path, data_root="data"):
    """A complete run directory from the real registered references (no model): START, a head, and every read file
    with the old and the new head reading exactly the reference."""
    out = str(tmp_path / "out")
    m.stage_start(0, out=out, data_root=data_root)
    open(os.path.join(out, "head.pt"), "wb").write(b"new head")
    h = m.sha(os.path.join(out, "head.pt"))
    fit = {"%s/%s/%s" % (w, s, f): 1.0 for w in ("old", "new") for s in ("fit_O", "fit_L", "train_V", "train_P") for f in ("all",) + m.e61.FIELDS}
    json.dump({"head_sha256": h, "fit": fit}, open(os.path.join(out, "train.json"), "w"))
    m.register(out, "head", {"head.pt": h, "train.json": m.sha(os.path.join(out, "train.json"))})
    want = m.templates_by_key()
    for group, (load, key, names) in m.REFS.items():
        ref = load()
        idk = "picture" if group.startswith("tune_V") else "id"
        items = [{idk: k, "reads": [dict(ref[k][t]) for t in want[group][k]]} for k in sorted(ref)]
        for n in names:
            m.write_reads(out, n, items, m.HEAD_SHA if n.endswith("_old") else h)
    open(os.path.join(out, "details_dev24_new_head.jsonl"), "w").write(open(os.path.join(m.E62, "details_dev24_new.jsonl")).read())
    return out, h


def edit_reads(out, name, h, fn):
    f = os.path.join(out, name + ".jsonl")
    lines = m.jsonl(f)
    items = fn(lines[1:])
    m.write_reads(out, name, items, m.HEAD_SHA if name.endswith("_old") else h)


@real
def test_complete_run_is_valid_and_each_fault_is_not(tmp_path):
    out, h = fake_run(tmp_path)
    rep = m.report(out)
    assert rep["validity"]["state"] == "valid" and all(g["reproduced"] for g in rep["validity"]["groups"].values())
    assert not rep["branch"].startswith(("invalid", "incomplete"))

    def flip_string(items):
        items[0]["reads"][0]["decoded"] = items[0]["reads"][0]["decoded"] + "1"
        items[0]["reads"][0]["valid"], items[0]["reads"][0]["value"] = False, "unreadable"
        return items
    edit_reads(out, "rl_tune_old", h, flip_string)
    rep = m.report(out)
    assert rep["validity"]["state"] == "invalid" and rep["validity"]["groups"]["rl_tune"]["mismatches"] == 1
    assert rep["branch"].startswith("invalid") and rep["candidate_eligible"] is False


@real
def test_valid_flag_duplicate_template_and_missing_control(tmp_path):
    out, h = fake_run(tmp_path)

    def flip_valid(items):
        r = items[0]["reads"][0]
        r["valid"] = not r["valid"]
        r["value"] = "unreadable" if not r["valid"] else r["decoded"]
        return items
    edit_reads(out, "tune_V_auto_old", h, flip_valid)
    assert m.report(out)["validity"]["state"] == "invalid"
    out2, h2 = fake_run(tmp_path / "b")
    edit_reads(out2, "dev24_auto_new", h2, lambda items: [dict(items[0], reads=items[0]["reads"] + items[0]["reads"][:1])] + items[1:])
    v = m.report(out2)["validity"]
    assert v["state"] == "invalid" and any("dev24_auto_new" in x for x in v["invalid"])
    out3, h3 = fake_run(tmp_path / "c")
    os.remove(os.path.join(out3, "tune_V_oracle_new.jsonl"))
    rep = m.report(out3)
    assert rep["validity"]["state"] == "incomplete" and rep["branch"].startswith("incomplete")
    tj = json.load(open(os.path.join(out3, "train.json")))                     # a failed fit does not mask it
    tj["fit"] = {k: 0.0 for k in tj["fit"]}
    json.dump(tj, open(os.path.join(out3, "train.json"), "w"))
    m.register(out3, "head", {"head.pt": h3, "train.json": m.sha(os.path.join(out3, "train.json"))})
    assert m.report(out3)["branch"].startswith("incomplete")


def test_failed_reproduction_is_on_disk_before_the_stop(tmp_path, monkeypatch):
    out = str(tmp_path)
    open(os.path.join(out, "START.json"), "w").write("{}")
    ref = {"p1": {"t": {"decoded": "14", "valid": True}}}
    results = {"g_old": [{"id": "p1", "reads": [{"question": "t", "decoded": "15", "valid": True, "value": "15"}]}],
               "g_new": [{"id": "p1", "reads": [{"question": "t", "decoded": "14", "valid": True, "value": "14"}]}]}
    with pytest.raises(SystemExit, match="does not reproduce"):
        m.close_group(out, "g", results, ["g_old", "g_new"], ref, lambda x: x["id"], "h")
    rep = json.load(open(os.path.join(out, "repro_g.json")))
    assert rep["reproduced"] is False and rep["mismatches"] == [{"picture": "p1", "question": "t", "expected": ["14", True], "actual": ["15", True]}]
    reg = json.load(open(os.path.join(out, "registry.json")))
    assert {"repro_g", "reads_g_old", "reads_g_new"} <= set(reg)


@real
def test_dev_picture_and_baseline_identity(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "data"
    root.mkdir()
    for name in os.listdir(os.path.join(ROOT, "data")):                        # a mirror of data/ by symlinks ...
        if name != "exp61":
            os.symlink(os.path.join(ROOT, "data", name), root / name)
    (root / "exp61").mkdir()
    for name in os.listdir(os.path.join(ROOT, "data", "exp61")):
        if name != "images":
            os.symlink(os.path.join(ROOT, "data", "exp61", name), root / "exp61" / name)
    (root / "exp61" / "images").mkdir()
    imgs = sorted(os.listdir(os.path.join(ROOT, "data", "exp61", "images")))
    for name in imgs:
        os.symlink(os.path.join(ROOT, "data", "exp61", "images", name), root / "exp61" / "images" / name)
    out = str(tmp_path / "out")
    m.stage_start(0, out=out, data_root=str(root))
    m.check_start(out, data_root=str(root))
    victim = [n for n in imgs if n.startswith("tune_V")][0]                      # ... with one dev picture replaced
    os.remove(root / "exp61" / "images" / victim)
    shutil.copy(os.path.join(ROOT, "data", "exp61", "images", imgs[0]), root / "exp61" / "images" / victim)
    with pytest.raises(SystemExit, match="pictures differ"):
        m.check_start(out, data_root=str(root))
    e56 = tmp_path / "exp56"
    e56.mkdir()
    for f in ("reads.jsonl", "registry.json"):
        shutil.copy(os.path.join(ROOT, m.e61.E56, f), e56 / f)
    (e56 / "reads.jsonl").write_text((e56 / "reads.jsonl").read_text() + "\n")
    m.e54.register(str(e56), "reads", {"reads.jsonl": m.sha(str(e56 / "reads.jsonl"))})   # the current registry updated too
    monkeypatch.setattr(m.e61, "E56", str(e56))
    with pytest.raises(SystemExit, match="EXP56"):
        m.check_start(out, data_root="")
    assert m.report(out)["branch"].startswith("invalid")
