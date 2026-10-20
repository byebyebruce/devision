"""PANEL-NUMBER-PREP-01, CPU only, no real model: align's number mask on EXP26 panel captions (every training /
validation caption, real tokenizer; only the two numbers are targets; old real-layout captions unchanged); the two
arm configs differ only in name / stage / number_mask_sources; a tiny alignment with the panel number mask runs and
supervises only numbers; the comparison's pre-written outcomes; the runner's order, one start and limits."""
import importlib.util
import json
import os
import random
import shutil
import stat
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def real_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, "runs/exp74-panel-d1/d1/tokenizer"))


def test_every_panel_caption_gives_exactly_its_two_numbers():
    from devision.train.align import AlignConfig, _masked_batch, _number_rows, number_spans
    probe = load("scripts/panel_bind/probe.py", "pb_probe")
    tok = real_tok()
    c = AlignConfig(max_caption_tokens=40, mask_prob=0.5)
    caps = [s["captions"][0] for f in ("data/exp26/align_train.jsonl", "data/exp26/align_val.jsonl") for s in map(json.loads, open(os.path.join(ROOT, f)))]
    assert len(caps) == 21000
    for cap in caps:
        sp = number_spans(cap)
        assert [x["field"] for x in sp] == ["A", "B"] and [x["text"] for x in sp] == [probe.spans(cap)[e][2] for e in ("A", "B")]
    batch = caps[:16]
    ids, att, tgt = _masked_batch(tok, batch, c, random.Random(3))
    _number_rows(tok, batch, list(range(16)), ids, tgt, c)
    for i, cap in enumerate(batch):
        want_ids, pos = probe.number_tokens(tok, cap)
        masked = sorted(pos["A"] + pos["B"])
        assert [t for t in range(len(want_ids)) if tgt[i, 1 + t] != -100] == masked
        assert all(ids[i, 1 + t] == tok.mask_token_id for t in masked)
        assert all(ids[i, 1 + t] == want_ids[t] for t in range(len(want_ids)) if t not in masked)


def test_old_real_layout_captions_and_refusals():
    from devision.train.align import AlignConfig, _masked_batch, _number_rows, number_spans
    temp = ("Sample A: mass of each particle 46 u, average particle speed 1,890 m/s. "
            "Sample B: mass of each particle 12 u, average particle speed 980 m/s.")
    assert [x["field"] for x in number_spans(temp)] == ["mass", "speed", "mass", "speed"]           # unchanged
    tok = real_tok()
    c = AlignConfig(max_caption_tokens=40, mask_prob=0.5)
    for bad in ("A: speed m/s. B: speed 6 m/s.", "A batter swings at the ball."):
        ids, att, tgt = _masked_batch(tok, [bad], c, random.Random(0))
        with pytest.raises(ValueError, match="number mask"):
            _number_rows(tok, [bad], [0], ids, tgt, c)


def test_the_arms_differ_only_in_the_number_mask():
    n, r = (yaml.safe_load(open(os.path.join(ROOT, "configs/exp75-panel-%s.yaml" % a))) for a in ("n", "r"))
    assert n["stages"][0]["params"].pop("number_mask_sources") == "exp26-panels"
    for x in (n, r):
        x.pop("name")
        x["stages"][0].pop("name")
    assert n == r
    d1 = yaml.safe_load(open(os.path.join(ROOT, "configs/exp74-panel-d1.yaml")))
    assert r["stages"][0]["params"] == d1["stages"][0]["params"] and r["stages"][0]["data"] == d1["stages"][0]["data"]
    assert r["stages"][0]["init"] == "runs/exp74-panel-d1/d1"


def test_a_tiny_panel_number_alignment_runs(tmp_path):
    from conftest import tiny_decider
    from PIL import Image
    from devision.train.align import AlignConfig, align
    os.makedirs(tmp_path / "img")
    caps = []
    for i in range(12):
        Image.new("RGB", (40, 30), (20 * i, 90, 200 - 15 * i)).save(tmp_path / "img" / ("%d.png" % i))
        caps.append({"id": "c%d" % i, "source": "exp26-panels", "image_id": "c%d" % i, "image": "img/%d.png" % i,
                     "captions": ["A: red %d blue. B: red %d blue." % (i % 3 + 1, (i + 1) % 3 + 1)]})
    rep = align(tiny_decider(seed=1), caps[:8], data_root=tmp_path, out_dir=tmp_path / "n", val_samples=caps[8:],
                config=AlignConfig(epochs=2, batch=4, lr=1e-2, warmup=1, mlm_head="", eval_every=0, lora_r=4, lr_lora=1e-2, seed=0,
                                   device="cpu", number_mask_sources="exp26-panels"))
    assert len(rep["losses"]) == 4 and all(x == x for x in rep["losses"])


@pytest.fixture
def cmp(tmp_path, monkeypatch):
    m = load("scripts/panel_number/compare.py", "pn_compare")
    monkeypatch.chdir(tmp_path)

    def rows(both):
        out = []
        for k in range(100):
            ok = k < both * 100
            for mode in ("real", "shuffled"):
                good = ok and mode == "real"
                out.append({"id": "r%d" % k, "family": "f%d" % k, "mode": mode, "gold": {"A": "1", "B": "2"},
                            "exact": {"A": good, "B": good}})
        return out

    def setup(n=None, r=None, d1=0.234, exp26=0.693, tamper=None, arm_status=None):
        if arm_status:
            monkeypatch.setattr(m, "ARM_STATUS", "arms.json")
            json.dump(arm_status, open("arms.json", "w"))
        ref = {"d1": m.both_exact(rows(d1)), "exp26": m.both_exact(rows(exp26))}
        eps = {}
        for a, v in (("n", n), ("r", r)):
            if v is None:
                eps[a] = {"status": "missing"}
                continue
            os.makedirs(os.path.join("out", a), exist_ok=True)
            open(os.path.join("out", a, "numbers.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in rows(v)))
            eps[a] = {"status": "completed", "true_minus_wrong_both_exact": {"value": v, "ci95": [v - 0.05, v + 0.05]},
                      "numbers": {"real": {}, "shuffled": {}}}

        class Rep:
            boot = staticmethod(m.bind().boot)

            @staticmethod
            def report(out):
                return {"status": "completed" if all(e["status"] == "completed" for e in eps.values()) else "incomplete", "endpoints": eps}
        monkeypatch.setattr(m, "bind", lambda: Rep)
        monkeypatch.setattr(m, "fixed", lambda rep: ref)
        os.makedirs("out", exist_ok=True)
        json.dump({"val": {"sha256": m.APPROVED["val"][1], "pictures_sha256": m.APPROVED["pictures"]}}, open("out/START.json", "w"))
        open("old_d1.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in rows(d1)))
        monkeypatch.setattr(m, "FIXED", {"d1": ("old_d1.jsonl", "-"), "exp26": ("old_d1.jsonl", "-")})
        if tamper:
            tamper(eps)
        return m.compare("out")
    return setup


def test_same_family_other_gold_or_picture_is_never_compared(cmp):
    """R1-01: an arm read on a caption / gold or picture that is not the frozen one gets no comparison."""
    def change_gold(eps):
        rs = [json.loads(x) for x in open("out/n/numbers.jsonl")]
        rs[0]["gold"] = {"A": "9", "B": "2"}
        open("out/n/numbers.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in rs))
    res = cmp(n=0.80, r=0.40, tamper=change_gold)
    assert res["arms"]["n"]["status"].startswith("invalid") and res["outcome"].startswith("incomplete") and "n_minus_r" not in res
    assert res["arms"]["r"]["status"] == "completed" and "both_exact" in res["arms"]["r"]           # the good arm is kept


def test_another_start_list_is_never_compared(cmp, monkeypatch):
    def other_start(eps):
        json.dump({"val": {"sha256": "0" * 64, "pictures_sha256": "1" * 64}}, open("out/START.json", "w"))
    res = cmp(n=0.80, r=0.40, tamper=other_start)
    assert all(a["status"].startswith("invalid") for a in res["arms"].values()) and res["outcome"].startswith("incomplete")


def test_a_bad_arm_keeps_its_state_and_the_other_arm(cmp):
    """R1-02: R's training report was truncated (no START, no probe); N is read; no attribution."""
    res = cmp(n=0.80, r=None, arm_status={"n": {"valid": True, "state": "trained"}, "r": {"valid": False, "state": "invalid: align_report.json unreadable"}})
    assert res["arms"]["r"]["status"].startswith("invalid: align_report") and res["arms"]["n"]["worth_decision_diagnostic"]
    assert res["outcome"].startswith("incomplete") and "n_minus_r" not in res


@pytest.mark.parametrize("case,want", [("missing", "missing"), ("truncated", "invalid"), ("step625", "incomplete"),
                                       ("last625", "invalid"), ("nan", "invalid"), ("good", "trained")])
def test_arm_training_reports(tmp_path, case, want):
    m = load("scripts/panel_number/compare.py", "pn_c2")
    p = tmp_path / "align_report.json"
    good = {"losses": [0.5] * 1250, "val": [{"step": 0, "val/mlm_acc_real": 0.9, "val/mlm_acc_shuffled": 0.6},
                                           {"step": 1250, "val/mlm_acc_real": 0.95, "val/mlm_acc_shuffled": 0.6}]}
    if case != "missing":
        r = json.loads(json.dumps(good))
        if case == "step625":
            r["losses"] = r["losses"][:625]
        if case == "last625":
            r["val"][-1]["step"] = 625
        if case == "nan":
            r["val"][-1]["val/mlm_acc_real"] = float("nan")
        text = json.dumps(r)
        p.write_text(text[: len(text) // 2] if case == "truncated" else text)
    ok, state = m.arm_report(str(p))
    assert ok == (want == "trained") and state.startswith(want)


def test_start_takes_only_valid_arms(tmp_path, monkeypatch):
    m = load("scripts/panel_number/compare.py", "pn_c3")
    monkeypatch.chdir(tmp_path)
    good = {"losses": [0.5] * 1250, "val": [{"step": 1250, "val/mlm_acc_real": 0.95, "val/mlm_acc_shuffled": 0.6}]}
    for a, text in (("n", json.dumps(good)), ("r", json.dumps(good)[:30])):
        os.makedirs("arm_" + a)
        open("arm_%s/align_report.json" % a, "w").write(text)
        open("arm_%s/model.safetensors" % a, "w").write("w")
    monkeypatch.setattr(m, "ARMS", {"n": "arm_n", "r": "arm_r"})
    monkeypatch.setattr(m, "REFS", {"n": ("arm_n/align_report.json", -1), "r": ("arm_r/align_report.json", -1)})
    monkeypatch.setattr(m, "ARM_STATUS", "arms.json")
    seen = {}

    class Rep:
        @staticmethod
        def start(out, endpoints, refs):
            seen.update(endpoints=endpoints, refs=refs)
            return {"endpoints": endpoints}
    monkeypatch.setattr(m, "bind", lambda: Rep)
    assert m.main(["start"]) == 0
    assert seen["endpoints"] == {"n": "arm_n"} and set(seen["refs"]) == {"n"}
    st = json.load(open("arms.json"))
    assert st["n"]["valid"] and not st["r"]["valid"] and st["r"]["state"].startswith("invalid")


def test_preflight_on_the_real_files(tmp_path, monkeypatch):
    m = load("scripts/panel_number/compare.py", "pn_c4")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(m, "PREFLIGHT", str(tmp_path / "preflight.json"))
    res = m.preflight()
    assert res["ok"], res["problems"]


@pytest.mark.parametrize("case", ["caption", "picture", "d1_config", "bind_start"])
def test_preflight_refusals(tmp_path, monkeypatch, case):
    """R1-01: a changed caption (same family), changed picture bytes, a foreign D1 component or another panel-bind
    START is refused before anything is trained."""
    m = load("scripts/panel_number/compare.py", "pn_c5_" + case)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/img")
    os.makedirs("d1/tokenizer")
    open("d1/model.safetensors", "w").write("m")
    open("d1/config.json", "w").write("{}")
    open("d1/tokenizer/t.json", "w").write("t")
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save("data/img/0.png")
    open("data/val.jsonl", "w").write(json.dumps({"id": "v0", "family": "f0", "image": "img/0.png", "captions": ["A: x 1 u. B: x 2 u."]}) + "\n")
    open("data/train.jsonl", "w").write("t\n")
    rep = m.bind()
    d1 = {"path": "d1", "model.safetensors": m.sha("d1/model.safetensors"), "config.json": m.sha("d1/config.json"), "tokenizer": rep.tree("d1/tokenizer")}
    pics = m.pictures_digest("data/val.jsonl")
    json.dump({"val": {"sha256": m.sha("data/val.jsonl"), "pictures_sha256": pics}, "endpoints": {"d1": d1}}, open("bind_start.json", "w"))
    monkeypatch.setattr(m, "APPROVED", {"d1": d1, "train": ("data/train.jsonl", m.sha("data/train.jsonl")),
                                        "val": ("data/val.jsonl", m.sha("data/val.jsonl")), "pictures": pics,
                                        "bind_start": ("bind_start.json", m.sha("bind_start.json"))})
    monkeypatch.setattr(m, "fixed", lambda rep: {})
    monkeypatch.setattr(m, "PREFLIGHT", "preflight.json")
    assert m.preflight()["ok"]
    if case == "caption":
        open("data/val.jsonl", "w").write(json.dumps({"id": "v0", "family": "f0", "image": "img/0.png", "captions": ["A: x 7 u. B: x 2 u."]}) + "\n")
    elif case == "picture":
        Image.new("RGB", (8, 8), "blue").save("data/img/0.png")
    elif case == "d1_config":
        open("d1/config.json", "w").write('{"other": 1}')
    else:
        open("bind_start.json", "a").write(" ")
    res = m.preflight()
    assert not res["ok"] and res["problems"] and json.load(open("preflight.json"))["ok"] is False


@pytest.mark.parametrize("n,r,want", [(0.40, 0.35, "stop"), (0.40, 0.70, "R only"), (0.68, 0.62, "N, not attributed"),
                                      (0.80, 0.40, "N attributed"), (0.80, None, "incomplete")])
def test_outcomes(cmp, n, r, want):
    assert cmp(n=n, r=r)["outcome"].startswith(want)


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"compare.py preflight"*) [ -n "${PREFLIGHT_FAIL:-}" ] && exit 12; exit 0;;
  *"compare.py start"*) mkdir -p runs/panel-number; echo '{"endpoints": {"n": {"path": "runs/exp75-panel-n/n"}}}' > runs/panel-number/START.json; exit 0;;
esac
exit 0
"""


def test_runner(tmp_path):
    root = tmp_path / "repo"
    files = ("scripts/panel_number/run.sh", "scripts/panel_number/compare.py", "scripts/panel_bind/probe.py", "scripts/panel_bind/report.py",
             "configs/exp75-panel-n.yaml", "configs/exp75-panel-r.yaml", "src/devision/train/align.py", "scripts/exp35/guard.sh",
             "scripts/exp38/steps.sh")
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    for rel in ("data/exp26/align_train.jsonl", "data/exp26/align_val.jsonl", "runs/exp74-panel-d1/d1/model.safetensors"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    for arm in ("n", "r"):
        (root / ("runs/exp75-panel-%s/%s" % (arm, arm))).mkdir(parents=True)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "EXP75_LIMIT": "60", "EXP75_RESERVE": "4"}
    p = subprocess.run(["bash", str(root / "scripts/panel_number/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1 and "exists (fresh only)" in p.stdout                         # an arm directory already there
    for arm in ("n", "r"):
        shutil.rmtree(root / ("runs/exp75-panel-%s" % arm))
    p = subprocess.run(["bash", str(root / "scripts/panel_number/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    keys = ["compare.py preflight", "exp75-panel-n.yaml", "exp75-panel-r.yaml", "compare.py start", "runs/exp75-panel-n/n", "compare.py report"]
    pos = [next(i for i, c in enumerate(calls) if k in c) for k in keys]
    assert pos == sorted(pos)
    assert sum("probe.py" in c for c in calls) == 1 and not any("exp75-panel-r/r" in c and "probe.py" in c for c in calls)   # only START's arms
    p2 = subprocess.run(["bash", str(root / "scripts/panel_number/run.sh")], env=env, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    shutil.rmtree(root / "runs")
    (tmp_path / "calls").unlink()
    p3 = subprocess.run(["bash", str(root / "scripts/panel_number/run.sh")], env={**env, "PREFLIGHT_FAIL": "1"}, capture_output=True, text=True, timeout=60)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert p3.returncode == 12 and "nothing trained" in p3.stdout and not any("devision-pipeline" in c for c in calls)
    s = open(os.path.join(HERE, "run.sh")).read()
    assert "EXP75_LIMIT:-7200" in s and "EXP75_RESERVE:-120" in s
