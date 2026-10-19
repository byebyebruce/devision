"""PANEL-BIND-PREP-01 probe, CPU, no real model: the span parser on the real tokenizer over all 1,000 validation captions
(and its refusals); the random mode equals align._evaluate exactly on a tiny model (token counts recompose the
total); the numbers mode masks exactly both spans; report statistics and statuses; the runner's order / one start."""
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name):
    spec = importlib.util.spec_from_file_location("pb_" + name, os.path.join(HERE, name + ".py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def real_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, "runs/v04-k/cand/tokenizer"))


def test_every_validation_caption_parses_exactly():
    p = load("probe")
    tok = real_tok()
    rows = [json.loads(x) for x in open(os.path.join(ROOT, p.VAL))]
    for r in rows:
        ids, pos = p.number_tokens(tok, r["captions"][0])
        for e in ("A", "B"):
            assert tok.decode([ids[i] for i in pos[e]]).strip() == p.spans(r["captions"][0])[e][2]
    assert len(rows) == 1000


@pytest.mark.parametrize("cap", ["A: speed 5 m/s B: speed 6 m/s.", "B: speed 5 m/s. A: speed 6 m/s.", "A: speed x m/s. B: speed 6 m/s."])
def test_refusals(cap):
    p = load("probe")
    with pytest.raises(ValueError):
        p.number_tokens(real_tok(), cap)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from conftest import tiny_decider
    from PIL import Image
    p = load("probe")
    d = tiny_decider(seed=3)
    os.makedirs(tmp_path / "data" / "img")
    rows = []
    for i in range(20):
        Image.new("RGB", (40, 30), (12 * i, 200 - 9 * i, 90)).save(tmp_path / "data" / "img" / ("%d.png" % i))
        rows.append({"id": "r%d" % i, "family": "f%d" % i, "image": "img/%d.png" % i,
                     "captions": ["A: red %d blue. B: red %d blue." % (i % 3 + 1, (i + 1) % 3 + 1)]})
    monkeypatch.chdir(tmp_path)
    from devision.train.rlcd import _Images
    from devision.model import text_encoder
    from devision.train.align import _MLMHead
    head = _MLMHead(text_encoder(d.model.decision))
    d.model.eval()
    return p, d, head, _Images("data", d), torch.device("cpu"), rows


def test_random_mode_is_align_evaluate(tiny):
    p, d, head, images, dev, rows = tiny
    from devision.train.align import AlignConfig, _evaluate
    want = _evaluate(d.model, head, d.tok, rows, images, AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0), dev)
    got = p.run_random(d, head, images, dev, rows)["summary"]
    for m in ("real", "shuffled"):
        assert got[m]["all"]["accuracy"] == want["val/mlm_acc_" + m]
        assert got[m]["number"]["n"] + got[m]["other"]["n"] == got[m]["all"]["n"]
        assert got[m]["number"]["right"] + got[m]["other"]["right"] == got[m]["all"]["right"]


def test_numbers_mode_masks_both_spans(tiny):
    p, d, head, images, dev, rows = tiny
    out = p.run_numbers(d, head, images, dev, rows)["rows"]
    assert len(out) == 2 * len(rows)
    for r in out:
        ids, pos = p.number_tokens(d.tok, next(x for x in rows if x["id"] == r["id"])["captions"][0])
        assert [len(r["token_right"][e]) for e in ("A", "B")] == [len(pos["A"]), len(pos["B"])]
    real = {r["id"]: r["picture"] for r in out if r["mode"] == "real"}
    wrong = {r["id"]: r["picture"] for r in out if r["mode"] == "shuffled"}
    assert all(real[k] != wrong[k] for k in real) and set(wrong.values()) == set(real.values())     # a fixed cyclic map


def test_number_stats():
    rep = load("report")
    rows = [{"mode": "real", "gold": {"A": "5", "B": "7"}, "pred": {"A": "7", "B": "5"}, "exact": {"A": False, "B": False},
             "token_right": {"A": [False], "B": [False]}},
            {"mode": "real", "gold": {"A": "5", "B": "7"}, "pred": {"A": "5", "B": "5"}, "exact": {"A": True, "B": False},
             "token_right": {"A": [True], "B": [False]}},
            {"mode": "real", "gold": {"A": "4", "B": "4"}, "pred": {"A": "4", "B": "4"}, "exact": {"A": True, "B": True},
             "token_right": {"A": [True], "B": [True]}}]
    s = rep.number_stats(rows, "real")
    assert s["different"]["rows"] == 2 and s["different"]["swapped"] == 0.5 and s["different"]["same_prediction"] == 0.5
    assert s["different"]["both_exact"] == 0 and s["different"]["wrong_end_is_other_gold"] == 1.0 and s["equal"]["both_exact"] == 1.0
    assert s["number_token_accuracy"] == 0.5


def test_report_statuses(tmp_path):
    rep = load("report")
    st = {"val": {"rows": 1}, "tolerance": 0.003, "references": {n: {"real": 0.9, "shuffled": 0.6} for n in ("start", "d1", "exp26")}}
    os.makedirs(tmp_path / "out")
    json.dump(st, open(tmp_path / "out" / "START.json", "w"))
    row = lambda mode, ok: {"id": "r0", "family": "f0", "mode": mode, "gold": {"A": "5", "B": "7"}, "pred": {"A": "5", "B": "7"} if ok else {"A": "7", "B": "5"},
                            "exact": {"A": ok, "B": ok}, "token_right": {"A": [ok], "B": [ok]}, "quantity_same_as_shown": True}
    for name, shuffled in (("start", 0.6), ("d1", 0.6)):
        os.makedirs(tmp_path / "out" / name)
        json.dump({"random": {"real": {"all": {"accuracy": 0.9}}, "shuffled": {"all": {"accuracy": shuffled}}}}, open(tmp_path / "out" / name / "summary.json", "w"))
        open(tmp_path / "out" / name / "numbers.jsonl", "w").write(json.dumps(row("real", True)) + "\n" + json.dumps(row("shuffled", False)) + "\n")
    res = rep.report(str(tmp_path / "out"))
    assert res["endpoints"]["exp26"]["status"] == "missing" and res["status"] == "incomplete"
    assert res["endpoints"]["d1"]["status"] == "completed" and res["endpoints"]["d1"]["true_minus_wrong_both_exact"]["value"] == 1.0
    json.dump({"random": {"real": {"all": {"accuracy": 0.9}}, "shuffled": {"all": {"accuracy": 0.61}}}}, open(tmp_path / "out" / "start" / "summary.json", "w"))
    res = rep.report(str(tmp_path / "out"))
    assert res["endpoints"]["start"]["status"].startswith("invalid") and res["status"] == "invalid"


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
exit 0
"""


def test_runner(tmp_path):
    root = tmp_path / "repo"
    for rel in ("scripts/panel_bind/run.sh", "scripts/panel_bind/probe.py", "scripts/panel_bind/report.py", "src/devision/train/align.py",
                "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n"}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "BIND_LIMIT": "60", "BIND_RESERVE": "4"}
    p = subprocess.run(["bash", str(root / "scripts/panel_bind/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    keys = ["report.py start", "runs/v04-k/cand", "runs/exp74-panel-d1/d1", "runs/exp26-align/stage1", "report.py report"]
    pos = [next(i for i, c in enumerate(calls) if k in c) for k in keys]
    assert pos == sorted(pos)
    p2 = subprocess.run(["bash", str(root / "scripts/panel_bind/run.sh")], env=env, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    s = open(os.path.join(HERE, "run.sh")).read()
    assert "BIND_LIMIT:-1200" in s and "BIND_RESERVE:-120" in s
