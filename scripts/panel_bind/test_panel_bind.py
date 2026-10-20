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
    assert s["different"]["both_exact_all_tokens"] == 0 and s["different"]["wrong_end_is_other_gold"] == 1.0
    assert s["equal"]["both_exact_all_tokens"] == 1.0 and s["A_string_equal_descriptive"] == 2 / 3
    assert s["number_token_accuracy"] == 0.5


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A frozen 20-row question list, its START, and a writer of consistent end-point records."""
    rep = load("report")
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    rows = [{"id": "r%02d" % i, "family": "f%02d" % i, "image": "img/%02d.png" % i,
             "captions": ["A: %s %d m. B: %s %d m." % ("speed" if i % 2 else "length", 10 + i, "speed" if i % 2 else "length", 10 + (i * 7) % 5)]}
            for i in range(20)]
    open("data/val.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    exp, wrong = rep.expected_rows({"val": {"file": "data/val.jsonl"}})

    def records(name, right=0.9):
        rnd, nums = [], []
        for k, r in enumerate(rows):
            for mode in ("real", "shuffled"):
                ok = (k % 10) < (right * 10 if mode == "real" else 3)
                e = exp[r["id"]]
                rnd.append({"id": r["id"], "family": r["family"], "mode": mode, "picture": e[mode], "tokens": {"number": [int(ok), 1], "other": [2, 3]}})
                pred = dict(e["gold"]) if ok else {"A": e["gold"]["B"], "B": e["gold"]["A"]}
                tr = {x: [pred[x] == e["gold"][x]] for x in ("A", "B")}
                nums.append({"id": r["id"], "family": r["family"], "mode": mode, "picture": e[mode], "gold": e["gold"], "pred": pred,
                             "token_right": tr, "exact": {x: all(tr[x]) for x in ("A", "B")}, "quantity_same_as_shown": True})
        cnt = {m: {k: [0, 0] for k in ("number", "other", "all")} for m in ("real", "shuffled")}
        for x in rnd:
            for k in ("number", "other"):
                for kk in (k, "all"):
                    cnt[x["mode"]][kk][0] += x["tokens"][k][0]
                    cnt[x["mode"]][kk][1] += x["tokens"][k][1]
        summ = {"random": {m: {k: {"right": v[0], "n": v[1], "accuracy": v[0] / v[1]} for k, v in c.items()} for m, c in cnt.items()}}
        d = os.path.join("out", name)
        os.makedirs(d, exist_ok=True)
        json.dump(summ, open(os.path.join(d, "summary.json"), "w"))
        for f, rs in (("random", rnd), ("numbers", nums)):
            open(os.path.join(d, f + ".jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in rs))
        return {m: summ["random"][m]["all"]["accuracy"] for m in ("real", "shuffled")}

    refs = {name: records(name, right) for name, right in (("start", 0.3), ("d1", 0.9), ("exp26", 0.9))}
    st = {"val": {"file": "data/val.jsonl", "sha256": rep.sha("data/val.jsonl"), "rows": 20}, "wrong_picture": wrong, "tolerance": 0.003,
          "references": refs}
    json.dump(st, open("out/START.json", "w"))
    return rep


def edit(path, fn):
    rs = [json.loads(x) for x in open(path)]
    rs = fn(rs)
    open(path, "w").write("".join(json.dumps(x) + "\n" for x in rs))


def test_complete_records_complete(world):
    res = world.report("out")
    assert res["status"] == "completed" and all(e["status"] == "completed" for e in res["endpoints"].values())
    assert res["d1_minus_start_both_exact"]["value"] > 0 and json.load(open("out/report.json"))["status"] == "completed"


@pytest.mark.parametrize("case,where,want", [
    ("missing_random", "d1", "incomplete"),
    ("shuffled_dup", "d1", "invalid"),            # one shuffled row replaced by a copy of another: same length
    ("wrong_picture", "d1", "invalid"), ("wrong_gold", "d1", "invalid"), ("summary_mismatch", "d1", "invalid"),
    ("exact_inconsistent", "d1", "invalid"), ("truncated_summary", "exp26", "invalid"), ("truncated_numbers", "exp26", "invalid")])
def test_bad_records_never_complete(world, case, where, want):
    d = os.path.join("out", where)
    if case == "missing_random":
        os.remove(os.path.join(d, "random.jsonl"))
    elif case == "shuffled_dup":
        def f(rs):
            sh = [k for k, r in enumerate(rs) if r["mode"] == "shuffled"]
            rs[sh[1]] = dict(rs[sh[0]])
            return rs
        edit(os.path.join(d, "numbers.jsonl"), f)
    elif case == "wrong_picture":
        edit(os.path.join(d, "random.jsonl"), lambda rs: [dict(r, picture="img/00.png") if r["mode"] == "shuffled" and r["id"] == "r05" else r for r in rs])
    elif case == "wrong_gold":
        edit(os.path.join(d, "numbers.jsonl"), lambda rs: [dict(r, gold={"A": "1", "B": "2"}) if r["id"] == "r03" else r for r in rs])
    elif case == "summary_mismatch":
        s = json.load(open(os.path.join(d, "summary.json")))
        s["random"]["real"]["number"]["right"] += 1
        json.dump(s, open(os.path.join(d, "summary.json"), "w"))
    elif case == "exact_inconsistent":
        edit(os.path.join(d, "numbers.jsonl"), lambda rs: [dict(r, exact={"A": True, "B": True}) if r["id"] == "r09" and r["mode"] == "real" else r for r in rs])
    elif case == "truncated_summary":
        t = open(os.path.join(d, "summary.json")).read()
        open(os.path.join(d, "summary.json"), "w").write(t[: len(t) // 2])
    else:
        t = open(os.path.join(d, "numbers.jsonl")).read()
        open(os.path.join(d, "numbers.jsonl"), "w").write(t[: len(t) // 2])
    res = world.report("out")
    assert res["endpoints"][where]["status"].startswith(want) and res["status"] != "completed", res["endpoints"][where]
    others = [n for n in ("start", "d1", "exp26") if n != where]
    assert all(res["endpoints"][n]["status"] == "completed" for n in others)          # the other end points are kept
    if where == "d1":
        assert "d1_minus_start_both_exact" not in res                                 # an unchecked end point is never compared
    else:
        assert "d1_minus_start_both_exact" in res and "d1_minus_exp26_both_exact" not in res
    assert json.load(open("out/report.json"))["status"] == res["status"]


@pytest.mark.parametrize("case,want", [("missing", "incomplete"), ("truncated", "invalid")])
def test_start_problems_still_write_a_report(world, case, want):
    if case == "missing":
        os.remove("out/START.json")
    else:
        t = open("out/START.json").read()
        open("out/START.json", "w").write(t[:10])
    res = world.report("out")
    assert res["status"] == want and "no conclusion" in res["reason"] or res["status"] == want
    assert json.load(open("out/report.json"))["status"] == want and not res["endpoints"]


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
