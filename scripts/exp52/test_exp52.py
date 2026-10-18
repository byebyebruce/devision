"""exp52 scripts without the real model: the number sidecar covers every training question once with a strict target;
the readout runs the ordinary forward on a tiny reader checkpoint over real pictures and writes EXP49-format details
whose donor is the twin's own prediction, plus with / without content logits."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def test_sidecar_covers_training_once():
    side = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp52", "numbers.jsonl"))]
    train = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp42", "train.jsonl"))]
    assert len(side) == sum(len(r["questions"]) for r in train) == len({(s["id"], s["qid"]) for s in side}) == 4608
    gold = {(r["id"], q): next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0) for r in train for q in r["questions"]}
    for s in side:
        assert gold[(s["id"], s["qid"])].split(" ")[0].replace(",", "") == s["target"]


def test_readout_smoke(tmp_path):
    import torch
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=4)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    torch.manual_seed(5)
    d.model.add_number_reader()
    d.cfg.number_reader = True
    run = tmp_path / "arm"
    d.save(run / "stage2")
    p = subprocess.run([sys.executable, os.path.join(HERE, "readout.py"), "--run", str(run), "--device", "cpu", "--families", "2"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [json.loads(x) for x in open(run / "eval" / "readout_details.jsonl")]
    rows = lines[1:]
    assert lines[0]["identity"]["path"].startswith("rlcd._forward") and rows
    by = {r["key"]: r for r in rows}
    qs = {q["key"]: q for q in map(json.loads, open(os.path.join(ROOT, "data", "exp49", "questions.jsonl"))) if q["key"] in by}
    for k, r in by.items():
        assert len(r["pred"]) == 5
        if qs[k]["twin_changed"] and qs[k]["twin"] in by:
            assert r["donor"] == by[qs[k]["twin"]]["pred"]                       # the same question on the twin picture
    use = [json.loads(x) for x in open(run / "eval" / "readout_use.jsonl")][1:]
    assert len(use) == len(rows) and all(len(u["on"]) == len(u["off"]) >= 2 for u in use)
    assert any(u["on"] != u["off"] for u in use)                                 # the content is read by the scoring


# ---- critic EXP52-R1-01 / 02

import importlib.util  # noqa: E402

import pytest  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ro = load("exp52_readout", "readout.py")
rep = load("exp52_report", "report.py")


def tiny_reader_decider():
    import torch
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=4)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    torch.manual_seed(5)
    d.model.add_number_reader()
    d.cfg.number_reader = True
    for p in d.model.reader.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return d


def test_one_vision_encoding_per_picture_and_same_outputs():
    import torch
    from laya.common import collate_items
    from PIL import Image
    from devision.infer import image_tensor
    from devision.train import rlcd
    d = tiny_reader_decider()
    m = d.model.eval()
    m.reader.keep_chars = True
    rows = {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, "data/real_layout/questions_tune.jsonl")))}
    r = next(iter(rows.values()))
    items = rlcd._items(d, [r])
    batch = collate_items([items], d.tok.pad_token_id)
    assert batch is not None
    px = image_tensor(Image.open(os.path.join(ROOT, "data", r["image"])), d.cfg)[None]
    calls = []
    real = m.visual_inputs
    m.visual_inputs = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        with torch.no_grad():
            on, chars, off = ro.forward_on_off(m, batch, px, "cpu")
    finally:
        m.visual_inputs = real
    assert len(calls) == 1 and m.reader.off is False                                  # one encoding; switch restored
    index = torch.zeros(len(items), dtype=torch.long)
    with torch.no_grad():
        ref_on = rlcd._forward(m, batch, px, "cpu", index=index).float()
        ref_chars = m.reader.last_chars.argmax(-1).tolist()
        m.reader.off = True
        ref_off = rlcd._forward(m, batch, px, "cpu", index=index).float()
        m.reader.off = False
    assert torch.allclose(on, ref_on, atol=1e-5) and torch.allclose(off, ref_off, atol=1e-5) and chars == ref_chars
    assert not torch.allclose(on, off, atol=1e-6)


@pytest.fixture(scope="module")
def arm(tmp_path_factory):
    """A tiny reader arm as the chain leaves it: stage2, FROZEN, the full readout (2,376 questions), recorded."""
    fz = load("v04b_freeze", "../v04b/freeze.py")
    run = tmp_path_factory.mktemp("arm") / "exp52-x"
    tiny_reader_decider().save(run / "stage2")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        fz.freeze(str(run))
        assert ro.main(["--run", str(run), "--device", "cpu"]) == 0
        ro.record(str(run))
    finally:
        os.chdir(cwd)
    return str(run), fz


def copy_arm(arm, tmp_path):
    import shutil
    run, fz = arm
    dst = str(tmp_path / "exp52-copy")
    shutil.copytree(os.path.join(run, "eval"), os.path.join(dst, "eval"))
    os.makedirs(os.path.join(dst, "stage2"))
    for f in ("model.safetensors", "config.json"):
        os.symlink(os.path.join(run, "stage2", f), os.path.join(dst, "stage2", f))
    shutil.copy(fz.path(run), fz.path(dst))
    return dst, fz


def reregister(run, fz):
    """What re-recording after an edit would write: so the content checks, not only the hashes, are exercised."""
    ev = os.path.join(run, "eval")
    info = json.load(open(os.path.join(ev, "readout.json")))
    info["details_sha256"] = ro.sha(os.path.join(ev, "readout_details.jsonl"))
    info["use_sha256"] = ro.sha(os.path.join(ev, "readout_use.jsonl"))
    json.dump(info, open(os.path.join(ev, "readout.json"), "w"), indent=1)
    fr = json.load(open(fz.path(run)))
    fr["evals"]["readout"] = {f: ro.sha(os.path.join(ev, f)) for f in ro.FILES}
    json.dump(fr, open(fz.path(run), "w"), indent=1)


def edit(path, k, change):
    lines = open(path).read().splitlines()
    row = json.loads(lines[k])
    change(row)
    lines[k] = json.dumps(row)
    open(path, "w").write("\n".join(lines) + "\n")


def evidence(run, fz):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return rep.readout_evidence(run, fz.check(run, [])["weights"], "completed")
    finally:
        os.chdir(cwd)


def test_recorded_readout_passes(arm, tmp_path):
    run, fz = copy_arm(arm, tmp_path)
    nums, use = evidence(run, fz)
    assert nums["state"] == "completed" and use["state"] == "completed" and use["rl_tune"]["n"] == 1800


def test_one_changed_prediction_is_refused(arm, tmp_path):
    run, fz = copy_arm(arm, tmp_path)
    p = os.path.join(run, "eval", "readout_details.jsonl")
    edit(p, 1, lambda r: r.update(pred=[9, 9, 11, -100, -100] if r["pred"][:2] != [9, 9] else [8, 8, 11, -100, -100]))
    nums, use = evidence(run, fz)
    assert nums["state"] == use["state"] == "invalid" and "changed after it was recorded" in nums["reason"]


@pytest.mark.parametrize("kind", ["foreign_use", "wrong_gold", "missing", "duplicate", "nonfinite"])
def test_use_content_is_checked_even_if_rerecorded(arm, tmp_path, kind):
    run, fz = copy_arm(arm, tmp_path)
    p = os.path.join(run, "eval", "readout_use.jsonl")
    if kind == "foreign_use":                                       # another arm's use file: another identity header
        edit(p, 0, lambda r: r["identity"].update({"model.safetensors": "0" * 64}))
    elif kind == "wrong_gold":
        edit(p, 1, lambda r: r.update(gold=(r["gold"] + 1) % len(r["options"])))
    elif kind == "nonfinite":
        edit(p, 1, lambda r: r.update(on=[float("nan")] + r["on"][1:]))
    else:
        lines = open(p).read().splitlines()
        lines = lines[:-1] if kind == "missing" else lines + [lines[-1]]
        open(p, "w").write("\n".join(lines) + "\n")
    reregister(run, fz)
    nums, use = evidence(run, fz)
    assert nums["state"] == use["state"] == "invalid", (kind, nums)


def test_missing_readout_keeps_the_report_going(arm, tmp_path):
    run, fz = copy_arm(arm, tmp_path)
    assert rep.readout_evidence(run, {}, "interrupted")[0] == {"state": "missing", "reason": "readout step interrupted"}
    os.remove(os.path.join(run, "eval", "readout_use.jsonl"))
    assert evidence(run, fz)[1]["state"] == "invalid"
