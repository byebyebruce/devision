"""exp53 without the real model: the oracle box through the selector's box rule gives EXP51's region exactly; the
selector's per-question input is the question text and the picture only (box, target, field, object, qid, twin changes
leave it unchanged; another question changes it and the predicted box); the global features are visual_inputs' first 256
keys / values; training changes only the selector; the frozen head loads strictly and needs no gradient; the box rule's
clipping / degenerate cases; an incomplete run reports missing paths and never an automatic result from the oracle; the
whole run finalizes within its limit."""
import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = load("exp53_stages", "stages.py")
sel = st.sel


@pytest.fixture(scope="module")
def qs():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        q = st.questions()
        st.load_texts(q)
        return q
    finally:
        os.chdir(cwd)


def test_oracle_box_through_the_box_rule_is_exp51s_region(qs):
    oracle = json.load(gzip.open(os.path.join(ROOT, st.ORACLE_REGIONS), "rt"))
    for i in range(0, len(qs), 47):
        q = qs[i]
        px, bad = sel.pixel_box(sel.target_of(q["box"], q["width"], q["height"]), q["width"], q["height"])
        assert not bad and np.allclose(px, q["box"], atol=1e-4)
        r = st.reg51.region(px, q["width"], q["height"])
        assert r["index"] == oracle[i]["index"] and r["rel"] == oracle[i]["rel"]
        assert np.allclose(r["geom"], oracle[i]["geom"], atol=1e-4)


def tiny():
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.model.enable_local_views(12)
    return d


def real_tokenizer_decider():
    """EXP50-T's real tokenizer files (no weights) with a small random embedding table standing in for the model's."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, st.RUN, "stage2", "tokenizer"))
    torch.manual_seed(0)
    emb = torch.nn.Embedding(len(tok), 16)
    enc = SimpleNamespace(get_input_embeddings=lambda: emb)
    return SimpleNamespace(tok=tok, model=SimpleNamespace(decision=SimpleNamespace(encoder=enc)))


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "runs/exp50-t/stage2/tokenizer")), reason="needs EXP50-T's tokenizer files")
def test_selector_input_is_the_question_text_and_the_picture(qs):
    d = real_tokenizer_decider()
    q = dict(qs[0])
    meta = dict(q, box=[0, 0, 5, 5], target="999", field="speed", object="B", qid="zz", key=q["key"], twin=None, family="x")
    w1, m1 = st.question_words(d, [q])
    w2, m2 = st.question_words(d, [meta])                    # same text, other metadata: same input
    assert torch.equal(w1, w2) and torch.equal(m1, m2)
    other = next(x for x in qs if st.q_text(x) != st.q_text(q))
    w3, m3 = st.question_words(d, [q, other])                 # another question: another input
    assert m3[0].sum() != m3[1].sum() or not torch.equal(w3[0, : m3[0].sum()], w3[1, : m3[1].sum()])
    torch.manual_seed(0)
    s = sel.Selector(32, w3.shape[-1]).eval()
    inp = sel.Inputs(torch.randn(1, 256, 32), torch.rand(1, 256, 4), [0, 0], w3, m3)
    b = sel.predict(s, inp, [0, 1])
    assert b[0] != b[1]                                       # A / B on the same picture: different boxes


def test_global_features_are_visual_inputs_first_256():
    from PIL import Image, ImageDraw

    from devision.image import local_pixels, to_pixel_values
    d = tiny()
    img = Image.new("RGB", (560, 405), "white")
    ImageDraw.Draw(img).text((20, 380), "Mass 14 u", fill="black")
    with torch.no_grad():
        g = d.model.vision_patches(to_pixel_values(img, 256)[None])
        kv = d.model.visual_inputs(local_pixels(img, 256, 12, 0.25, "T")[None])[0]
    assert torch.allclose(g, kv[:, :256], atol=1e-6)


def test_training_changes_only_the_selector_and_the_head_is_frozen(tmp_path):
    torch.manual_seed(1)
    inp = sel.Inputs(torch.randn(4, 256, 32), torch.rand(4, 256, 4), [0, 1, 2, 3, 0, 1], torch.randn(6, 5, 16), torch.ones(6, 5, dtype=torch.bool))
    tg = torch.rand(6, 4) * 0.5 + 0.25
    before = inp.patches.clone()
    model, stats = sel.train(inp, [0, 1, 2, 3], tg, steps=20)
    assert torch.equal(before, inp.patches) and stats["exposures"] == 20 * sel.BATCH
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    head = st.frozen_head(os.path.join(ROOT, st.HEAD)) if os.path.exists(os.path.join(ROOT, st.HEAD)) else None
    if head is not None:
        assert not any(p.requires_grad for p in head.parameters())
    bad = tmp_path / "head.pt"
    torch.save({"inp.weight": torch.zeros(1)}, bad)
    with pytest.raises(SystemExit):
        st.frozen_head(str(bad))                              # not the registered head


def test_box_rule():
    px, bad = sel.pixel_box([0.5, 0.5, 0.2, 0.1], 500, 400)
    assert not bad and np.allclose(px, [200, 180, 100, 40])
    px, bad = sel.pixel_box([0.99, 0.5, 0.5, 0.0001], 500, 400)  # clipped right, height below 1 px
    assert bad and px[0] + px[2] <= 500 and px[3] == pytest.approx(1.0) and px[2] >= 1
    px, bad = sel.pixel_box([0.0, 0.0, 0.0, 0.0], 10, 10)
    assert bad and px == [0.0, 0.0, 1.0, 1.0]
    a = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    assert float(sel.box_loss(a, a)) == pytest.approx(0.0, abs=1e-6)


def test_incomplete_run_reports_missing_and_no_automatic_result(tmp_path):
    out = str(tmp_path / "exp53")
    os.makedirs(out)
    json.dump({"started": "x"}, open(os.path.join(out, "START.json"), "w"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        res = st.report(out, finalize=" local=124")
    finally:
        os.chdir(cwd)
    assert res["states"]["auto"]["state"] == res["states"]["oracle"]["state"] == "missing"
    assert res["branch"].startswith("incomplete") and "INCOMPLETE" in open(os.path.join(out, "report.txt")).read()


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *"stages.py start"*) exit 0;;
  *"stages.py global"*) exit 0;;
  *"stages.py train"*) exec sleep 300;;
  *"stages.py report --finalize"*) mkdir -p runs/exp53; echo final > runs/exp53/report.txt; exit 0;;
esac
exit 9
"""


def test_run_finalizes_within_the_limit(tmp_path):
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp53/run.sh", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "EXP53_LIMIT": "24", "EXP53_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp53/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 124 and time.time() - t < 24 + 3, p.stdout + p.stderr
    calls = (tmp_path / "calls").read_text()
    assert "report --finalize" in calls and "stages.py local" not in calls
    assert "train interrupted" in (root / "runs/exp53-status.txt").read_text()
    left = subprocess.run(["/usr/bin/pgrep", "-f", "sleep 300"], capture_output=True, text=True).stdout.split()
    assert not [x for x in left if subprocess.run(["ps", "-o", "ppid=", "-p", x], capture_output=True, text=True).stdout.strip() == "1"]


def fake_run(out, qs):
    """A run directory as the stages leave it, with perfect automatic and oracle readouts (registered)."""
    os.makedirs(out, exist_ok=True)
    json.dump({"started": "x"}, open(os.path.join(out, "START.json"), "w"))
    for f in ("oracle.npz", "auto.npz", "selector.pt", "train.json"):
        open(os.path.join(out, f), "wb").write(f.encode())
    with open(os.path.join(out, "boxes.jsonl"), "w") as fh:
        for q in qs:
            fh.write(json.dumps({"key": q["key"], "frac": sel.target_of(q["box"], q["width"], q["height"]), "box": q["box"], "degenerate": False}) + "\n")
    st.register(out, "selector", {f: st.sha(os.path.join(out, f)) for f in ("selector.pt", "boxes.jsonl", "train.json")})
    perfect = [st.p49.labels(q["target"]) for q in qs]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    twin = [key_i[q["twin"]] if q["twin"] else i for i, q in enumerate(qs)]
    reg = {}
    for name in ("oracle", "auto"):
        ident = {"path": name, "start_sha256": st.sha(os.path.join(out, "START.json")), "head_sha256": st.HEAD_SHA,
                 "features_sha256": st.sha(os.path.join(out, name + ".npz")),
                 "boxes_sha256": st.sha(os.path.join(out, "boxes.jsonl")) if name == "auto" else None}
        reg["details_%s.jsonl" % name] = st.p49.write_details(os.path.join(out, "details_%s.jsonl" % name), qs, perfect,
                                                               [perfect[j] for j in twin], twin, ident)
    st.register(out, "readout", reg)
    return out


def run_report(out):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return st.report(out)
    finally:
        os.chdir(cwd)


def test_registered_paths_and_refusals(tmp_path, qs):
    out = fake_run(str(tmp_path / "ok"), qs)
    res = run_report(out)
    assert res["states"] == {"oracle": {"state": "completed"}, "auto": {"state": "completed"}}
    assert res["paths"]["auto"]["conditions"]["invest_original"] and res["branch"].startswith("automatic path readable")
    assert res["boxes"]["rl_tune"]["iou_mean"] == pytest.approx(1.0)
    bad = fake_run(str(tmp_path / "edited"), qs)                     # one prediction edited after registration
    p = os.path.join(bad, "details_auto.jsonl")
    lines = open(p).read().splitlines()
    row = json.loads(lines[1])
    row["pred"] = st.p49.labels("9" if row["target"] != "9" else "8")
    lines[1] = json.dumps(row)
    open(p, "w").write("\n".join(lines) + "\n")
    r2 = run_report(bad)
    assert r2["states"]["auto"]["state"] == "invalid" and r2["branch"].startswith("incomplete")
    sw = fake_run(str(tmp_path / "swapped"), qs)                     # oracle and automatic details exchanged
    a, o = os.path.join(sw, "details_auto.jsonl"), os.path.join(sw, "details_oracle.jsonl")
    os.rename(a, a + ".x")
    os.rename(o, a)
    os.rename(a + ".x", o)
    st.register(sw, "readout", {"details_oracle.jsonl": st.sha(o), "details_auto.jsonl": st.sha(a)})
    r3 = run_report(sw)
    assert r3["states"]["auto"]["state"] == r3["states"]["oracle"]["state"] == "invalid"
