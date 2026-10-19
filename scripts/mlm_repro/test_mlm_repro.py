"""MLM-REPRO-PREP-01 (round 1), CPU, tiny model, no real weights:
  - the runtime wrappers record what A (align._evaluate) and B (probe.run_random) actually ran; with C they align batch
    by batch (input_equal) on an unchanged path; a changed mask / picture at B's entry is caught per batch;
  - identity refusals before any load (START, config, validation file, picture bytes, wrong map, MLM head);
  - three scorings and the disagreement record with row / token positions;
  - finalize: complete only when every segment is; a truncated / missing segment is partial with no conclusion;
  - the runner: identity refusal, a failing or over-time run still finalizes inside the limit, one start."""
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load():
    spec = importlib.util.spec_from_file_location("mr_repro", os.path.join(HERE, "repro.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    from conftest import tiny_decider
    from PIL import Image
    from devision.model import text_encoder
    from devision.train.align import _MLMHead
    from devision.train.rlcd import _Images
    m = load()
    d = tiny_decider(seed=3)
    os.makedirs(tmp_path / "data" / "img")
    rows = []
    for i in range(32):
        Image.new("RGB", (40, 30), (7 * i, 200 - 5 * i, 90)).save(tmp_path / "data" / "img" / ("%d.png" % i))
        rows.append({"id": "r%d" % i, "family": "f%d" % i, "image": "img/%d.png" % i, "captions": ["A: red %d blue. B: red %d blue." % (i % 3 + 1, (i + 1) % 3 + 1)]})
    monkeypatch.chdir(tmp_path)
    head = _MLMHead(text_encoder(d.model.decision))
    d.model.eval()
    monkeypatch.setattr(m, "BATCHES", 4)
    return m, d, head, _Images("data", d), torch.device("cpu"), rows


def segments(m, d, head, images, dev, rows, out, b_images=None, b_mask=None):
    """A, B, C written as run() writes them (tiny: 2 modes x 2 batches)."""
    from devision.train.align import AlignConfig, _evaluate
    probe = m._probe()
    os.makedirs(out)
    for seg in ("a", "b"):
        recs = []
        with m.recording(recs):
            if seg == "a":
                summ = _evaluate(d.model, head, d.tok, rows, images, AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0), dev)
            elif b_mask is not None:                       # B's entry with another mask probability
                summ = _evaluate(d.model, head, d.tok, rows, images, AlignConfig(batch=16, max_caption_tokens=40, mask_prob=b_mask, micro_batch=0), dev)
            else:
                summ = probe.run_random(d, head, b_images or images, dev, rows)["summary"]
        os.makedirs(os.path.join(out, seg))
        open(os.path.join(out, seg, "batches.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
        m.write_atomic(os.path.join(out, seg, "summary.json"), {"summary": summ})
    os.makedirs(os.path.join(out, "c"))
    with open(os.path.join(out, "c", "batches.jsonl"), "w") as fh:
        tot = m.instrumented(d, head, images, dev, rows, lambda r: fh.write(json.dumps(r) + "\n"))
    m.write_atomic(os.path.join(out, "c", "summary.json"), {"summary": tot})
    return tot


def test_unchanged_paths_align_batch_by_batch(tiny):
    m, d, head, images, dev, rows = tiny
    tot = segments(m, d, head, images, dev, rows, "out")
    res = m.finalize("out")
    assert res["status"] == "complete" and res["input_equal"] and res["differences"] == [] and res["disagreeing_tokens"] == 0
    for mode in ("real", "shuffled"):
        assert res["segments"]["a"]["recounted"][mode] == res["segments"]["b"]["recounted"][mode] == res["segments"]["c"]["recounted"][mode] == tot[mode]["acc_align"]


@pytest.mark.parametrize("change", ["picture", "mask"])
def test_a_changed_entry_is_caught(tiny, change):
    m, d, head, images, dev, rows = tiny
    if change == "picture":
        from devision.train.rlcd import _Images

        class Other(_Images):
            def __call__(self, rel):
                return super().__call__("img/0.png")
        segments(m, d, head, images, dev, rows, "out", b_images=Other("data", d))
        want = "pictures"
    else:
        segments(m, d, head, images, dev, rows, "out", b_mask=0.3)
        want = "target_hash"
    res = m.finalize("out")
    assert res["status"] == "complete" and not res["input_equal"] and any(x.get("hash") == want for x in res["differences"])


@pytest.mark.parametrize("damage", ["c_truncated", "c_no_summary", "b_missing"])
def test_partial_runs_give_no_conclusion(tiny, damage):
    m, d, head, images, dev, rows = tiny
    segments(m, d, head, images, dev, rows, "out")
    if damage == "c_truncated":
        lines = open("out/c/batches.jsonl").read().splitlines()
        open("out/c/batches.jsonl", "w").write("\n".join(lines[:2]) + "\n" + lines[2][:20])
    elif damage == "c_no_summary":
        os.remove("out/c/summary.json")
    else:
        shutil.rmtree("out/b")
    res = m.finalize("out")
    assert res["status"].startswith("partial") and "input_equal" not in res and res["note"].startswith("no conclusion")
    assert res["segments"]["a"]["state"] == "complete"                                  # the finished segment is kept
    assert json.load(open("out/report.json"))["status"] == res["status"]


def test_disagreement_record():
    m = load()
    g = torch.tensor([1, 2, 3])
    lc = torch.tensor([[0.0, 2.0, 1.9, 0.0], [0.0, 0.0, 5.0, 0.0], [float("nan"), 0.0, 0.0, 1.0]])
    out = m.compare_predictions(torch.tensor([1, 2, 3]), torch.tensor([2, 2, 3]), torch.tensor([1, 2, 3]), g, lc, [[0, 4], [1, 2], [3, 7]])
    assert out["hits_align"] == 3 and out["hits_probe"] == 2 and out["hits_cpu"] == 3 and out["nonfinite"] == 1
    assert out["disagree"] == [{"gold": 1, "align": 1, "probe": 2, "cpu": 1, "margin": pytest.approx(0.1), "row": 0, "token": 4}]


@pytest.fixture
def ident(tmp_path, monkeypatch):
    from PIL import Image
    m = load()
    monkeypatch.chdir(tmp_path)
    os.makedirs("ck/tokenizer")
    os.makedirs("data/img")
    open("ck/model.safetensors", "w").write("m")
    open("ck/config.json", "w").write("{}")
    open("ck/tokenizer/t.json", "w").write("t")
    open("head.safetensors", "w").write("h")
    rows = []
    for i in range(3):
        Image.new("RGB", (8, 8), (40 * i, 0, 0)).save("data/img/%d.png" % i)
        rows.append({"id": "v%d" % i, "image": "img/%d.png" % i})
    open("data/val.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    rep = m._bind_report()
    wrong = {"v0": "v1", "v1": "v2", "v2": "v0"}
    pics = hashlib.sha256(json.dumps([m.sha("data/img/%d.png" % i) for i in range(3)]).encode()).hexdigest()
    st = {"endpoints": {"n": {"path": "ck", "model.safetensors": m.sha("ck/model.safetensors"), "config.json": m.sha("ck/config.json"),
                              "tokenizer": rep.tree("ck/tokenizer")}},
          "mlm_head": {"repo": "x", "model.safetensors": m.sha("head.safetensors")},
          "val": {"sha256": m.sha("data/val.jsonl"), "pictures_sha256": pics}, "wrong_picture": wrong}
    json.dump(st, open("START.json", "w"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: "head.safetensors")
    return m, ("START.json", m.sha("START.json"))


@pytest.mark.parametrize("change", [None, "start", "config", "val", "picture", "head"])
def test_identity(ident, change):
    m, start = ident
    if change == "start":
        start = (start[0], "0" * 64)
    elif change == "config":
        open("ck/config.json", "w").write('{"x": 1}')
    elif change == "val":
        open("data/val.jsonl", "a").write("\n")
    elif change == "picture":
        from PIL import Image
        Image.new("RGB", (8, 8), "blue").save("data/img/1.png")
    elif change == "head":
        open("head.safetensors", "w").write("other")
    bad = m.identity_problems("ck", start, "data/val.jsonl")
    assert (bad == []) == (change is None), bad


def test_refused_identity_loads_nothing(ident, monkeypatch):
    m, _ = ident
    monkeypatch.setattr(m, "START", ("START.json", "0" * 64))
    monkeypatch.setattr(m, "_probe", lambda: pytest.fail("loaded despite a refused identity"))
    assert m.run("ck", "out", "cpu") == 12 and json.load(open("out/identity.json"))["ok"] is False


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
case "$*" in
  *"repro.py run"*) [ -n "${RUN_SLEEP:-}" ] && sleep "$RUN_SLEEP"; [ -n "${RUN_FAIL:-}" ] && exit 7; exit 0;;
esac
exit 0
"""


def chain(tmp_path, extra):
    root = tmp_path / "repo"
    if not root.exists():
        for rel in ("scripts/mlm_repro/run.sh", "scripts/mlm_repro/repro.py", "scripts/panel_bind/probe.py", "src/devision/train/align.py",
                    "scripts/exp35/guard.sh"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(os.path.join(ROOT, rel), root / rel)
        (root / "runs/exp75-panel-n/n").mkdir(parents=True)
        (root / "runs/exp75-panel-n/n/model.safetensors").write_text("m")
        (root / "runs/exp75-panel-n/n/align_report.json").write_text("r")
        b = tmp_path / "bin"
        b.mkdir()
        for name, body in {"uv": STUB, "pgrep": "#!/bin/bash\nexit 1\n", "ps": "#!/bin/bash\nexit 0\n"}.items():
            (b / name).write_text(body)
            (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
    env = {**os.environ, "PATH": "%s:%s" % (tmp_path / "bin", os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"),
           "REPRO_N_MODEL": sha("m"), "REPRO_N_REPORT": sha("r"), **extra}
    t0 = time.time()
    p = subprocess.run(["bash", str(root / "scripts/mlm_repro/run.sh")], env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls, time.time() - t0, root


def test_runner_identity_refusal(tmp_path):
    p, calls, _, _ = chain(tmp_path, {"REPRO_N_MODEL": "0" * 64})
    assert p.returncode == 12 and not calls


@pytest.mark.parametrize("extra,rc", [({}, 0), ({"RUN_FAIL": "1"}, 7), ({"RUN_SLEEP": "60", "REPRO_LIMIT": "75"}, 124)])
def test_runner_always_finalizes_inside_the_limit(tmp_path, extra, rc):
    p, calls, took, root = chain(tmp_path, extra)
    assert p.returncode == rc, p.stdout + p.stderr
    assert [c.split()[4] for c in calls] == ["run", "finalize"]
    if "REPRO_LIMIT" in extra:
        assert took < 75 and "limit" in p.stdout                                         # stopped, then finalized in time
    p2 = subprocess.run(["bash", str(root / "scripts/mlm_repro/run.sh")], env={**os.environ, "PATH": "%s:%s" % (tmp_path / "bin", os.environ["PATH"]),
                        "MARK_CALLS": str(tmp_path / "x")}, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    assert "REPRO_LIMIT:-600" in open(os.path.join(HERE, "run.sh")).read()
