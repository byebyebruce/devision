"""MLM-REPRO-PREP-01, CPU, tiny model, no real weights: the instrumented pass reproduces align._evaluate and the probe's
run_random exactly and records per-batch input / picture hashes; the three scorings and the disagreement record; the
runner's identity refusal, one start and limit."""
import importlib.util
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


def load():
    spec = importlib.util.spec_from_file_location("mr_repro", os.path.join(HERE, "repro.py"))
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_instrumented_equals_align_and_probe(tmp_path, monkeypatch):
    from conftest import tiny_decider
    from PIL import Image
    from devision.model import text_encoder
    from devision.train.align import AlignConfig, _MLMHead, _evaluate
    from devision.train.rlcd import _Images
    m = load()
    d = tiny_decider(seed=3)
    os.makedirs(tmp_path / "data" / "img")
    rows = []
    for i in range(20):
        Image.new("RGB", (40, 30), (12 * i, 200 - 9 * i, 90)).save(tmp_path / "data" / "img" / ("%d.png" % i))
        rows.append({"id": "r%d" % i, "family": "f%d" % i, "image": "img/%d.png" % i, "captions": ["A: red %d blue. B: red %d blue." % (i % 3 + 1, (i + 1) % 3 + 1)]})
    monkeypatch.chdir(tmp_path)
    head = _MLMHead(text_encoder(d.model.decision))
    d.model.eval()
    images, dev = _Images("data", d), torch.device("cpu")
    a = _evaluate(d.model, head, d.tok, rows, images, AlignConfig(batch=16, max_caption_tokens=40, mask_prob=0.5, micro_batch=0), dev)
    b = m._probe().run_random(d, head, images, dev, rows)["summary"]
    c = m.instrumented(d, head, images, dev, rows)
    for mode in ("real", "shuffled"):
        s = c["summary"][mode]
        assert s["acc_align"] == s["acc_probe"] == s["acc_cpu"] == a["val/mlm_acc_" + mode] == b[mode]["all"]["accuracy"]
        assert s["disagree"] == 0
    assert len(c["batches"]) == 4 and all(len(x["pictures"]) == 64 for x in c["batches"])
    real = [x for x in c["batches"] if x["mode"] == "real"]
    wrong = [x for x in c["batches"] if x["mode"] == "shuffled"]
    assert [x["input_ids"] for x in real] == [x["input_ids"] for x in wrong]          # same text and masks, other pictures
    assert [x["pictures"] for x in real] != [x["pictures"] for x in wrong]


def test_disagreement_record():
    m = load()
    g = torch.tensor([1, 2, 3])
    lc = torch.tensor([[0.0, 2.0, 1.9, 0.0], [0.0, 0.0, 5.0, 0.0], [float("nan"), 0.0, 0.0, 1.0]])
    out = m.compare_predictions(torch.tensor([1, 2, 3]), torch.tensor([2, 2, 3]), torch.tensor([1, 2, 3]), g, lc)
    assert out["hits_align"] == 3 and out["hits_probe"] == 2 and out["hits_cpu"] == 3 and out["nonfinite"] == 1
    assert out["disagree"] == [{"gold": 1, "align": 1, "probe": 2, "cpu": 1, "margin": pytest.approx(0.1)}]


STUB = """#!/bin/bash
echo "uv $*" >> "${MARK_CALLS}"
exit 0
"""


def test_runner(tmp_path):
    root = tmp_path / "repo"
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
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK_CALLS": str(tmp_path / "calls"), "REPRO_LIMIT": "120"}
    p = subprocess.run(["bash", str(root / "scripts/mlm_repro/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 12 and "not the accepted one" in p.stdout and not (tmp_path / "calls").exists()
    sha = lambda t: __import__("hashlib").sha256(t.encode()).hexdigest()
    env.update(REPRO_N_MODEL=sha("m"), REPRO_N_REPORT=sha("r"))
    p = subprocess.run(["bash", str(root / "scripts/mlm_repro/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert any("repro.py --checkpoint runs/exp75-panel-n/n --out runs/mlm-repro" in c for c in (tmp_path / "calls").read_text().splitlines())
    p2 = subprocess.run(["bash", str(root / "scripts/mlm_repro/run.sh")], env=env, capture_output=True, text=True)
    assert p2.returncode == 1 and "already started once" in p2.stdout
    assert "REPRO_LIMIT:-600" in open(os.path.join(HERE, "run.sh")).read()
