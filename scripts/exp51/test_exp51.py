"""exp51 without the real model: the region tokens are exactly the patches (full picture + 12 views) overlapping the field
box on view content, with rel / geometry as documented; gathered features are visual_inputs' keys / values at those
indices, the full-picture tokens identical for T and C; the donor swap changes only the features and refuses a
different region; the vision-identity check refuses one changed byte; the conditions / attribution / branch are
computed from per-question predictions and are incomplete without both conditions."""
import importlib.util
import json
import os
import struct
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reg = load("exp51_regions", "regions.py")
ext = load("exp51_extract", "extract.py")
prb = load("exp51_probe", "probe.py")
vid = load("exp51_vision", "vision_identity.py")


def test_region_tokens_are_the_overlapping_content_patches():
    from devision.image import patch_geometry, view_rects
    w, h, box = 560, 405, [10.0, 300.0, 260.0, 26.0]
    r = reg.region(box, w, h)
    views = [(0, 0, w, h)] + view_rects(w, h, 12, 0.25)
    want = set()
    for k, v in enumerate(views):                       # brute force on a 0.5-px lattice
        g = patch_geometry(v, w, h, 256, 16).numpy()
        for i, (cx, cy, pw, ph) in enumerate(g):
            xs = np.arange(max(cx * w - pw * w / 2, v[0], box[0]) + 0.25, min(cx * w + pw * w / 2, v[2], box[0] + box[2]), 0.5)
            ys = np.arange(max(cy * h - ph * h / 2, v[1], box[1]) + 0.25, min(cy * h + ph * h / 2, v[3], box[1] + box[3]), 0.5)
            if len(xs) and len(ys):
                want.add(k * 256 + i)
    assert set(r["index"]) == want and 0 in r["view"] and len(set(r["view"])) > 1
    assert all(0 <= a < 16 and 0 <= b < 16 for a, b in r["rel"])
    for k in set(r["view"]):                            # rel from the first selected cell of the same view
        cells = [i % 256 for i, v in zip(r["index"], r["view"]) if v == k]
        rel = [rr for rr, v in zip(r["rel"], r["view"]) if v == k]
        assert min(a for a, _ in rel) == 0 and min(b for _, b in rel) == 0 and len(cells) == len(rel)
    gx = [g[0] for g in r["geom"]]                      # geometry relative to the box: centres around [0, 1]
    assert min(gx) < 0.1 and max(gx) > 0.9


def test_region_ignores_the_letterbox_margin():
    """A box touching only a view's grey margin area gets no token from that view."""
    w, h = 650, 140
    r = reg.region([0.0, 0.0, 2.0, 2.0], w, h)          # the top-left corner: only views holding it
    from devision.image import view_rects
    views = [(0, 0, w, h)] + view_rects(w, h, 12, 0.25)
    assert all(views[k][0] <= 2 and views[k][1] <= 2 for k in set(r["view"]))


def tiny_local():
    from conftest import tiny_decider  # type: ignore
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.model.enable_local_views(12)
    return d.model.eval()


def test_gathered_tokens_are_visual_inputs_patches_and_full_branch_is_shared():
    from devision.image import local_pixels, to_pixel_values, view_rects
    m = tiny_local()
    img = Image.new("RGB", (560, 405), "white")
    ImageDraw.Draw(img).text((20, 310), "Mass 1,360 u", fill="black")
    r = reg.region([10.0, 300.0, 260.0, 26.0], 560, 405)
    with torch.no_grad():
        kv = {s: m.visual_inputs(local_pixels(img, 256, 12, 0.25, s)[None])[0][0].numpy() for s in "TC"}
        rects = view_rects(560, 405, 12, 0.25)
        k = max(r["view"])
        own = m.vision_patches(to_pixel_values(img.crop(rects[k - 1]), 256)[None])[0].numpy()
    t = ext.gather(kv["T"], r)
    for n, (i, v) in enumerate(zip(r["index"], r["view"])):
        if v == k:
            assert np.allclose(t[n], own[i % 256], atol=1e-5)
    c = ext.gather(kv["C"], r)
    full = [n for n, v in enumerate(r["view"]) if v == 0]
    assert np.array_equal(t[full], c[full])                                       # same full-picture tokens
    assert not np.array_equal(t, c)                                               # views differ in pixels


def test_donor_swap_and_refusal():
    regs = [{"index": [1, 2], "rel": [[0, 0], [0, 1]], "geom": [[0, 0, 1, 1], [1, 0, 1, 1]]},
            {"index": [1, 2], "rel": [[0, 0], [0, 1]], "geom": [[0, 0, 1, 1], [1, 0, 1, 1]]},
            {"index": [5], "rel": [[0, 0]], "geom": [[0, 0, 1, 1]]}]
    flat = np.arange(5 * 3, dtype=np.float16).reshape(5, 3)
    tok = prb.Tokens(flat, np.array([0, 2, 4, 5]), regs)
    x, m, r, g = tok.batch([0], [1])
    assert torch.equal(x[0], torch.from_numpy(flat[2:4].astype(np.float32))) and torch.equal(r[0], torch.tensor([[0, 0], [0, 1]]))
    with pytest.raises(SystemExit):
        tok.batch([0], [2])


def fake_safetensors(path, data: bytes):
    header = json.dumps({"vision.a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                         "decision.b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}).encode()
    open(path, "wb").write(struct.pack("<Q", len(header)) + header + data)


def test_vision_identity(tmp_path):
    runs = {}
    for k in "XTC":
        runs[k] = str(tmp_path / (k + ".safetensors"))
        fake_safetensors(runs[k], b"\x00" * 8 + (b"\x01\x02\x03\x04" if k == "T" else b"\x09\x09\x09\x09"))
    assert vid.check(runs)["T"]["tensors"] == 1                                    # decision tensors may differ
    fake_safetensors(runs["C"], b"\x00" * 7 + b"\x01" + b"\x09" * 4)
    with pytest.raises(SystemExit):
        vid.check(runs)


def test_conditions_branch_and_incomplete():
    qs = [json.loads(x) for x in open(os.path.join(ROOT, "data", "exp49", "questions.jsonl"))]
    perfect = [prb.p49.labels(q["target"]) for q in qs]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor = [perfect[key_i[q["twin"]]] if q["twin"] else perfect[i] for i, q in enumerate(qs)]
    wrong = [[prb.p49.EOS] + [-100] * 4 for _ in qs]                              # empty: invalid everywhere
    res = prb.report_from(qs, {"T": (perfect, donor), "C": (wrong, wrong)})
    assert res["conditions"]["T"]["status"]["original"] == "positive" and res["T_minus_C"]["rl_tune"]["value"] == pytest.approx(1.0)
    assert res["conditions"]["C"]["status"]["original"].startswith("not learned")
    assert str(res["attribution"]).startswith("not located")                       # an unlearned C is no control
    assert res["branch"].startswith("T alone readable") and "attribution not located" in res["branch"]   # T's positive kept
    res2 = prb.report_from(qs, {"T": (perfect, donor), "C": (perfect, donor)})
    assert res2["attribution"] is False and res2["branch"].startswith("T and C both readable")
    assert prb.report_from(qs, {"T": (perfect, donor)})["branch"].startswith("incomplete")
    assert res["conditions"]["T"]["cut_slice"] == {"right": 15, "of": 15}


# ---- critic EXP51-R1-01 / 02 / 03: identity, incomplete runs, branch

QUESTIONS, REGIONS = os.path.join(ROOT, "data/exp49/questions.jsonl"), os.path.join(ROOT, "data/exp51/regions.json.gz")


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def preds():
    qs = [json.loads(x) for x in open(QUESTIONS)]
    perfect = [prb.p49.labels(q["target"]) for q in qs]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    donor = [perfect[key_i[q["twin"]]] if q["twin"] else perfect[i] for i, q in enumerate(qs)]
    wrong = [[prb.p49.EOS] + [-100] * 4 for _ in qs]
    key_i = {q["key"]: i for i, q in enumerate(qs)}
    return qs, perfect, donor, wrong, [key_i[q["twin"]] if q["twin"] else i for i, q in enumerate(qs)]


def fake_run(out, preds, conds=("T", "C"), half=None, register=True):
    """A run directory as start / extract / probe leave it: START, features.json, feature / head files, details."""
    qs, perfect, donor, wrong, donor_idx = preds
    os.makedirs(out, exist_ok=True)
    json.dump({"started": "x"}, open(os.path.join(out, "START.json"), "w"))
    info = {"start_sha256": sha(os.path.join(out, "START.json")), "model": {"run": "runs/exp50-t"},
            "questions_sha256": sha(QUESTIONS), "regions_sha256": sha(REGIONS), "conditions": {}}
    for cond in ("T", "C"):
        open(os.path.join(out, "features_%s.npz" % cond), "wb").write(cond.encode() * 10)
        info["conditions"][cond] = {"sha256": sha(os.path.join(out, "features_%s.npz" % cond))}
    json.dump(info, open(os.path.join(out, "features.json"), "w"))
    for cond in conds:
        open(os.path.join(out, "head_%s.pt" % cond), "wb").write(b"head" + cond.encode())
        ident = {"features_sha256": info["conditions"][cond]["sha256"], "questions_sha256": info["questions_sha256"],
                 "regions_sha256": info["regions_sha256"], "model": info["model"], "condition": cond,
                 "start_sha256": info["start_sha256"], "head_sha256": sha(os.path.join(out, "head_%s.pt" % cond))}
        pred, dn = (perfect, donor) if cond == "T" else (wrong, wrong)
        path = os.path.join(out, "details_%s.jsonl" % cond)
        dsha = prb.p49.write_details(path, qs, pred, dn, donor_idx, ident)
        if register:
            prb.register(out, cond, info["start_sha256"], ident["head_sha256"], dsha)
        if half == cond:                                                        # cut mid-write
            lines = open(path).read().splitlines()
            open(path, "w").write("\n".join(lines[: len(lines) // 2]) + "\n")
    return out


def test_recompute_keeps_identity_and_matches(tmp_path, preds):
    out = fake_run(str(tmp_path / "run"), preds)
    res = prb.recompute(QUESTIONS, out, REGIONS)
    assert {c: s["state"] for c, s in res["states"].items()} == {"T": "completed", "C": "completed"}
    direct = prb.report_from(preds[0], {"T": (preds[1], preds[2]), "C": (preds[3], preds[3])})
    assert prb.same_result(res, direct) and res["T_minus_C"]["rl_tune"]["value"] == pytest.approx(1.0)


def test_swapped_or_foreign_details_are_invalid(tmp_path, preds):
    out = fake_run(str(tmp_path / "run"), preds)
    t, c = os.path.join(out, "details_T.jsonl"), os.path.join(out, "details_C.jsonl")
    os.rename(t, t + ".x")
    os.rename(c, t)
    os.rename(t + ".x", c)                                                      # T and C details exchanged
    res = prb.recompute(QUESTIONS, out, REGIONS)
    assert res["states"]["T"]["state"] == res["states"]["C"]["state"] == "invalid"
    assert res["T_minus_C"] is None and str(res["attribution"]).startswith("incomplete")
    out2 = fake_run(str(tmp_path / "run2"), preds)
    open(os.path.join(out2, "features_C.npz"), "ab").write(b"changed")       # registered content changed
    json.dump({"started": "other"}, open(os.path.join(out2, "START.json"), "w"))  # another START
    res2 = prb.recompute(QUESTIONS, out2, REGIONS)
    assert res2["states"]["C"]["state"] == "invalid" and "features file changed" in res2["states"]["C"]["reason"]
    assert "other START" in res2["states"]["T"]["reason"]


def test_finalize_no_features_only_t_and_half_written(tmp_path, preds):
    out = str(tmp_path / "none")
    os.makedirs(out)
    open(out + "-status.txt", "w").write("start completed\nextract interrupted\n")
    res = prb.finalize(QUESTIONS, REGIONS, out, " start=0 extract=124")
    assert {c: s["state"] for c, s in res["states"].items()} == {"T": "missing", "C": "missing"}
    assert "INCOMPLETE" in open(os.path.join(out, "report.txt")).read() and res["finalized"]["stages"][1] == "extract interrupted"
    only_t = fake_run(str(tmp_path / "t"), preds, conds=("T",))
    r = prb.finalize(QUESTIONS, REGIONS, only_t, " probe=124")
    assert r["states"]["T"]["state"] == "completed" and r["states"]["C"]["state"] == "partial"
    assert r["conditions"]["T"]["status"]["original"] == "positive" and r["T_minus_C"] is None
    half = fake_run(str(tmp_path / "half"), preds, half="C")
    r = prb.finalize(QUESTIONS, REGIONS, half, " probe=124")
    assert r["states"]["C"]["state"] == "partial" and "incomplete" in r["states"]["C"]["reason"]


def test_c_alone_positive_is_kept():
    pos = {"original": "positive"}
    assert prb.branch({"T": {"original": "valid, below the condition"}, "C": pos}, False).startswith("C alone readable")
    assert prb.branch({"T": {"original": "not learned: not located"}, "C": pos}, False).startswith("C alone readable")
    assert prb.branch({"T": {"original": "valid, below the condition"}, "C": {"original": "not learned: not located"}}, False).startswith("no condition positive")
    assert prb.branch({"T": {"original": "valid, below the condition"}, "C": {"original": "valid, below the condition"}}, False).startswith("valid heads")


STUB_UV = """#!/bin/bash
echo "uv $*" >> "${MARK}"
case "$*" in
  *start.py*) exit 0;;
  *extract.py*) [ "${HANG}" = extract ] && exec sleep 300; exit 7;;
  *--finalize*) mkdir -p runs/exp51; echo final > runs/exp51/report.txt; exit 0;;
  *probe.py*) exec sleep 300;;
esac
"""


@pytest.mark.parametrize("hang", ["extract", "fail"])
def test_run_finalizes_within_the_limit(tmp_path, hang):
    """Extraction stuck (stopped at the limit) or failing: finalize still runs, inside the limit, no subtree left."""
    import shutil
    import stat
    import subprocess
    import time
    root = tmp_path / "repo"
    for rel in ("scripts/exp51/run.sh", "scripts/exp35/guard.sh", "scripts/exp38/steps.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), root / rel)
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in {"uv": STUB_UV, "pgrep": '#!/bin/bash\n[ "$1" = "-fl" ] && exit 1\nexec /usr/bin/pgrep "$@"\n'}.items():
        (b / name).write_text(body)
        (b / name).chmod((b / name).stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": "%s:%s" % (b, os.environ["PATH"]), "MARK": str(tmp_path / "calls"), "HANG": hang,
           "EXP51_LIMIT": "24", "EXP51_RESERVE": "4"}
    t = time.time()
    p = subprocess.run(["bash", str(root / "scripts/exp51/run.sh")], env=env, capture_output=True, text=True, timeout=60)
    took = time.time() - t
    assert p.returncode == (124 if hang == "extract" else 7), p.stdout + p.stderr
    assert took < 24 + 3
    calls = (tmp_path / "calls").read_text()
    assert "--finalize" in calls and "probe.py --device" not in calls
    assert (root / "runs/exp51/report.txt").read_text() == "final\n"
    st = (root / "runs/exp51-status.txt").read_text()
    assert ("extract interrupted" if hang == "extract" else "extract failed") in st
    left = subprocess.run(["/usr/bin/pgrep", "-f", "sleep 300"], capture_output=True, text=True).stdout.split()
    assert not [x for x in left if subprocess.run(["ps", "-o", "ppid=", "-p", x], capture_output=True, text=True).stdout.strip() == "1"]


def test_a_changed_prediction_is_refused_against_the_registry(tmp_path, preds):
    """critic R2-01: one valid prediction edited (header, target, donor, features and head untouched) -> invalid."""
    out = fake_run(str(tmp_path / "run"), preds)
    path = os.path.join(out, "details_T.jsonl")
    lines = open(path).read().splitlines()
    row = json.loads(lines[1])
    row["pred"] = prb.p49.labels("9" if row["target"] != "9" else "8")
    lines[1] = json.dumps(row)
    open(path, "w").write("\n".join(lines) + "\n")
    res = prb.recompute(QUESTIONS, out, REGIONS)
    assert res["states"]["T"]["state"] == "invalid" and "registered" in res["states"]["T"]["reason"]
    assert res["states"]["C"]["state"] == "completed" and res["T_minus_C"] is None
    unreg = fake_run(str(tmp_path / "unreg"), preds, register=False)        # details without a registry entry
    r = prb.recompute(QUESTIONS, unreg, REGIONS)
    assert r["states"]["T"]["state"] == r["states"]["C"]["state"] == "invalid" and "not registered" in r["states"]["T"]["reason"]


def test_only_t_registered_is_verifiable(tmp_path, preds):
    out = fake_run(str(tmp_path / "t"), preds, conds=("T",))
    reg = json.load(open(os.path.join(out, "registry.json")))
    assert set(reg) == {"T"} and reg["T"]["details_sha256"] == sha(os.path.join(out, "details_T.jsonl"))
    r = prb.finalize(QUESTIONS, REGIONS, out, " probe=124")
    assert r["states"]["T"]["state"] == "completed" and r["states"]["C"]["state"] == "partial"
    assert r["conditions"]["T"]["status"]["original"] == "positive" and str(r["attribution"]).startswith("incomplete")


def test_truncated_features_json_still_finalizes(tmp_path, preds):
    """critic R2-02: features.json cut mid-write -> finalize writes the incomplete report with the reason."""
    out = fake_run(str(tmp_path / "trunc"), preds)
    fj = os.path.join(out, "features.json")
    text = open(fj).read()
    open(fj, "w").write(text[: len(text) // 2])
    r = prb.finalize(QUESTIONS, REGIONS, out, " extract=124")
    assert r["states"]["T"]["state"] == r["states"]["C"]["state"] == "invalid"
    assert "features.json unreadable" in r["states"]["T"]["reason"] and r["T_minus_C"] is None
    assert str(r["attribution"]).startswith("incomplete") and "INCOMPLETE" in open(os.path.join(out, "report.txt")).read()
    assert json.load(open(os.path.join(out, "report.json")))["finalized"]["exit_codes"] == " extract=124"
    nodet = str(tmp_path / "nodetails")                                       # truncated metadata, no details yet
    os.makedirs(nodet)
    open(os.path.join(nodet, "features.json"), "w").write('{"conditions": {"T"')
    r2 = prb.finalize(QUESTIONS, REGIONS, nodet, " extract=124")
    assert r2["states"]["T"]["state"] == "invalid" and "unreadable" in r2["states"]["T"]["reason"]


def test_register_and_write_json_are_atomic(tmp_path):
    p = str(tmp_path / "x.json")
    prb.write_json(p, {"a": 1})
    assert json.load(open(p)) == {"a": 1} and not os.path.exists(p + ".tmp")
    open(str(tmp_path / "registry.json"), "w").write("{broken")
    with pytest.raises(SystemExit):
        prb.register(str(tmp_path), "T", "s", "h", "d")                       # a damaged registry is never overwritten
