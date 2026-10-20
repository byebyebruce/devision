"""The reading combination behind the public entry (critic V04-EVIDENCE-R3-01; docs/experiments/2026-10-19-v04-reader-
evidence-prep03.md 5.1): a checkpoint with a reading/ bundle loads through devision.load as the verified reader; the
picture is resolved once for both the image= argument and an image part in state, so they give the same reads, table,
state and probabilities; the evaluation entry records the route each question took and the picture actually read
(the swapped one under the mismatched control); the public interface (predict, cfg, raw temperatures, /health, 422s)
holds; a bundle whose declared rules differ from the code is refused; an "off" load says so."""
import base64
import io
import json

import pytest
import torch
from conftest import tiny_decider
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import devision
from devision import reading as R
from devision.infer import Decider, picture_identity
from devision.serve import create_app
from devision.train.evaluate import evaluate

TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
MASS_A = "What is the mass of each particle in sample A?"
DOG = "is there a dog ?"


def q(text, opts=("red", "blue")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


def local_view_decider(seed=3, temperature=None):
    d = tiny_decider(seed=seed)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    if temperature is not None:
        d.cfg.temperature = temperature
    return d


def checkpoint(tmp_path, name="ck", temperature=None, bundle=True):
    """A tiny decision checkpoint with its reading bundle in <checkpoint>/reading/ (as a release would carry it)."""
    local_view_decider(temperature=temperature).save(tmp_path / name)
    torch.manual_seed(7)
    if not (tmp_path / "sel.pt").exists():
        torch.save(R.Selector(32, 64).state_dict(), tmp_path / "sel.pt")
        torch.save(R.CharHead(32).state_dict(), tmp_path / "head.pt")
    if bundle:
        R.write_bundle(str(tmp_path / name / "reading"), str(tmp_path / name), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"),
                       {}, dims={"selector": [32, 64], "head": 32})
    return str(tmp_path / name)


def drawn(colour="white", text="Mass 14 u   speed 1,360 m/s"):
    img = Image.new("RGB", (560, 405), colour)
    ImageDraw.Draw(img).text((20, 380), text, fill="black")
    return img


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_load_gives_the_reader_and_off_says_so(tmp_path):
    ck = checkpoint(tmp_path)
    rd = devision.load(ck, device="cpu")
    assert isinstance(rd, R.Reading) and rd.reading_status["mode"] == "on"
    assert rd.reading_status["reading_json_sha256"] == R.sha(ck + "/reading/reading.json")
    off = devision.load(ck, device="cpu", reading="off")
    assert isinstance(off, Decider) and off.reading_status["mode"] == "off"
    plain = checkpoint(tmp_path, "plain", bundle=False)
    d = devision.load(plain, device="cpu")
    assert isinstance(d, Decider) and d.reading_status == {"mode": "none"}
    img = drawn()
    qs = {"t": q(TEMP), "o": q(DOG)}
    assert d.decide("x", qs, image=img) == Decider.load(plain, device="cpu").decide("x", qs, image=img)   # unchanged
    with pytest.raises(ValueError, match="no reading/ bundle"):
        devision.load(plain, device="cpu", reading="on")


def test_hub_ids_are_resolved_before_looking_for_reading(tmp_path, monkeypatch):
    ck = checkpoint(tmp_path)
    seen = {}

    def fake_snapshot(repo, revision=None, token=None):
        seen.update(repo=repo, revision=revision, token=token)
        return ck
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    rd = devision.load("someone/model", device="cpu", revision="v0.4", token="t")
    assert isinstance(rd, R.Reading) and seen == {"repo": "someone/model", "revision": "v0.4", "token": "t"}


def test_image_argument_and_state_image_part_are_one_request(tmp_path):
    rd = devision.load(checkpoint(tmp_path), device="cpu")
    img = drawn()
    for qs in ({"t": q(TEMP)}, {"m": q(MASS_A), "o": q(DOG)}, {"o": q(DOG)}):
        a, b = {}, {}
        top = rd.decide("Two samples.", qs, image=img, diagnostics=a)
        part = rd.decide([{"type": "image", "base64": b64(img)}, "Two samples."], qs, diagnostics=b)
        for x in (a, b):
            x.pop("read_seconds", None)                                              # wall time
        assert top == part and a == b
        assert a["picture"] == picture_identity(img)
        assert a["route"] == {k: ("read" if R.domain_of(v["instructions"]) else "ordinary") for k, v in qs.items()}
    assert rd.predict("Two samples.", {"t": q(TEMP)}, image=img) == rd.decide("Two samples.", {"t": q(TEMP)}, image=img)


def samples(root):
    """Two drawn pictures; each asks a temperature comparison, a reading question and an ordinary one."""
    (root / "img").mkdir()
    rows = []
    for i, colour in enumerate(("white", "lightyellow")):
        drawn(colour, "Mass %d u   speed %d m/s" % (10 + i, 900 + i)).save(root / "img" / ("%d.png" % i))
        rows.append({"id": "s%d" % i, "source": "syn", "kind": "k", "image_id": "p%d" % i, "image": "img/%d.png" % i,
                     "questions": {"t": q(TEMP), "m": q(MASS_A, ("blue", "red")), "o": q(DOG)},   # short options: the tiny input is short
                     "gold": {"t": {"probabilities": {"red": 1.0, "blue": 0.0}},
                              "m": {"probabilities": {"blue": 1.0, "red": 0.0}},
                              "o": {"probabilities": {"red": 0.0, "blue": 1.0}}}})
    return rows


@pytest.mark.parametrize("control", ["none", "mismatched", "reversed"])
def test_the_evaluation_entry_reads_the_picture_it_sends(tmp_path, control):
    rd = devision.load(checkpoint(tmp_path), device="cpu")
    rows = samples(tmp_path)
    recs: list = []
    out = evaluate(rd, rows, tmp_path, control=control, records_out=recs)
    for r in recs:
        assert r["route"] == ("read" if r["qid"] in ("t", "m") else "ordinary")
        sent = picture_identity(Image.open(tmp_path / r["image_used"]).convert("RGB"))
        assert sent is not None and r["picture_pixels_sha256"] == sent["pixels_sha256"]
    if control == "mismatched":
        assert all(r["image_used"] != row["image"] for r in recs for row in rows if row["id"] == r["sample_id"])
    routed = sum(r["qid"] in ("t", "m") for r in recs)
    assert out["reading"]["mode"] == "on" and out["reading"]["routed_questions"] == out["reading"]["read_questions"] == routed
    if control == "reversed":
        plain: list = []
        evaluate(rd, rows, tmp_path, records_out=plain)
        tables = {(r["sample_id"], r["qid"]): r.get("read_table") for r in plain}
        assert all(r["read_table"] == tables[(r["sample_id"], r["qid"])] for r in recs if r["route"] == "read")
    off: list = []
    res = evaluate(devision.load(tmp_path / "ck", device="cpu", reading="off"), rows, tmp_path, control=control, records_out=off)
    assert res["reading"]["mode"] == "off" and res["reading"]["read_questions"] == 0 < res["reading"]["routed_questions"]


def test_raw_temperatures_cover_both_paths(tmp_path):
    hot = devision.load(checkpoint(tmp_path, "hot", temperature=[2.5, 1.0, 3.0]), device="cpu")
    one = devision.load(checkpoint(tmp_path, "one", temperature=[1.0, 1.0, 1.0]), device="cpu")
    img = drawn()
    qs = {"t": q(TEMP), "o": q(DOG)}
    assert hot.decide("", qs, image=img) != one.decide("", qs, image=img)
    hot.use_raw_temperatures()
    assert hot.cfg is hot.decider.cfg and hot.model is hot.decider.model
    assert hot.decide("", qs, image=img) == one.decide("", qs, image=img)


def test_http_with_the_reader(tmp_path):
    rd = devision.load(checkpoint(tmp_path), device="cpu")
    c = TestClient(create_app(rd))
    img = drawn()
    qs = {"t": q(TEMP)}
    top = c.post("/v1/systemone", json={"image": b64(img), "state": "x", "questions": qs})
    part = c.post("/v1/systemone", json={"state": [{"type": "image", "base64": b64(img)}, "x"], "questions": qs})
    assert top.status_code == part.status_code == 200 and top.json() == part.json()
    both = c.post("/v1/systemone", json={"image": b64(img), "state": [{"type": "image", "base64": b64(img)}], "questions": qs})
    assert both.status_code == 422 and "not both" in both.json()["error"]
    path = tmp_path / "pic.png"
    img.save(path)
    assert c.post("/v1/systemone", json={"image": str(path), "state": "", "questions": qs}).status_code == 422
    assert c.post("/v1/systemone", json={"image": "bm90IGFuIGltYWdl", "state": "", "questions": qs}).status_code == 422
    h = c.get("/health").json()
    assert h["reading"]["mode"] == "on" and h["reading"]["reading_json_sha256"] == rd.reading_status["reading_json_sha256"]


@pytest.mark.parametrize("change", ["template", "header", "missing_header"])
def test_declared_rules_must_match_the_code(tmp_path, change):
    ck = checkpoint(tmp_path)
    p = tmp_path / "ck" / "reading" / "reading.json"
    ident = json.loads(p.read_text())
    if change == "template":
        ident["templates"][MASS_A] = ["Sample A", "weight"]
    elif change == "header":
        ident["header"] = "Values:"
    else:                                   # an older bundle (exp66-exp71): no header / table format / rules hash
        for k in ("header", "table_format", "rules_sha256"):
            ident.pop(k)
    p.write_text(json.dumps(ident))
    with pytest.raises(ValueError, match="rules differ"):
        devision.load(ck, device="cpu")
    if change == "missing_header":         # still loads for its own experiment through Reading.load
        assert R.Reading.load(ck, ck + "/reading", device="cpu").reading_status["mode"] == "on"
    else:
        with pytest.raises(ValueError, match="rules differ"):
            R.Reading.load(ck, ck + "/reading", device="cpu")


def test_calibrate_refuses_a_bound_checkpoint(tmp_path):
    from devision.train.cli import calibrate_main
    ck = checkpoint(tmp_path)
    (tmp_path / "d.jsonl").write_text("")
    with pytest.raises(SystemExit, match="reading/ bundle"):
        calibrate_main(["--checkpoint", ck, "--data", str(tmp_path / "d.jsonl"), "--out", str(tmp_path / "out")])
