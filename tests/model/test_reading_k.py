"""LIVE-K-PREP-01's optional K rule (devision.reading.K_RULE / k_block; docs/experiments/2026-10-19-live-k-plan.md): from
the table actually read it gives exactly EXP70's K block -- the 600 states of data/facts/tune_P.jsonl (EXP56's automatic
tables) become data/protect_p/P_K.jsonl byte for byte, the 3 rows with an unreadable or decimal value staying without K;
off unless the bundle declares it (old bundles unchanged); it changes only temperature questions' state (magnets and
ordinary questions keep their contract), does not depend on question ids or option order, goes through both picture
inputs alike; a bundle with it has another identity, and a changed rule is refused."""
import base64
import io
import json
import os

import pytest
import torch
from conftest import tiny_decider
from PIL import Image

import devision
from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
DOG = "is there a dog ?"
TABLE = [["Sample A", "mass of each particle", "14"], ["Sample B", "mass of each particle", "8"],
         ["Sample A", "average particle speed", "1,360"], ["Sample B", "average particle speed", "1,900"],
         ["Pair 1", "distance between the magnets", "2"], ["Pair 2", "distance between the magnets", "3"]]
has = os.path.exists(os.path.join(ROOT, "data", "protect_p", "P_K.jsonl"))


def table_of(state: str):
    """The table a state was serialised from (exp54's lines after the header), value strings as written."""
    lines = state.split("\n")
    i = lines.index(R.HEADER)
    out = []
    for line in lines[i + 1:]:
        if " = " not in line or ": " not in line:
            break
        obj, rest = line.split(": ", 1)
        q, v = rest.rsplit(" = ", 1)
        out.append([obj, q, v])
    return out


@pytest.mark.skipif(not has, reason="data/protect_p not built")
def test_the_600_automatic_tables_give_p_k_exactly():
    src = [json.loads(x) for x in open(os.path.join(ROOT, "data", "facts", "tune_P.jsonl"))]
    want = {r["id"]: r["state_text"] for r in map(json.loads, open(os.path.join(ROOT, "data", "protect_p", "P_K.jsonl")))}
    without = 0
    for r in src:
        if r["kind"] != "temp":
            assert want[r["id"]] == r["state_text"]                         # magnets: never a K block
            continue
        kb = R.k_block(table_of(r["state_text"]))
        without += kb is None
        assert want[r["id"]] == (r["state_text"] + "\n" + kb if kb else r["state_text"]), r["id"]
    assert len(src) == len(want) == 600 and without == 3


def test_k_block_rule():
    kb = R.k_block(TABLE)
    assert kb == R.K_RULE["header"] + "\nSample A: mass times speed squared = 25,894,400\nSample B: mass times speed squared = 28,880,000"
    for bad in ("unreadable", "1.80"):
        t = [list(x) for x in TABLE]
        t[3][2] = bad
        assert R.k_block(t) is None
    t = [list(x) for x in TABLE]
    t[0][2] = "14.0"                                                         # integral, as EXP70 accepts
    assert R.k_block(t) == kb
    assert R.k_block(TABLE[4:]) is None                                       # no temperature table


def picture():
    return Image.new("RGB", (560, 405), "white")


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def bundles(tmp_path):
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.save(tmp_path / "ck")
    torch.manual_seed(7)
    torch.save(R.Selector(32, 64).state_dict(), tmp_path / "sel.pt")
    torch.save(R.CharHead(32).state_dict(), tmp_path / "head.pt")
    for name, k in (("plain", False), ("k", True)):
        R.write_bundle(str(tmp_path / name), str(tmp_path / "ck"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                       dims={"selector": [32, 64], "head": 32}, k_rule=k)
    return [R.Reading.load(str(tmp_path / "ck"), str(tmp_path / n), device="cpu", full_rules=True) for n in ("plain", "k")]


@pytest.fixture
def readers(tmp_path, monkeypatch):
    monkeypatch.setattr(R.Reading, "read", lambda self, image, templates: {
        "table": [row for row in TABLE if row[:2] in [list(R.TEMPLATES[t]) for t in templates]],
        "reads": [{"question": t} for t in templates], "seconds": 0.0})
    return bundles(tmp_path)


def q(text, opts=("red", "blue")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


def test_off_by_default_and_on_only_for_temperature(readers):
    plain, k = readers
    assert plain.k_rule is False and k.k_rule is True and k.reading_status["k_rule"] is True
    img = picture()
    qs = {"t": q(TEMP), "m": q(MAG), "o": q(DOG)}
    dp, dk = {}, {}
    ap, ak = plain.decide("Two samples.", qs, image=img, diagnostics=dp), k.decide("Two samples.", qs, image=img, diagnostics=dk)
    assert dp["k_block"] is None and dk["k_block"] == R.k_block(TABLE) and dk["state"] == dp["state"] + "\n" + R.k_block(TABLE)
    assert ak["answers"]["m"] == ap["answers"]["m"] and ak["answers"]["o"] == ap["answers"]["o"]     # magnets / ordinary unchanged
    assert dk["route"] == dp["route"] == {"t": "read", "m": "read", "o": "ordinary"}
    want = plain.decider._decide(dp["state"] + "\n" + R.k_block(TABLE), {"t": q(TEMP)}, None)["answers"]["t"]
    assert ak["answers"]["t"] == want                                      # the temperature question sees the K state


def test_k_ignores_ids_and_option_order_and_both_picture_inputs_agree(readers):
    _, k = readers
    img = picture()
    a, b, c = {}, {}, {}
    k.decide("", {"q1": q(TEMP, ("red", "blue"))}, image=img, diagnostics=a)
    k.decide("", {"zz": q(TEMP, ("blue", "red"))}, image=img, diagnostics=b)
    k.decide([{"type": "image", "base64": b64(img)}], {"q1": q(TEMP, ("red", "blue"))}, diagnostics=c)
    assert a["k_block"] == b["k_block"] == c["k_block"] and a["state"] == b["state"] == c["state"]


def test_k_is_another_identity_and_a_changed_rule_is_refused(tmp_path):
    plain, k = bundles(tmp_path)
    assert R.sha(str(tmp_path / "plain" / "reading.json")) != R.sha(str(tmp_path / "k" / "reading.json"))
    assert plain.identity["rules_sha256"] != k.identity["rules_sha256"]
    p = tmp_path / "k" / "reading.json"
    ident = json.loads(p.read_text())
    ident["k_rule"]["formula"] = "mass * speed"
    p.write_text(json.dumps(ident))
    with pytest.raises(ValueError, match="rules differ"):
        R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "k"), device="cpu")


def test_through_the_public_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(R.Reading, "read", lambda self, image, templates: {
        "table": [row for row in TABLE if row[:2] in [list(R.TEMPLATES[t]) for t in templates]], "reads": [], "seconds": 0.0})
    bundles(tmp_path)
    import shutil
    shutil.copytree(tmp_path / "k", tmp_path / "ck" / "reading")
    rd = devision.load(str(tmp_path / "ck"), device="cpu")
    d: dict = {}
    rd.decide("", {"t": q(TEMP)}, image=picture(), diagnostics=d)
    assert rd.reading_status["k_rule"] is True and d["k_block"] == R.k_block(TABLE)
