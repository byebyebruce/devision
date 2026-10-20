"""temperature_by_route (critic V04-K-CALIB-RESULT-01 / V04-ROUTE-TEMP-PREP-01): a read question is scored with its
domain's temperature, applied once on the logits -- softmax(logits / T_domain); in a mixed request each question keeps
its own temperature and the question order does not matter; without K or with a failed read the domain still decides;
ordinary questions, requests without a picture, reading off and configs without the field keep the old temperatures;
raw switches every temperature off; 0 / negative / NaN / Inf / unknown domains are refused; save / load and the formal
evaluation record the same field and give the same probabilities."""
import math

import pytest
import torch
from conftest import tiny_decider
from PIL import Image

from devision import reading as R
from devision.infer import Decider
from devision.model import ModelConfig

TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
DOG = "is there a dog ?"
TABLE = [["Sample A", "mass of each particle", "14"], ["Sample B", "mass of each particle", "8"],
         ["Sample A", "average particle speed", "1,360"], ["Sample B", "average particle speed", "1,900"],
         ["Pair 1", "distance between the magnets", "2"], ["Pair 2", "distance between the magnets", "3"]]


def q(text, opts=("red", "blue", "green")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


@pytest.fixture(autouse=True)
def fixed_reads(monkeypatch):
    monkeypatch.setattr(R.Reading, "read", lambda self, image, templates: {
        "table": [row for row in (self._table if hasattr(self, "_table") else TABLE) if row[:2] in [list(R.TEMPLATES[t]) for t in templates]],
        "reads": [{"question": t} for t in templates], "seconds": 0.0})


def reader(tmp_path, name, by_route=None, temperature=(1.4, 1.0, 1.1), k_rule=True):
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.cfg.temperature = list(temperature)
    d.cfg.temperature_by_route = dict(by_route or {})
    d.save(tmp_path / name)
    torch.manual_seed(7)
    torch.save(R.Selector(32, 64).state_dict(), tmp_path / "sel.pt")
    torch.save(R.CharHead(32).state_dict(), tmp_path / "head.pt")
    R.write_bundle(str(tmp_path / name / "reading"), str(tmp_path / name), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, k_rule=k_rule)
    return R.Reading.load(str(tmp_path / name), str(tmp_path / name / "reading"), device="cpu", full_rules=True)


def probs(ans):
    return list(ans["probabilities"].values())


def raw_logits(rd, state, qs, img):
    """log of the T = 1 probabilities (= the logits up to a constant per question)."""
    saved = (list(rd.cfg.temperature), dict(rd.cfg.temperature_by_options), dict(rd.cfg.temperature_by_route))
    rd.use_raw_temperatures()
    out = {k: [math.log(p) for p in probs(a)] for k, a in rd.decide(state, qs, image=img)["answers"].items()}
    rd.cfg.temperature, rd.cfg.temperature_by_options, rd.cfg.temperature_by_route = saved
    return out


def softmax(z, t):
    m = max(x / t for x in z)
    e = [math.exp(x / t - m) for x in z]
    return [x / sum(e) for x in e]


def test_read_questions_take_their_domain_temperature_once(tmp_path):
    rd = reader(tmp_path, "ck", {"temperature": 3.0, "magnets": 5.0})
    img = Image.new("RGB", (560, 405), "white")
    qs = {"t": q(TEMP), "m": q(MAG), "o": q(DOG)}
    z = raw_logits(rd, "x", qs, img)
    got = rd.decide("x", qs, image=img)["answers"]
    assert probs(got["t"]) == pytest.approx(softmax(z["t"], 3.0), abs=1e-6)
    assert probs(got["m"]) == pytest.approx(softmax(z["m"], 5.0), abs=1e-6)
    assert probs(got["o"]) == pytest.approx(softmax(z["o"], 1.4), abs=1e-6)                       # ordinary: type temperature
    p = got["t"]["probabilities"]
    best = max(p, key=p.get)
    assert got["t"]["confidence"] == pytest.approx((3 * p[best] - 1) / 2)                        # confidence from the final probabilities
    order = rd.decide("x", {"o": qs["o"], "m": qs["m"], "t": qs["t"]}, image=img)["answers"]
    assert all(probs(order[k]) == pytest.approx(probs(got[k]), abs=1e-6) for k in qs)          # order does not matter


def test_no_k_or_a_failed_read_still_uses_the_domain(tmp_path):
    rd = reader(tmp_path, "nok", {"temperature": 3.0}, k_rule=False)
    setattr(rd, "_table", [[o, qn, "unreadable" if qn == "average particle speed" else v] for o, qn, v in TABLE])
    img = Image.new("RGB", (560, 405), "white")
    qs = {"t": q(TEMP)}
    z = raw_logits(rd, "", qs, img)
    assert probs(rd.decide("", qs, image=img)["answers"]["t"]) == pytest.approx(softmax(z["t"], 3.0), abs=1e-6)


def test_ordinary_paths_and_old_configs_keep_their_temperatures(tmp_path):
    rd = reader(tmp_path, "ck", {"temperature": 3.0, "magnets": 5.0})
    old = reader(tmp_path, "old", None)
    img = Image.new("RGB", (560, 405), "white")
    assert rd.decide("x", {"o": q(DOG)}, image=img) == old.decide("x", {"o": q(DOG)}, image=img)          # ordinary question
    assert rd.decide("x", {"t": q(TEMP)}) == old.decide("x", {"t": q(TEMP)})                             # no picture: not read
    off = Decider.load(str(tmp_path / "ck"), device="cpu")                                              # reading off
    plain = Decider.load(str(tmp_path / "old"), device="cpu")
    assert off.cfg.temperature_by_route == {"temperature": 3.0, "magnets": 5.0}
    assert off.decide("x", {"t": q(TEMP)}, image=img) == plain.decide("x", {"t": q(TEMP)}, image=img)    # the plain Decider ignores it
    assert ModelConfig.from_dict({"temperature": [1.0, 1.0, 1.0]}).temperature_by_route == {}          # an old config
    assert "temperature_by_route" not in ModelConfig().to_dict()                                        # saved exactly as before when empty


def test_raw_switches_every_temperature_off(tmp_path):
    rd = reader(tmp_path, "ck", {"temperature": 3.0, "magnets": 5.0})
    rd.use_raw_temperatures()
    assert rd.cfg.temperature == [1.0, 1.0, 1.0] and rd.cfg.temperature_by_options == {} and rd.cfg.temperature_by_route == {}


@pytest.mark.parametrize("bad", [{"temperature": 0.0}, {"temperature": -1.0}, {"magnets": float("nan")}, {"magnets": float("inf")},
                                 {"other": 2.0}, {"temperature": True}])
def test_invalid_route_temperatures_are_refused(bad):
    with pytest.raises(ValueError, match="temperature_by_route"):
        ModelConfig.from_dict({"temperature_by_route": bad})


def test_save_load_and_the_formal_evaluation_agree(tmp_path):
    import json
    import os

    from devision.train.evaluate import evaluate
    rd = reader(tmp_path, "ck", {"temperature": 3.0, "magnets": 5.0})
    again = R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "ck" / "reading"), device="cpu", full_rules=True)
    assert json.load(open(tmp_path / "ck" / "config.json"))["temperature_by_route"] == {"temperature": 3.0, "magnets": 5.0}
    img = Image.new("RGB", (560, 405), "white")
    qs = {"t": q(TEMP), "m": q(MAG)}
    assert again.decide("x", qs, image=img) == rd.decide("x", qs, image=img)
    os.makedirs(tmp_path / "img")
    img.save(tmp_path / "img" / "a.png")
    rows = [{"id": "s0", "image": "img/a.png", "image_id": "a", "state_text": "x", "questions": qs,
             "gold": {k: {"probabilities": {"red": 1.0, "blue": 0.0, "green": 0.0}} for k in qs}}]
    recs: list = []
    out = evaluate(again, rows, tmp_path, records_out=recs)
    assert out["temperature"]["by_route"] == {"temperature": 3.0, "magnets": 5.0}
    direct = rd.decide("x", qs, image=img)["answers"]
    assert all([r["probabilities"][o] for o in ("red", "blue", "green")] == pytest.approx(probs(direct[r["qid"]]), abs=1e-6) for r in recs)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ce_rt", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "release", "check_eval.py"))
    assert spec and spec.loader
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    assert ce.temperature_table(str(tmp_path / "ck")) == out["temperature"]                 # the frozen table is what evaluation records
    old = reader(tmp_path, "old", None)
    assert "by_route" not in evaluate(old, rows, tmp_path)["temperature"] and "by_route" not in ce.temperature_table(str(tmp_path / "old"))
