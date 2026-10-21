"""The optional reading mode (devision.reading; docs/spec/reading-mode.md): the migrated components behave exactly as the
experiment originals (random weights); routing depends on the natural question text only -- question ids and option
order do not change the internal reads, a single reading question, a comparison question alone and several questions
all read their domain, unknown / text-only requests are ordinary decide; only instructions / type / criteria of a
question are touched; the response keeps the Jev structure; a saved bundle loads to the same answers and a bundle that
does not bind its files is refused."""
import importlib.util
import json
import os

import pytest
import torch
from conftest import tiny_decider
from PIL import Image, ImageDraw

from devision import reading as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_components_match_the_originals():
    sel53 = load("exp53_selector", "scripts/exp53/selector.py")
    p49 = load("exp49_probe", "scripts/exp49/probe.py")
    reg51 = load("exp51_regions", "scripts/exp51/regions.py")
    e55 = load("exp55", "scripts/exp55/exp55.py")
    torch.manual_seed(0)
    a, b = sel53.Selector(32, 64).eval(), R.Selector(32, 64).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    x = (torch.randn(3, 256, 32), torch.rand(3, 256, 4), torch.randn(3, 7, 64), torch.ones(3, 7, dtype=torch.bool))
    assert torch.equal(a(*x), b(*x))
    h1, h2 = p49.Head(32).eval(), R.CharHead(32).eval()
    h2.load_state_dict(h1.state_dict(), strict=True)
    y = (torch.randn(2, 9, 32), torch.ones(2, 9, dtype=torch.bool), torch.randint(0, 16, (2, 9, 2)), torch.rand(2, 9, 4))
    assert torch.equal(h1(*y), h2(*y))
    for box, w, h in (([10, 300, 260, 26], 560, 405), ([3.5, 120, 200, 18.2], 650, 140), ([0, 0, 1, 1], 30, 20)):
        assert R.region(box, w, h) == reg51.region(box, w, h)
    for f in ([0.5, 0.5, 0.2, 0.1], [0.99, 0.5, 0.5, 0.0001], [0, 0, 0, 0]):
        assert R.pixel_box(f, 500, 400) == sel53.pixel_box(f, 500, 400)
    for ids in ([1, 4, 11, -100, -100], [1, 2, 3, 4, 5], [10, 1, 11, 0, 0], [11, 0, 0, 0, 0]):
        assert R.decode(ids) == p49.decode(ids)
    for v in ("1360", "14", "0.75", "1234567.25", R.UNREADABLE):
        assert R.fmt(v) == e55.fmt(v)


TEMP = "Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?"
MAG = "Think about the magnetic force between the magnets in each pair. Which of the following statements is true?"
MASS_A = "What is the mass of each particle in sample A?"


def q(text, opts=("red", "blue")):
    return {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}


def test_routing_by_the_natural_question_only():
    four, two = list(R.TEMPLATES)[:4], list(R.TEMPLATES)[4:]
    assert R.reads_for({"x": q(MASS_A)}) == four                               # a single reading question
    assert R.reads_for({"zz": q(TEMP, ("blue", "red"))}) == four       # the comparison alone
    assert R.reads_for({"a": q(MAG), "b": q(list(R.TEMPLATES)[5])}) == two
    assert R.reads_for({"b": q(MASS_A), "a": q(TEMP)}) == R.reads_for({"q7": q(TEMP), "q1": q(MASS_A, ("blue", "red"))})
    assert R.reads_for({"a": q("What color is the dog?")}) == [] and R.reads_for({}) == []
    assert R.domain_of("  what is THE mass of each particle in sample b?  ") == "temperature"


def tiny_reading(tmp_path):
    d = tiny_decider(seed=3)
    d.model.add_question_conditioning()
    d.cfg.question_conditioning = True
    d.model.enable_local_views(12)
    d.cfg.local_views, d.cfg.local_overlap, d.cfg.local_source = 12, 0.25, "T"
    d.save(tmp_path / "ck")
    torch.manual_seed(7)
    sel, head = R.Selector(32, 64), R.CharHead(32)
    torch.save(sel.state_dict(), tmp_path / "sel.pt")
    torch.save(head.state_dict(), tmp_path / "head.pt")
    R.write_bundle(str(tmp_path / "bundle"), str(tmp_path / "ck"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32})
    rd = R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle"), device="cpu")
    return rd.decider, rd


def picture():
    img = Image.new("RGB", (560, 405), "white")
    ImageDraw.Draw(img).text((20, 380), "Mass 14 u   speed 1,360 m/s", fill="black")
    return img


class Watched(dict):
    seen: set = set()

    def __getitem__(self, k):
        Watched.seen.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        Watched.seen.add(k)
        return super().get(k, default)


def test_reading_mode_requests(tmp_path):
    d, rd = tiny_reading(tmp_path)
    img = picture()
    plain = {"a": q("is there a dog ?")}
    assert rd.decide({}, plain, image=img) == d.decide({}, plain, image=img)            # unsupported: ordinary decide
    assert rd.decide("some text", {"x": q(MASS_A)}) == d.decide("some text", {"x": q(MASS_A)})   # no image: ordinary
    for qs in ({"x": q(MASS_A)}, {"x": q(TEMP, ("red", "blue"))}, {"m": q(MASS_A), "t": q(TEMP), "o": q("is there a dog ?")}):
        diag: dict = {}
        res = rd.decide("Two samples.", qs, image=img, diagnostics=diag)
        assert set(res) == {"model", "answers", "usage"} and set(res["answers"]) == set(qs)  # Jev structure only
        assert [r["question"] for r in diag["reads"]] == list(R.TEMPLATES)[:4] and len(diag["table"]) == 4
        assert diag["state"].startswith("Two samples.\n" + R.HEADER)
    mixed = {"m": q(MASS_A), "o": q("is there a dog ?")}
    assert rd.decide("", mixed, image=img)["answers"]["o"] == d.decide("", {"o": q("is there a dog ?")}, image=img)["answers"]["o"]
    d1, d2 = {}, {}
    rd.decide("", {"q1": q(TEMP, ("red", "blue"))}, image=img, diagnostics=d1)
    rd.decide("", {"zz": q(TEMP, ("blue", "red"))}, image=img, diagnostics=d2)    # other qid / option order
    assert d1["reads"] == d2["reads"] and d1["table"] == d2["table"]
    Watched.seen = set()
    rd.decide("", {"w": Watched(q(TEMP), id="s1", kind="temp", gold="A", image="f.png")}, image=img)
    assert Watched.seen <= {"instructions", "type", "criteria"}                            # no dataset fields read


def test_bundle_round_trip_and_refusal(tmp_path):
    d, rd = tiny_reading(tmp_path)
    img = picture()
    a = rd.decide("x", {"t": q(TEMP, ("red", "blue"))}, image=img)
    rd2 = R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle"), device="cpu")
    assert rd2.decide("x", {"t": q(TEMP, ("red", "blue"))}, image=img) == a
    ident = json.loads((tmp_path / "bundle" / "reading.json").read_text())
    assert set(ident["sha256"]) == {"selector.pt", "head.pt", "model.safetensors", "config.json", "tokenizer"}
    with open(tmp_path / "bundle" / "head.pt", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ValueError, match="does not bind"):
        R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle"), device="cpu")


def test_the_verified_entry_loads_its_own_decider(tmp_path):
    """critic EXP56-R1-03: the verified entry takes no caller model; a different checkpoint in the bound directory (or
    the right directory with another model's files) is refused before anything is loaded."""
    import inspect
    import shutil
    d, rd = tiny_reading(tmp_path)
    assert "decider" not in inspect.signature(R.Reading.load).parameters
    other = tiny_decider(seed=11)                                              # another model, same structure
    other.model.add_question_conditioning()
    other.cfg.question_conditioning = True
    other.model.enable_local_views(12)
    other.cfg.local_views, other.cfg.local_overlap, other.cfg.local_source = 12, 0.25, "T"
    other.save(tmp_path / "other")
    shutil.copy(tmp_path / "other" / "model.safetensors", tmp_path / "ck" / "model.safetensors")
    with pytest.raises(ValueError, match="model.safetensors"):
        R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle"), device="cpu")


def test_needs_the_local_view_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="local views"):
        R.Reading(tiny_decider(), R.Selector(32, 64), R.CharHead(32), {})


def test_corrector_matches_exp69():
    """The optional speed corrector (exp69) migrated unchanged: window, local tokens, pixel box and masked forward."""
    import random
    e69 = load("exp69_runner", "scripts/exp69/exp69.py")
    rng = random.Random(0)
    for _ in range(15):
        w, h = rng.randint(400, 650), rng.randint(300, 420)
        box = [rng.uniform(0, w / 2), rng.uniform(h / 2, h - 40), rng.uniform(100, w / 2), rng.uniform(20, 38)]
        win = R.corrector_window(box, w, h)
        assert win == e69.window_of(box, w, h) and R.corrector_tokens(win, w, h) == e69.window_tokens(win, w, h)
        f = [rng.random(), rng.random(), rng.random() * 0.8, rng.random() * 0.8]
        assert R.corrector_box(f, win, w, h) == e69.to_pixels(f, win, w, h)
    torch.manual_seed(0)
    m = R.Selector(32, 64).eval()
    x, g = torch.randn(2, 30, 32), torch.rand(2, 30, 4)
    pm = torch.ones(2, 30, dtype=torch.bool)
    pm[1, 20:] = False
    w_, wm = torch.randn(2, 5, 64), torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(R.corrector_forward(m, x, g, pm, w_, wm), e69.corrector_forward(m, x, g, pm, w_, wm))
    assert R.SPEED == e69.SPEED


def corrected_reading(tmp_path):
    d, rd = tiny_reading(tmp_path)
    torch.manual_seed(9)
    torch.save(R.Selector(32, 64).state_dict(), tmp_path / "corr.pt")
    R.write_bundle(str(tmp_path / "bundle_c"), str(tmp_path / "ck"), str(tmp_path / "sel.pt"), str(tmp_path / "head.pt"), {},
                   dims={"selector": [32, 64], "head": 32}, corrector_pt=str(tmp_path / "corr.pt"))
    return rd, R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle_c"), device="cpu")


def test_optional_corrector_changes_only_speed_boxes(tmp_path):
    rd, rc = corrected_reading(tmp_path)
    assert rd.corrector is None and rc.corrector is not None
    img = picture()
    a = rd.read(img, list(R.TEMPLATES))
    b = rc.read(img, list(R.TEMPLATES))
    for x, y in zip(a["reads"], b["reads"]):
        if x["question"] in R.SPEED:
            assert y["selector_box"] == x["box"] and y["box"] != x["box"] and not y["degenerate"]
            win = R.corrector_window(x["box"], 560, 405)
            assert win[0] <= y["box"][0] and y["box"][0] + y["box"][2] <= win[2] + 1e-9
        else:
            assert y == x                                                        # mass / distance: the uncorrected read
    q2 = {"m": q(MAG, ("red", "blue"))}
    assert rd.decide("x", {"m": q2["m"]}, image=img) == rc.decide("x", {"m": q2["m"]}, image=img)   # magnets untouched
    ident = json.loads((tmp_path / "bundle_c" / "reading.json").read_text())
    assert ident["corrector"] == R.CORRECTOR and "corrector.pt" in ident["sha256"]
    plain = json.loads((tmp_path / "bundle" / "reading.json").read_text())
    assert "corrector" not in plain and "corrector.pt" not in plain["sha256"]                     # off: the earlier bundle


@pytest.mark.parametrize("change", ["file", "rule"])
def test_corrector_bundle_refusal(tmp_path, change):
    corrected_reading(tmp_path)
    if change == "file":
        with open(tmp_path / "bundle_c" / "corrector.pt", "ab") as fh:
            fh.write(b"x")
    else:
        p = tmp_path / "bundle_c" / "reading.json"
        ident = json.loads(p.read_text())
        ident["corrector"]["grow_x_px"] = 12.0
        p.write_text(json.dumps(ident))
    with pytest.raises(ValueError, match="does not bind" if change == "file" else "rules differ from this code"):
        R.Reading.load(str(tmp_path / "ck"), str(tmp_path / "bundle_c"), device="cpu")
