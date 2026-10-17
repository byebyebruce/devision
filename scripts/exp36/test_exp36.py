"""exp36 without a model: the data split (both views of a family, 1,000 steps of 16, fit slice from the chosen
families) and the readout statistics (family-paired B - A, both-objects / twin denominators, picture dependence only
on changed values whose wrong picture is the twin, the continuation conditions)."""
import importlib.util
import json
import os

HERE = os.path.dirname(__file__)


def load(name):
    spec = importlib.util.spec_from_file_location("exp36_" + name, os.path.join(HERE, name + ".py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records(right_real, right_wrong, families=40):
    """Two views per family; temperature families: A / B masses differ, the twin swaps them."""
    out = []
    for f in range(families):
        fam = "rl%05d" % f
        for v in (0, 1):
            sid = "real-layout:%s:v%d" % (fam, v)
            vals = {"Sample A": ("10", "500"), "Sample B": ("20", "500")}
            if v:
                vals = {"Sample A": vals["Sample B"], "Sample B": vals["Sample A"]}
            for obj, (mass, speed) in vals.items():
                for field, text in (("mass", mass), ("speed", speed)):
                    for mode in ("real", "wrong"):
                        ok = (right_real if mode == "real" else right_wrong)(f, field)
                        out.append({"sample_id": sid, "family": fam, "field": field, "object": obj, "text": text,
                                    "right": ok, "mode": mode, "wrong_picture_kind": "twin, numbers changed"})
    return out


def test_statistics_and_conditions(tmp_path):
    r = load("report")
    arms = {}
    for arm, real, wrong in (("A", lambda f, x: False, lambda f, x: False), ("B", lambda f, x: True, lambda f, x: False)):
        for split in ("tune", "fit"):
            d = tmp_path / arm / ("numbers_" + split)
            d.mkdir(parents=True)
            with open(d / "numbers.jsonl", "w") as fh:
                for x in records(real, wrong):
                    fh.write(json.dumps(x) + "\n")
        arms[arm] = str(tmp_path / arm)
    fa, fb = (r.units(r.load(os.path.join(arms[a], "numbers_tune", "numbers.jsonl"))) for a in ("A", "B"))
    sb = r.arm_stats(fb)
    assert sb["denominators"]["both:mass"] == 80 and sb["denominators"]["twin:mass"] == 80   # masses differ / change
    assert sb["denominators"]["both:speed"] == 0 and sb["denominators"]["twin:speed"] == 0   # speeds equal, unchanged
    assert sb["dep"]["n"] == 160 and sb["dep"]["value"] == 1.0                               # only the changed masses
    families = sorted(fb)
    d = r.interval(lambda fs: r.value(fb, fs, "whole") - r.value(fa, fs, "whole"), families)
    assert d["value"] == 1.0 and d["ci95"][0] == 1.0


def test_split(tmp_path):
    b = load("build")
    (tmp_path / "real_layout").mkdir()
    (tmp_path / "exp32").mkdir()
    with open(tmp_path / "real_layout" / "captions_train.jsonl", "w") as fh:
        for f in range(40):
            for v in (0, 1):
                src = "real-layout-temp" if f % 2 else "real-layout-force"
                fh.write(json.dumps({"id": "real-layout:rl%05d:v%d" % (f, v), "family": "rl%05d" % f, "source": src,
                                     "captions": ["x"]}) + "\n")
    with open(tmp_path / "exp32" / "align_mix.jsonl", "w") as fh:
        for i in range(50):
            fh.write(json.dumps({"id": "coco:%d" % i, "source": "coco-captions", "captions": ["y"]}) + "\n")
    m = b.build(str(tmp_path), n_fam=10, n_coco=16, n_fit=3)
    rows = [json.loads(x) for x in open(tmp_path / "exp36" / "align_mix.jsonl")]
    assert m["align_mix"]["rows"] == 56 and sum(r["source"] == "coco-captions" for r in rows) == 16
    fams = {}
    for x in rows:
        fams.setdefault(x.get("family"), []).append(x["id"])
    assert all(len(v) == 2 for k, v in fams.items() if k)
    fit = [json.loads(x) for x in open(tmp_path / "exp36" / "fit_slice.jsonl")]
    chosen = {f for v in m["families_chosen"].values() for f in v}
    assert len(fit) == 12 and {x["family"] for x in fit} <= chosen
    assert all(fit[i]["family"] == fit[i + 1]["family"] for i in range(0, 12, 2))


def test_twin_macro_is_equal_weight_and_kinds_recombine():
    r = load("report")
    fam = {"f1": {"twin:mass": [1.0, 1.0], "twin:speed": [0.0, 9.0], "twin:distance": [1.0, 2.0],
                  "kind:twin, numbers changed:real": [3.0, 4.0], "kind:twin, numbers changed:wrong": [1.0, 4.0],
                  "kind:other family, same form:real": [1.0, 2.0], "kind:other family, same form:wrong": [1.0, 2.0]}}
    assert abs(r.value(fam, ["f1"], "twin") - (1.0 + 0.0 + 0.5) / 3) < 1e-12        # not (2 / 12)
    k = r.by_kind(fam)
    assert k["twin, numbers changed"]["n"] == 4 and k["twin, numbers changed"]["real_minus_wrong"] == 0.5
    assert sum(v["n"] for v in k.values()) == 6
