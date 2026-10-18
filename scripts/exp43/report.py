"""exp43 readout (critic EXP43-PREP-03; rules written before the run; ORACLE diagnosis of the one fixed endpoint
exp42-C, not a candidate result and not a release gate). All sets at temperature 1 through decide:
  - name (reused, identity re-checked): exp42-C's own fit_O / fit_L / rl_tune / rl_tune_L records, recorded under its
    frozen weights / config (scripts/v04b/freeze.py);
  - new (runs/exp43/eval): fit_spatial_O / _L, tune_spatial_O / _L, fit_transcript, tune_transcript.
Each summary must name runs/exp42-c/stage2, "none (raw)", no control and the file's sha256 (data/exp43/MANIFEST.json for
the new ones); each record set every question once with the data's gold / picture id, valid probabilities, argmax
prediction and consistent correct (scripts/exp41/report.check_details); the transcript records must show the blank
picture as image_used. Anything missing: "incomplete".

Per set and field (reading questions only): different (single questions where the two objects' golds differ on that
picture), both, twin (gold changes between the views, both right), same_answer -- exp42's statistics, field-equal
macros; name -> spatial differences of both (and different, twin) resampled over case-linked clusters; prediction
changes name -> spatial on different-value questions (changed, wrong->right, right->wrong); different-value accuracy by
where the first object (A / Pair 1) is drawn (report only).

Screens (diagnostic, pre-written):
  text_usable: tune_transcript different macro >= 0.90, both >= 0.80, twin >= 0.80;
  spatial_rescue_O / _L: tune spatial both macro >= 0.20 and spatial - name both >= 0.10 with lower bound > 0, per
    drawing; only L passing does not count as the real input fixed; fit is supporting only.
Evidence pattern for the critic (no automatic next step): 1 = text usable and O rescued; 2 = text usable, O not
rescued; 3 = text not usable; incomplete otherwise.

    uv run python scripts/exp43/report.py
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(__file__)
C_RUN, CKPT, OUT = "runs/exp42-c", "runs/exp42-c/stage2", "runs/exp43"
C_SHA = {"model.safetensors": "6596907825f0b9c1ba9c0e9eb8dc136a91d6c2a48b331bc50ba34ea60ad36ca4",
         "config.json": "e5421f88ff74a665d4730725e56bb49246eed419002ca064d7f308baab0f89f9"}
NAME = {"fit_O": "data/exp42/fit_O.jsonl", "fit_L": "data/exp42/fit_L.jsonl",
        "rl_tune": "data/real_layout/questions_tune.jsonl", "rl_tune_L": "data/exp40/rl_tune_L.jsonl"}
NEW = {n: "data/exp43/%s.jsonl" % n for n in ("fit_spatial_O", "fit_spatial_L", "tune_spatial_O", "tune_spatial_L",
                                             "fit_transcript", "tune_transcript")}
PAIRS = {"fit_spatial_O": "fit_O", "fit_spatial_L": "fit_L", "tune_spatial_O": "rl_tune", "tune_spatial_L": "rl_tune_L",
         "fit_transcript": "fit_O", "tune_transcript": "rl_tune"}
BLANK = "exp21/images/blank.png"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r42 = _load("exp42_report", "../exp42/report.py")
r41, b40 = r42.r41, r42.b40


def records(run: str, name: str, data_path: str, want_sha: str) -> List[Dict]:
    info = json.load(open(os.path.join(run, "eval", name + ".json")))["run"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(CKPT) or info.get("temperature") != "none (raw)" \
            or info.get("control") != "none" or info.get("data_sha256") != want_sha or r41.sha(data_path) != want_sha:
        raise SystemExit("%s/%s: not exp42-C at temperature 1, no control, on %s" % (run, name, data_path))
    rows = r42.load(data_path)
    got_list = r42.load(os.path.join(run, "eval", name + ".details.jsonl"))
    got = r41.check_details("%s/%s" % (run, name), rows, got_list)
    out = []
    for r in rows:
        view = r.get("view") or r["id"].split(":")[2]
        for q, v in r["questions"].items():
            f = b40.field_of(v["instructions"])
            if f == "cmp":
                continue
            g = got[(r["id"], q)]
            if r.get("condition") == "transcript" and g.get("image_used") != BLANK:
                raise SystemExit("%s/%s: transcript record %s did not use the blank picture" % (run, name, r["id"]))
            obj = r["objects"][q] if "objects" in r else b40.object_of(v["instructions"])
            out.append({"family": r["family"], "case": r["case"], "field": f, "object": obj, "view": view,
                        "picture": "%s:%s" % (r["family"], view), "key": (r["orig_id"] if "condition" in r else r["id"], q),   # exp43 rows point to the name row
                        "first_right": r.get("sides", {}).get(q), "gold": g["gold_answer"],
                        "prediction": g["prediction"], "right": bool(g["correct"])})
    return out


def transitions(name_recs: List[Dict], new_recs: List[Dict]) -> Dict:
    """Name -> new on the different-value questions (paired by the original question)."""
    other = defaultdict(list)
    for r in name_recs:
        other[(r["picture"], r["field"])].append(r)
    diff_keys = {r["key"] for v in other.values() if len(v) == 2 and v[0]["gold"] != v[1]["gold"] for r in v}
    by = {r["key"]: r for r in name_recs}
    out = {"questions": 0, "changed": 0, "wrong_to_right": 0, "right_to_wrong": 0}
    for r in new_recs:
        a = by.get(r["key"])
        if a is None:
            raise SystemExit("no name record for %s" % (r["key"],))
        if r["key"] in diff_keys:
            out["questions"] += 1
            out["changed"] += a["prediction"] != r["prediction"]
            out["wrong_to_right"] += (not a["right"]) and r["right"]
            out["right_to_wrong"] += a["right"] and not r["right"]
    return out


def by_layout(recs: List[Dict], sides: Dict) -> Dict:
    """Different-value accuracy by where the first object (A / Pair 1) is drawn (sides from the spatial rows)."""
    pairs = defaultdict(list)
    for r in recs:
        pairs[(r["picture"], r["field"])].append(r)
    out: Dict = defaultdict(lambda: [0, 0])
    for (pic, _), v in pairs.items():
        if len(v) == 2 and v[0]["gold"] != v[1]["gold"]:
            lay = sides.get(pic, "?")
            for r in v:
                out[lay][0] += r["right"]
                out[lay][1] += 1
    return {k: {"right": a, "of": n} for k, (a, n) in sorted(out.items())}


def build(out_dir: str = OUT, manifest_path: str = "data/exp43/MANIFEST.json") -> Dict:
    man = json.load(open(manifest_path))
    fr = r41.frozen.check(C_RUN, list(NAME))["weights"]
    if fr != C_SHA:
        raise SystemExit("exp42-C is not 65969078 / e5421f88")
    res: Dict = {"identity": {"exp42_c": fr, "data": {n: f["sha256"] for n, f in man["files"].items()}},
                 "readout": {}, "differences": {}, "transitions": {}, "by_layout": {}, "missing": []}
    recs = {n: records(C_RUN, n, p, r41.sha(p)) for n, p in NAME.items()}
    st = r42.states(os.path.join(out_dir + "-status.txt")) if os.path.exists(out_dir + "-status.txt") else {}
    for n, p in NEW.items():
        if st.get("eval:" + n) != "completed":
            res["missing"].append(n)
            continue
        recs[n] = records(out_dir, n, p, man["files"][n]["sha256"])
    first_side = {}
    for n in ("tune_spatial_O", "fit_spatial_O"):
        for r in r42.load(NEW[n]):
            for q, o in r["objects"].items():
                if o.split()[-1] in ("A", "1"):
                    first_side["%s:%s" % (r["family"], r["view"])] = "first_" + r["sides"][q]
    fams = {n: r42.units(v) for n, v in recs.items()}
    for n, f in fams.items():
        res["readout"][n] = r42.summary(f)
        res["by_layout"][n] = by_layout(recs[n], first_side)
    for n, base in PAIRS.items():
        if n in recs:
            ca, cb = r42.clusters(recs[n]), r42.clusters(recs[base])
            if ca != cb:
                raise SystemExit("%s and %s were read on different questions" % (n, base))
            res["differences"]["%s-%s" % (n, base)] = {s: r42.diff(fams[n], fams[base], ca, s) for s in ("different", "both", "twin")}
            res["transitions"]["%s-%s" % (n, base)] = transitions(recs[base], recs[n])
    rd, d = res["readout"], res["differences"]
    tu = rd.get("tune_transcript")
    res["screens"] = {"text_usable": "incomplete" if tu is None else bool(tu["different"] >= 0.90 and tu["both"] >= 0.80 and tu["twin"] >= 0.80)}
    for k, n, base in (("O", "tune_spatial_O", "rl_tune"), ("L", "tune_spatial_L", "rl_tune_L")):
        key = "%s-%s" % (n, base)
        res["screens"]["spatial_rescue_" + k] = "incomplete" if key not in d else bool(
            rd[n]["both"] >= 0.20 and d[key]["both"]["value"] >= 0.10 and d[key]["both"]["ci95"][0] > 0)
    s = res["screens"]
    res["pattern"] = ("incomplete" if "incomplete" in (s["text_usable"], s["spatial_rescue_O"]) else
                      1 if s["text_usable"] and s["spatial_rescue_O"] else 2 if s["text_usable"] else 3)
    return res


def main() -> int:
    res = build()
    json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    print("identity %s; missing %s" % (res["identity"]["exp42_c"], res["missing"] or "none"))
    for n, s in res["readout"].items():
        print("%-15s different %.3f both %.3f twin %.3f same_answer %.3f | %s" % (
            n, s["different"], s["both"], s["twin"], s["same_answer"],
            {k: "%d/%d" % (v["right"], v["of"]) for k, v in s.items() if isinstance(v, dict) and not k.startswith("same")}))
    for k, v in res["differences"].items():
        print("%s %s | transitions %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()},
                                          res["transitions"][k]))
    for n, v in res["by_layout"].items():
        print("by layout %s %s" % (n, {k: "%d/%d" % (x["right"], x["of"]) for k, x in v.items()}))
    print("screens %s; evidence pattern %s" % (res["screens"], res["pattern"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
