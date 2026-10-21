"""exp41 readout (critic EXP41-PREP-01, EXP41-R1-01..03; rules written before any run; an ORACLE diagnosis, not a
candidate result). From devision-eval of exp40-M's frozen endpoint on data/exp41/questions.jsonl
(runs/exp41/m.details.jsonl) and, for the original pictures, M's own frozen rl_tune records (not re-run):

  - identity (refused otherwise): M's weights / config and its recorded rl_tune files unchanged since the freeze
    (scripts/v04b/freeze.py check); the exp41 summary names that checkpoint, fitted temperatures, control "none" and
    this question file's sha256 (data/exp41/MANIFEST.json); both record sets hold every question exactly once with the
    data's gold and picture id (devision.train.compare._require_same), valid probabilities, prediction = the most
    probable option, and "correct" = (prediction == gold);
  - conditions: original; keep (= keep_text: the asked object's text block on the original canvas); keep_object
    (the asked object's whole half with its jar and particles; temperature only -- on force the keep picture already
    holds the whole pair and stands for keep_object, no second identical input); zoom (the text block / pair cut out);
    other (the other object's region; the asked number is not in the picture);
  - per form and field (mass / speed / distance), same-value and different-value pairs (the two objects of one view
    and field): questions and right answers; on different-value pairs both objects right, wrong answers equal to the
    other object's gold, and answers equal to the other object's gold ("shown" for the other condition);
  - paired differences, resampled over case-linked clusters (families sharing a canonical case, exp40's algorithm;
    seed 0, 2,000 resamples; a cluster's families, views, objects, fields and conditions together), on different-value
    accuracy and both objects right, per field and form: temperature keep_object - original (the other object removed,
    own cues kept), keep - keep_object (own jar / particles removed), zoom - keep (the same text cut out and enlarged);
    force keep - original, zoom - keep.

No overall verdict. Per field, observations and the explanations NOT excluded: a zoom gain only fits "a larger region
helps" (cropping also changes position and share of the picture), not a font-size cause; no zoom gain does not exclude
readability (zoom ink stays under exp37's validated 16 px: MANIFEST digit_ink_256); mass has the particle-size cue
(original, keep_object only), distance the magnet-gap cue (every force condition); "other" is read on different-value
pairs only. The next training is decided with exp40's O / L results, not by this report alone.

    uv run python scripts/exp41/report.py
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

from devision.train.compare import _require_same, _valid_probabilities

HERE = os.path.dirname(__file__)
M_RUN = "runs/exp40-m"
OUT = "runs/exp41"
DATA = "data/exp41/questions.jsonl"
MANIFEST = "data/exp41/MANIFEST.json"
TUNE = "data/real_layout/questions_tune.jsonl"
CONDITIONS = ("original", "keep", "keep_object", "zoom", "other")
FIELDS = {"temp": ("mass", "speed"), "force": ("distance",)}
DIFFS = {"temp": (("keep_object", "original"), ("keep", "keep_object"), ("zoom", "keep")),
         "force": (("keep", "original"), ("zoom", "keep"))}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frozen = _load("v04b_freeze", "../v04b/freeze.py")
clusters = _load("exp40_report", "../exp40/report.py").clusters


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: str) -> List[Dict]:
    return [json.loads(x) for x in open(path) if x.strip()]


def gold_of(probs: Dict) -> str:
    return next(o for o, p in probs.items() if p == 1.0)


def check_details(name: str, rows: List[Dict], got: List[Dict], only=None) -> Dict:
    """`got` answers each question of `rows` (or the `only` (id, qid) keys of them) exactly once, with the data's gold
    and picture, valid probabilities, prediction = argmax, correct = (prediction == gold). Keyed by (id, qid)."""
    ref = [{"sample_id": r["id"], "qid": q, "gold": r["gold"][q]["probabilities"], "image_id": r["image_id"]}
           for r in rows for q in r["questions"] if only is None or (r["id"], q) in only]
    keys = {(x["sample_id"], x["qid"]) for x in ref}
    sub = [g for g in got if (g["sample_id"], g["qid"]) in keys] if only is not None else got
    try:
        _require_same(ref, sub)
    except ValueError as e:
        raise SystemExit("%s: %s" % (name, e))
    for g in sub:
        if not _valid_probabilities(g) or g["prediction"] != max(g["probabilities"], key=g["probabilities"].get) \
                or g["gold_answer"] != gold_of(g["gold"]) or bool(g["correct"]) != (g["prediction"] == g["gold_answer"]):
            raise SystemExit("%s: inconsistent record %s %s" % (name, g["sample_id"], g["qid"]))
    return {(g["sample_id"], g["qid"]): g for g in sub}


def check_identity(m_run: str = M_RUN, out: str = OUT, data_path: str = DATA, manifest: str = MANIFEST) -> Dict:
    fr = frozen.check(m_run, ["rl_tune"])            # weights, config and the recorded rl_tune files as frozen
    info = json.load(open(os.path.join(out, "m.json")))["run"]
    want = json.load(open(manifest))["sha256"]
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(os.path.join(m_run, "stage2")) \
            or info.get("temperature") != "fitted" or info.get("control") != "none":
        raise SystemExit("exp41 was not evaluated on exp40-M's frozen checkpoint, fitted temperatures, no control")
    if not (info.get("data_sha256") == want == sha(data_path)):
        raise SystemExit("exp41 question file: summary %s, manifest %s, file %s" % (info.get("data_sha256"), want, sha(data_path)))
    return fr["weights"]


def records(data: List[Dict], got: List[Dict], tune_rows: List[Dict], tune_got: List[Dict]) -> List[Dict]:
    """One record per (condition, original picture, object, qid); force keep also stands as keep_object."""
    have = check_details("exp41", data, got)
    want_tune = {(r["orig_id"], q) for r in data for q in r["questions"]}
    tune = check_details("exp40-M rl_tune", tune_rows, tune_got, only=want_tune)
    out = []
    for r in data:
        for q in r["questions"]:
            ins = r["questions"][q]["instructions"]
            f = "mass" if "mass" in ins else "speed" if "speed" in ins else "distance"
            base = {"family": r["family"], "case": r["case"], "kind": r["kind"], "orig_id": r["orig_id"], "object": r["object"], "field": f}
            g = have[(r["id"], q)]
            conds = [r["condition"]] + (["keep_object"] if r["condition"] == "keep" and r["kind"] == "force" else [])
            if r["condition"] == "keep":
                conds.append("original")
            for c in conds:
                x = tune[(r["orig_id"], q)] if c == "original" else g
                out.append(dict(base, condition=c, gold=x["gold_answer"], prediction=x["prediction"], right=bool(x["correct"])))
    return out


def units(recs: List[Dict]) -> Dict:
    """family -> statistic -> [sum, count]; statistics "<condition>|<scope>|<name>", scope = a field, a form or all."""
    pairs = defaultdict(list)
    for r in recs:
        pairs[(r["condition"], r["orig_id"], r["field"])].append(r)
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(family, cond, scopes, name, v):
        for sc in scopes:
            u = fam[family]["%s|%s|%s" % (cond, sc, name)]
            u[0] += v
            u[1] += 1
    for (cond, _, field), v in pairs.items():
        if len(v) != 2:
            raise SystemExit("not two objects for %s %s %s" % (cond, v[0]["orig_id"], field))
        a, b = v
        scopes = (field, a["kind"], "all")
        strat = "same" if a["gold"] == b["gold"] else "different"
        for x, y in ((a, b), (b, a)):
            add(x["family"], cond, scopes, strat, x["right"])
            if strat == "different":
                add(x["family"], cond, scopes, "is_other_gold", x["prediction"] == y["gold"])
                if not x["right"]:
                    add(x["family"], cond, scopes, "wrong_is_other_gold", x["prediction"] == y["gold"])
        if strat == "different":
            add(a["family"], cond, scopes, "both", a["right"] and b["right"])
    return fam


def total(fam, families, stat):
    return (sum(fam[f][stat][0] for f in families if stat in fam[f]), sum(fam[f][stat][1] for f in families if stat in fam[f]))


def value(fam, families, stat) -> float:
    n, d = total(fam, families, stat)
    return n / d if d else float("nan")


def interval(fn, units_, expand, resamples: int = 2000, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    boots = sorted(x for x in (fn(expand([rng.choice(units_) for _ in units_])) for _ in range(resamples)) if x == x)
    return {"value": fn(expand(list(units_))), "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}


def build(m_run: str = M_RUN, out: str = OUT, data_path: str = DATA, manifest: str = MANIFEST) -> Dict:
    weights = check_identity(m_run, out, data_path, manifest)
    data = load(data_path)
    recs = records(data, load(os.path.join(out, "m.details.jsonl")), load(TUNE),
                   load(os.path.join(m_run, "eval", "rl_tune.details.jsonl")))
    fam = units(recs)
    cl = clusters([{"family": r["family"], "case": r["case"]} for r in recs])
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731
    families = sorted(fam)
    ink = json.load(open(manifest))["digit_ink_256"]
    res: Dict = {"identity": {"m": weights, "data_sha256": sha(data_path)}, "families": len(families), "clusters": len(keys),
                 "counts": {}, "differences": {}, "observations": {}}
    for form, fields in FIELDS.items():
        for scope in fields + (form,):
            res["counts"][scope] = {c: {name: list(total(fam, families, "%s|%s|%s" % (c, scope, name)))
                                        for name in ("same", "different", "both", "is_other_gold", "wrong_is_other_gold")}
                                    for c in CONDITIONS if any("%s|%s|" % (c, scope) in k for f in families for k in fam[f])}
            for a, b in DIFFS[form]:
                for name in ("different", "both"):
                    res["differences"]["%s|%s-%s|%s" % (scope, a, b, name)] = interval(
                        lambda fs, a=a, b=b, s=scope, n=name: value(fam, fs, "%s|%s|%s" % (a, s, n)) - value(fam, fs, "%s|%s|%s" % (b, s, n)),
                        keys, expand)
        for field in fields:
            notes = []
            d = res["differences"]
            for a, b in DIFFS[form]:
                v = d["%s|%s-%s|different" % (field, a, b)]
                notes.append("%s - %s different-value accuracy %+.3f [%+.3f, %+.3f]%s" % (
                    a, b, v["value"], v["ci95"][0], v["ci95"][1], " (lower bound > 0)" if v["ci95"][0] > 0 else ""))
            if d["%s|zoom-keep|different" % field]["ci95"][0] <= 0:
                notes.append("no clear zoom gain: readability NOT excluded (zoom digit ink median %.1f px < 16)" % ink["zoom/" + form][1])
            else:
                notes.append("zoom gain fits 'a larger region helps'; not a font-size cause on its own")
            notes.append("not excluded: " + ("the particle-size cue (original / keep_object only)" if field == "mass" else
                                             "the magnet-gap cue (every force condition)" if field == "distance" else
                                             "no drawn speed cue; context / layout changes"))
            notes.append("'other' read on different-value pairs only: shown-number answers %d/%d" % tuple(res["counts"][field]["other"]["is_other_gold"]))
            res["observations"][field] = notes
    return res


def main() -> int:
    res = build()
    json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print("families %d, case clusters %d" % (res["families"], res["clusters"]))
    for scope, conds in res["counts"].items():
        for c, v in conds.items():
            print("%-8s %-11s same %d/%d different %d/%d both %d/%d other-gold %d/%d wrong=other %d/%d" % (
                scope, c, *v["same"], *v["different"], *v["both"], *v["is_other_gold"], *v["wrong_is_other_gold"]))
    for k, v in res["differences"].items():
        print("%s %+.3f %s" % (k, v["value"], [round(x, 3) for x in v["ci95"]]))
    for f, notes in res["observations"].items():
        print(f + ":\n  " + "\n  ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
