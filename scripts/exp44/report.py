"""exp44 readout (critic EXP44-PREP-03; rules written before the run). J = exp42-C's recipe + the number
reconstruction loss, from exp40-M's frozen end; M = that start (exp42's temperature-1 records, runs/exp42-start);
C = exp42-C (its frozen records). Main evidence: the ordinary decide path, original pictures, name questions,
temperature 1 -- fit_O / fit_L / rl_tune / rl_tune_L, read with exp42's statistics (scripts/exp42/report.py: different,
both, twin, same_answer; field-equal macros; case-linked clusters) and J - M, J - C. exist / science: risk only.

Identity (refused otherwise): exp40-M and exp42-C still their frozen identities; J's START.json / run.json /
train_report.json name objective ce, aux weight 1.0, 8 slots, exp40-M's weights / config and data/exp42/train.jsonl's
sha, and its head identity equals data/exp44/CHECK.json's; every J file recorded under J's frozen weights / config; every
summary the right checkpoint, temperature "none (raw)", no control, the set's sha256; every record set complete and
consistent (exp41's check_details). Anything missing: "incomplete", never filled in.

Number readout (frozen MLM head, 8 slots, no options; scripts/exp44/aux_readout.py) of J on the four sets and, lower
priority, of M on fit_O / rl_tune: whole-number exact per field, both objects, twin, same prediction (the same
statistics), and picture dependence: on the twins whose number changes between the views and whose number inputs are
identical apart from the picture, the other view's prediction scored against this question's gold (the wrong picture)
-- real - wrong per field with case-cluster intervals. A positive picture dependence is not proof of reading the print
(particle sizes and gaps change too).

Conditions (pre-written; investment, not release): progress -- rl_tune different macro >= 0.60, both >= 0.20 with
J - M lower bound > 0, twin >= 0.20 with J - M lower bound > 0; number-loss attribution -- rl_tune both J - C >= 0.10
with lower bound > 0. If the number readout is right but the decision is not: "the shared encoder reads the numbers
after training, but reading to option scoring is still not connected" -- not "vision fixed".

    uv run python scripts/exp44/report.py
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

from transformers import AutoTokenizer

from devision.train.rlcd import aux_example, aux_number

HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r42 = _load("exp42_report", "../exp42/report.py")
r41, b40, frozen = r42.r41, r42.b40, r42.frozen
SETS, RISK, INIT, INIT_SHA = r42.SETS, r42.RISK, r42.INIT, r42.INIT_SHA
RUNS = {"M": "runs/exp42-start", "C": "runs/exp42-c", "J": "runs/exp44-j"}
C_SHA = {"model.safetensors": "6596907825f0b9c1ba9c0e9eb8dc136a91d6c2a48b331bc50ba34ea60ad36ca4",
         "config.json": "e5421f88ff74a665d4730725e56bb49246eed419002ca064d7f308baab0f89f9"}
M_AUX = "runs/exp44-m-aux"
AUX_SETS = {"aux_" + n: p for n, p in SETS.items()}
NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?$")       # the parse rule of scripts/exp44/aux_readout.py


def j_identity(run: str, check: Dict, manifest: Dict) -> str:
    try:
        s = json.load(open(os.path.join(run, "START.json")))
        st = json.load(open(os.path.join(run, "run.json")))["config"]["stages"][0]
        tr = json.load(open(os.path.join(run, "stage2", "train_report.json")))
    except (OSError, ValueError, KeyError) as e:
        return "incomplete: %s" % e
    par = st.get("params") or {}
    if s.get("init") != INIT or s.get("init_sha256") != INIT_SHA or s.get("data_sha256") != manifest["files"]["train"]:
        return "wrong run: START %s" % s
    if st.get("data") != "data/exp42/train.jsonl" or st.get("init") != INIT or par.get("objective") != "ce" \
            or par.get("aux_weight") != 1.0 or par.get("aux_slots") != 8:
        return "wrong run: run.json %s" % par
    aux = tr.get("aux") or {}
    head = {k: v for k, v in (aux.get("head") or {}).items() if k != "repo"}
    want = {k: v for k, v in check["head"].items() if k != "file_sha256"}
    if tr.get("objective") != "ce" or aux.get("weight") != 1.0 or aux.get("slots") != 8 or head != want:
        return "wrong run: train_report %s %s" % (tr.get("objective"), aux)
    return ""


def aux_records(run: str, name: str, data_path: str, checkpoint: str, head: Dict) -> List[Dict]:
    info = json.load(open(os.path.join(run, "eval", name + ".json")))["run"]
    h = {k: v for k, v in info.get("head", {}).items() if k != "repo"}
    if os.path.normpath(info.get("checkpoint", "")) != os.path.normpath(checkpoint) or info.get("data_sha256") != r41.sha(data_path) \
            or info.get("slots") != 8 or h != head:
        raise SystemExit("%s/%s: not %s's number readout on %s with the frozen head" % (run, name, checkpoint, data_path))
    got = {}
    for g in r42.load(os.path.join(run, "eval", name + ".details.jsonl")):
        k = (g["sample_id"], g["qid"])
        if k in got:
            raise SystemExit("%s/%s: repeated %s" % (run, name, k))
        got[k] = g
    rows = r42.load(data_path)
    tok = AutoTokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))     # tokenizer only, no model
    out = []
    for r in rows:
        view = r.get("view") or r["id"].split(":")[2]
        for q, v in r["questions"].items():
            f = b40.field_of(v["instructions"])
            if f == "cmp":
                continue
            g = got.pop((r["id"], q), None)
            if g is None:
                raise SystemExit("%s/%s: missing %s %s" % (run, name, r["id"], q))
            gold = next(o for o, p in r["gold"][q]["probabilities"].items() if p == 1.0)
            ids, _ = aux_example(tok, r.get("state_text", ""), v["instructions"], gold, 8)      # the training input rule
            valid = g["prediction"] is not None and bool(NUMBER.match(g["prediction"]))       # the readout's parse rule
            if g["gold_number"] != aux_number(gold) or g["image_id"] != r["image_id"] or g["image_used"] != r["image"] \
                    or g["aux_input_sha256"] != hashlib.sha256(json.dumps(ids).encode()).hexdigest() \
                    or bool(g["valid"]) != valid or bool(g["correct"]) != (valid and g["prediction"] == g["gold_number"]):
                raise SystemExit("%s/%s: %s %s does not match its source question (gold, picture, input or validity)"
                                 % (run, name, r["id"], q))
            out.append({"family": r["family"], "case": r["case"], "field": f, "object": b40.object_of(v["instructions"]),
                        "view": view, "picture": r.get("orig_id", r["id"]), "gold": g["gold_number"],
                        "prediction": g["prediction"], "right": bool(g["correct"]), "valid": valid,
                        "aux_input": g["aux_input_sha256"]})
    if got:
        raise SystemExit("%s/%s: %d records not in the data" % (run, name, len(got)))
    return out


def dependence(recs: List[Dict]) -> Dict:
    """family -> 'dep:<field>' [sum of real - wrong, n] on value-changing twins with identical number inputs."""
    fam: Dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    twins = defaultdict(list)
    for r in recs:
        twins[(r["family"], r["field"], r["object"])].append(r)
    skipped = 0
    for (family, field, _), v in twins.items():
        if len(v) != 2 or v[0]["view"] == v[1]["view"] or v[0]["gold"] == v[1]["gold"]:
            continue
        if v[0]["aux_input"] != v[1]["aux_input"]:
            skipped += 1
            continue
        for x, y in ((v[0], v[1]), (v[1], v[0])):
            u = fam[family]["dep:" + field]
            u[0] += float(x["right"]) - float(y["prediction"] == x["gold"])
            u[1] += 1
    fam["_skipped"]["n"] = [skipped, 1]
    return fam


def build(runs: Dict[str, str] = RUNS, m_aux: str = M_AUX) -> Dict:
    manifest = json.load(open("data/exp42/MANIFEST.json"))
    check = json.load(open("data/exp44/CHECK.json"))
    head = {k: v for k, v in check["head"].items() if k != "file_sha256"}
    if frozen.check("runs/exp40-m", ["rl_tune"])["weights"] != INIT_SHA or frozen.check(runs["C"], list(SETS))["weights"] != C_SHA:
        raise SystemExit("exp40-M or exp42-C is not its frozen identity")
    out: Dict = {"identity": {"M": INIT_SHA, "C": C_SHA}, "runs": {}, "readout": {}, "differences": {}, "aux": {}, "risk": {}}
    ckpt = {"M": INIT, "C": os.path.join(runs["C"], "stage2")}
    j = runs["J"]
    why = j_identity(j, check, manifest) if os.path.exists(j) else "incomplete: not started"
    if why.startswith("wrong"):
        raise SystemExit("J: " + why)
    st = r42.states(j + "-status.txt")
    if why or st.get("train") != "completed" or not os.path.exists(frozen.path(j)):
        out["runs"]["J"] = why or "incomplete: training"
    else:
        out["identity"]["J"] = json.load(open(frozen.path(j)))["weights"]
        ckpt["J"] = os.path.join(j, "stage2")
        done = [n for n in list(SETS) + list(RISK) + list(AUX_SETS) if st.get("eval:" + n) == "completed"]
        frozen.check(j, done)
        out["runs"]["J"] = {"completed": done, "missing": [n for n in list(SETS) + list(RISK) + list(AUX_SETS) if n not in done]}
    recs: Dict = {}
    for who in ("M", "C", "J"):
        if who not in ckpt:
            continue
        run = runs[who]
        done = out["runs"]["J"]["completed"] if who == "J" else list(SETS) + list(RISK)
        for n in done:
            if n in SETS:
                recs[(who, n)] = r42.eval_records(run, n, SETS[n], ckpt[who])
            elif n in RISK:
                rows = r42.load(RISK[n])
                got = r41.check_details("%s/%s" % (run, n), rows, r42.load(os.path.join(run, "eval", n + ".details.jsonl")))
                out["risk"].setdefault(n, {})[who] = sum(g["correct"] for g in got.values()) / len(got)
            elif n in AUX_SETS:
                a = aux_records(run, n, AUX_SETS[n], ckpt[who], head)
                out["aux"].setdefault(n, {})["J"] = aux_summary(a)
    for n in ("aux_fit_O", "aux_rl_tune"):
        mst = r42.states(m_aux + "-status.txt")
        if mst.get("eval:" + n) == "completed":
            out["aux"].setdefault(n, {})["M"] = aux_summary(aux_records(m_aux, n, AUX_SETS[n], INIT, head))
        else:
            out["aux"].setdefault(n, {})["M"] = "missing"
    fams = {k: r42.units(v) for k, v in recs.items()}
    for (who, n), fam in fams.items():
        out["readout"].setdefault(n, {})[who] = r42.summary(fam)
    for n in SETS:
        for a, b in (("J", "M"), ("J", "C")):
            if (a, n) in fams and (b, n) in fams:
                ca, cb = r42.clusters(recs[(a, n)]), r42.clusters(recs[(b, n)])
                if ca != cb:
                    raise SystemExit("%s: %s and %s on different questions" % (n, a, b))
                out["differences"]["%s|%s-%s" % (n, a, b)] = {s: r42.diff(fams[(a, n)], fams[(b, n)], ca, s) for s in ("different", "both", "twin")}
    out["conditions"] = conditions(out)
    return out


def conditions(out: Dict) -> Dict:
    """The pre-written conditions; both "incomplete" unless all four main decision sets of J are read (critic
    EXP44-R1-03) -- the per-set values and intervals of what was read stay in the report."""
    if not all("J" in out["readout"].get(n, {}) for n in SETS) \
            or not all("%s|J-%s" % (n, b) in out["differences"] for n in SETS for b in ("M", "C")):
        return {"progress": "incomplete", "number_loss_attribution": "incomplete"}
    rd, d = out["readout"]["rl_tune"], out["differences"]
    k = "rl_tune|J-M"
    return {"progress": bool(rd["J"]["different"] >= 0.60 and rd["J"]["both"] >= 0.20 and d[k]["both"]["ci95"][0] > 0
                             and rd["J"]["twin"] >= 0.20 and d[k]["twin"]["ci95"][0] > 0),
            "number_loss_attribution": bool(d["rl_tune|J-C"]["both"]["value"] >= 0.10 and d["rl_tune|J-C"]["both"]["ci95"][0] > 0)}


def aux_summary(recs: List[Dict]) -> Dict:
    fam = r42.units(recs)
    out = r42.summary(fam)
    out["valid"] = {"valid": sum(r["valid"] for r in recs), "of": len(recs), "rate": sum(r["valid"] for r in recs) / len(recs)}
    out["whole"] = {f: {"right": sum(r["right"] for r in recs if r["field"] == f), "of": sum(r["field"] == f for r in recs)}
                    for f in r42.FIELDS if any(r["field"] == f for r in recs)}
    dep = dependence(recs)
    skipped = dep.pop("_skipped")["n"][0]
    cl = r42.clusters(recs)
    keys = sorted(cl)
    expand = lambda ks: [f for k in ks for f in sorted(cl[k])]  # noqa: E731
    out["picture_dependence"] = {f: dict(r41.interval(lambda fs, f=f: r42.value(dep, fs, "dep:" + f), keys, expand),
                                         n=sum(dep[x]["dep:" + f][1] for x in dep if "dep:" + f in dep[x]))
                                 for f in r42.FIELDS if any("dep:" + f in dep[x] for x in dep)}
    out["picture_dependence_skipped_twins"] = skipped
    return out


def main() -> int:
    out = build()
    json.dump(out, open("runs/exp44-report.json", "w"), indent=1, default=str)
    print("identity %s\nruns %s" % (out["identity"], out["runs"]))
    for n, by in out["readout"].items():
        for who, s in by.items():
            print("%-9s %s different %.3f both %.3f twin %.3f same_answer %.3f | %s" % (
                n, who, s["different"], s["both"], s["twin"], s["same_answer"],
                {k: "%d/%d" % (v["right"], v["of"]) for k, v in s.items() if isinstance(v, dict) and not k.startswith("same")}))
    for k, v in out["differences"].items():
        print("%s %s" % (k, {s: "%+.3f [%+.3f, %+.3f]" % (x["value"], *x["ci95"]) for s, x in v.items()}))
    for n, by in out["aux"].items():
        for who, s in by.items():
            if isinstance(s, str):
                print("%s %s: %s" % (n, who, s))
            else:
                print("%s %s: whole %s both %.3f twin %.3f valid %d/%d dependence %s" % (
                    n, who, {f: "%d/%d" % (v["right"], v["of"]) for f, v in s["whole"].items()}, s["both"], s["twin"],
                    s["valid"]["valid"], s["valid"]["of"],
                    {f: "%+.3f [%+.3f, %+.3f] n%d" % (v["value"], *v["ci95"], v["n"]) for f, v in s["picture_dependence"].items()}))
    print("risk %s\nconditions %s" % (out["risk"], out["conditions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
