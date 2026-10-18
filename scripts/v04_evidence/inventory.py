"""v0.4 reader-evidence inventory (critic V04-READER-EVIDENCE-PREP-02; docs/experiments/2026-10-19-v04-reader-evidence-inventory.md).
CPU only, no model.

  A  the 29 sets of configs/v04-full-eval.yaml and every extra of configs/v04-extra-evidence.yaml: rows, questions, how
     many questions the product router (devision.reading.domain_of) sends to the new reading path (temperature /
     magnets) and how many stay ordinary, the controls each gets (29 sets: mismatched + reversed for choice questions;
     extras: as listed), requests (devision-eval grouping, max 8 questions) and a rough time (ordinary requests at
     T's measured P50 0.334 s, routed ones at EXP71's measured 0.43 s per request).
  B  real temperature / magnet pictures: every question of the raw ScienceQA parquet (train / validation / test) the
     router sends to a reading domain, its picture's sha256, and whether that picture (by bytes) appears in any data file
     already used -- training files, dev / monitoring files (dev_sqa, exp29, exp28, panel_science, dev_mix), benchmarks
     (lv_bench scienceqa) or JevBench cases. Pictures used nowhere are the only possible fresh confirmation source.
  C  the combination identity of EXP71-on (what a release IDENTITY would have to bind beyond model.safetensors).

    uv run python scripts/v04_evidence/inventory.py [--out data/v04_evidence/inventory.json]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

ORDINARY_S, ROUTED_S = 0.334, 0.43
RAW = "data/raw/hf/datasets--derek-thomas--ScienceQA/snapshots/f18b0a70359ebfb41f658fd564208d0355b013f4/data"


def jsonl(p: str) -> List[Dict]:
    return [json.loads(x) for x in open(p) if x.strip()]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: str) -> str:
    return sha_bytes(open(p, "rb").read())


def set_profile(path: str, controls: List[str]) -> Dict:
    from devision.reading import domain_of
    from devision.train.evaluate import _requests
    rows = jsonl(path)
    qs = [q for r in rows for q in r["questions"].values()]
    routed = Counter(domain_of(q.get("instructions", "")) or "ordinary" for q in qs)
    req = len(_requests(rows, {}, "none", 8))
    n_routed_req = len({(r["image"], r.get("state_text", "")) for r in rows
                        if any(domain_of(q.get("instructions", "")) for q in r["questions"].values())})
    runs = 1 + len(controls)
    secs = runs * ((req - n_routed_req) * ORDINARY_S + n_routed_req * ROUTED_S)
    return {"rows": len(rows), "questions": len(qs), "routed": dict(routed), "requests": req, "routed_requests": n_routed_req,
            "choice_questions": sum(q["type"] == "choice" for q in qs), "controls": controls, "est_minutes": round(secs / 60, 1)}


def part_a() -> Dict:
    import yaml
    cfg = yaml.safe_load(open("configs/v04-full-eval.yaml"))["evaluate"]
    out: Dict = {"sets": {}, "extras": {}}
    for name, v in cfg["sets"].items():
        path = v["path"] if isinstance(v, dict) else v
        prof = set_profile(path, [])
        prof["controls"] = ["mismatched"] + (["reversed"] if prof["choice_questions"] else [])
        prof["est_minutes"] = round(prof["est_minutes"] * (1 + len(prof["controls"])), 1)
        prof["role"] = v.get("role") if isinstance(v, dict) else "by usage"
        out["sets"][name] = prof
    for e in yaml.safe_load(open("configs/v04-extra-evidence.yaml")):
        prof = set_profile(e["data"], [] if e["control"] == "none" else [e["control"]])
        prof.update(temperature=e["temperature"], role=e["role"], control=e["control"])
        out["extras"][e["name"]] = prof
    tot = lambda d, k: sum(x[k] for x in d.values())  # noqa: E731
    out["totals"] = {"sets_questions": tot(out["sets"], "questions"), "extras_questions": tot(out["extras"], "questions"),
                     "routed_questions_sets": sum(sum(v for k, v in x["routed"].items() if k != "ordinary") for x in out["sets"].values()),
                     "routed_questions_extras": sum(sum(v for k, v in x["routed"].items() if k != "ordinary") for x in out["extras"].values()),
                     "est_minutes": round(tot(out["sets"], "est_minutes") + tot(out["extras"], "est_minutes"), 1),
                     "note": "rough: request counts x measured per-request times; not a measured full evaluation"}
    out["confirmation_routing"] = {n: out["extras"][n]["routed"] for n in out["extras"] if n.startswith("confirm_")}
    return out


SPLIT_OF = {"train": "train", "test": "test", "val": "validation"}


def exposures() -> Dict[str, Dict]:
    """(ScienceQA split, row index) -> the uses of that question (data files holding it as a reading-domain row): the
    data files keep re-encoded pictures, so they are matched by their ScienceQA ids ("v3-scienceqa:train-N",
    "v3-sqa-test:test-N", "lv-scienceqa:val-N"), each checked against the raw question text in part_b."""
    from devision.reading import domain_of
    use: Dict = defaultdict(lambda: {"kinds": set(), "files": set(), "text": set()})
    for p in sorted(set(glob.glob("data/**/*.jsonl", recursive=True))):
        if os.path.getsize(p) > 400_000_000:
            continue
        base = os.path.basename(p)
        kind = ("benchmark" if "lv_bench" in p else "jevbench" if "jevbench" in p else
                "training" if ("train" in base or "/replay/" in p or base in ("scienceqa.jsonl", "science.jsonl")) else "dev / monitoring")
        try:
            rows = jsonl(p)
        except (ValueError, UnicodeDecodeError):
            continue
        for r in rows:
            qs = r.get("questions") if isinstance(r, dict) else None
            if not isinstance(qs, dict):
                continue
            texts = [q.get("instructions", "") for q in qs.values() if isinstance(q, dict) and domain_of(q.get("instructions", "") or "")]
            rid = r.get("id") if isinstance(r.get("id"), str) else ""
            tail = rid.split(":")[1] if rid.count(":") >= 1 else ""
            sp, _, idx = tail.rpartition("-")
            if not texts or sp not in SPLIT_OF or not idx.isdigit():
                continue
            u = use[(SPLIT_OF[sp], int(idx))]
            u["kinds"].add(kind)
            u["files"].add(p)
            u["text"].update(texts)
    return use


def part_b() -> Dict:
    import pyarrow.parquet as pq

    from devision.reading import domain_of
    use = exposures()
    out: Dict = {"splits": {}, "matching": "by ScienceQA split + row index from the data ids, question text checked"}
    for split in ("train", "validation", "test"):
        f = glob.glob(os.path.join(RAW, split + "-*.parquet"))[0]
        t = pq.read_table(f, columns=["question", "image"]).to_pylist()
        dom: Counter = Counter()
        fresh: Counter = Counter()
        by_kind: Counter = Counter()
        text_mismatch = 0
        for i, r in enumerate(t):
            d = domain_of(r["question"] or "")
            if not d or not r["image"]:
                continue
            dom[d] += 1
            u = use.get((split, i))
            if u is None:
                fresh[d] += 1
                continue
            text_mismatch += r["question"] not in u["text"]
            for k in sorted(u["kinds"]):
                by_kind["%s:%s" % (d, k)] += 1
        out["splits"][split] = {"routed_questions_with_picture": dict(dom), "used_nowhere": dict(fresh), "questions_by_use": dict(by_kind),
                                "id_text_mismatches": text_mismatch}
    files = defaultdict(set)
    for (sp, _), u in use.items():
        for p in u["files"]:
            files[sp].add(p)
    out["files_by_split"] = {k: sorted(v) for k, v in files.items()}
    return out


def part_c() -> Dict:
    """What EXP71-on actually is, beyond model.safetensors."""
    from devision.reading import CORRECTOR, GRID, HEADER, POLICY, TEMPLATES
    b = "runs/exp71/bundle_on"
    rj = json.load(open(os.path.join(b, "reading.json")))
    return {"bundle_reading_json": sha(os.path.join(b, "reading.json")), "components": {k: rj["sha256"][k] for k in ("selector.pt", "head.pt", "corrector.pt")},
            "decision": {k: rj["sha256"][k] for k in ("model.safetensors", "config.json", "tokenizer")}, "policy": POLICY, "grid": GRID,
            "corrector_rule": CORRECTOR, "table_header": HEADER, "templates_sha256": sha_bytes(json.dumps(TEMPLATES, sort_keys=True).encode()),
            "code": {f: sha(f) for f in ("src/devision/reading.py", "src/devision/image.py", "src/devision/infer.py")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/v04_evidence/inventory.json")
    a = ap.parse_args(argv)
    res = {"A": part_a(), "B": part_b(), "C": part_c()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"A_totals": res["A"]["totals"], "A_confirmation_routing": res["A"]["confirmation_routing"], "B": res["B"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
