"""PANEL-LEARN-PREP-01: the real transfer check of a new decision model without the reading step. The reading
components and rules stay frozen, so the states they produced for the sealed v04-k evaluation (each record's read_state,
the exact text decide scored with image=None) are what the same reading step would give a new decision model. Writes
"input": "text" rows (the data file's questions and gold, the sealed state) for sqa_temp / sqa_force and their
mismatched controls to data/panel_learn/read_states_<set>.jsonl, with the sealed details' sha256 in
read_states.MANIFEST.json. Refuses unless every record was read and the details hash to the sealed RESULTS.json."""
import hashlib
import json
import os
import sys

EVAL = "runs/v04-k/eval"
SETS = {"sqa_temp": "exp21/dev_sqa_temp.jsonl", "sqa_force": "exp21/dev_sqa_force.jsonl",
        "sqa_temp_mism": "exp21/dev_sqa_temp_mism.jsonl", "sqa_force_mism": "exp21/dev_sqa_force_mism.jsonl"}


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(root: str = "data") -> int:
    sealed = json.load(open(os.path.join(EVAL, "RESULTS.json")))
    out_dir = os.path.join(root, "panel_learn")
    man = {}
    for name, data in SETS.items():
        det = os.path.join(EVAL, "extra_%s.details.jsonl" % name)
        if sealed.get("extra_%s.json" % name, {}).get("extra_%s.details.jsonl" % name) != sha(det):
            raise SystemExit("%s is not the sealed details" % det)
        src = {r["id"]: r for r in map(json.loads, open(os.path.join(root, data)))}
        recs = [json.loads(x) for x in open(det)]
        if any(r.get("route") != "read" or not r.get("read_state") for r in recs):
            raise SystemExit("%s: not every record was read" % name)
        rows = []
        for r in recs:
            s = src[r["sample_id"]]
            rows.append({"id": r["sample_id"], "source": s.get("source", name), "input": "text", "image": "text-only:v04-k-read-state",
                         "image_id": s["image_id"], "group": r["sample_id"], "state_text": r["read_state"],
                         "questions": {r["qid"]: s["questions"][r["qid"]]}, "gold": {r["qid"]: s["gold"][r["qid"]]}})
        path = os.path.join(out_dir, "read_states_%s.jsonl" % name)
        if os.path.exists(path):
            raise SystemExit("%s exists" % path)
        open(path, "w").write("".join(json.dumps(x) + "\n" for x in rows))
        man[name] = {"rows": len(rows), "sealed_details_sha256": sha(det), "data_sha256": sha(os.path.join(root, data)),
                     "sealed_accuracy": sum(r["correct"] for r in recs) / len(recs), "file_sha256": sha(path)}
    json.dump(man, open(os.path.join(out_dir, "read_states.MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
