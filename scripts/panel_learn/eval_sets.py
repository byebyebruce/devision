"""PANEL-LEARN-PREP-01: D2's monitor sets in the mixed_input form ("input": "picture", "group": the family), the same
rows, questions, gold and pictures as data/exp22/fit.jsonl (training families: the learnability gate),
data/exp22/dev16.jsonl (unseen families, the original rule; a development monitor, not a confirmation) and
data/exp21/dev_syn_erased.jsonl (the erased-digit control). Writes data/panel_learn/{fit,dev16,erased}.jsonl and adds
their sha256 to eval_sets.MANIFEST.json. CPU only."""
import hashlib
import json
import os
import sys

SETS = {"fit": "exp22/fit.jsonl", "dev16": "exp22/dev16.jsonl", "erased": "exp21/dev_syn_erased.jsonl"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(root="data"):
    out, man = os.path.join(root, "panel_learn"), {}
    for name, rel in SETS.items():
        rows = [json.loads(x) for x in open(os.path.join(root, rel)) if x.strip()]
        path = os.path.join(out, name + ".jsonl")
        if os.path.exists(path):
            raise SystemExit("%s exists" % path)
        open(path, "w").write("".join(json.dumps(dict(r, input="picture", group=r["id"].split(":")[1])) + "\n" for r in rows))
        man[name] = {"source": rel, "source_sha256": sha(os.path.join(root, rel)), "rows": len(rows),
                     "questions": sum(len(r["questions"]) for r in rows), "sha256": sha(path)}
    json.dump(man, open(os.path.join(out, "eval_sets.MANIFEST.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
