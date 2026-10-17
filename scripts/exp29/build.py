"""exp29 (docs/experiments/2026-10-17-exp29-perm-avg-plan.md): every choice question of the development panels
asked in all k cyclic orders of its options (k = number of options), same picture, text and question; noul
questions are left out. Order 0 is the original order, i.e. the single-order baseline.

    uv run python scripts/exp29/build.py --root data

Sample ids get ":o<j>" (j = 0..k-1); the gold goes with its option. Outputs <root>/exp29/<panel>.jsonl, MANIFEST.json.
"""
import argparse
import hashlib
import json
import os

PANELS = {"science": "r13b/panel_science.jsonl", "ext": "ext/dev_ext.jsonl", "count_wide": "r13b/count_wide.jsonl",
          "count_narrow": "v11/dev_count_transfer.jsonl", "flip": "v6/dev_lr_pairs.jsonl",
          "sqa_diagram": "v10/dev_sqa_diagram.jsonl"}


def cyclic(sample: dict) -> list:
    out = []
    for qid, q in sample["questions"].items():
        if q["type"] != "choice":
            continue
        keys, gold = list(q["criteria"]), sample["gold"][qid]["probabilities"]
        for j in range(len(keys)):
            order = keys[j:] + keys[:j]
            out.append(dict(sample, id="%s|%s:o%d" % (sample["id"], qid, j),
                            questions={qid: dict(q, criteria={k: q["criteria"][k] for k in order})},
                            gold={qid: {"probabilities": {k: gold.get(k, 0.0) for k in order}}}))
    return out


def build(root: str) -> dict:
    out = os.path.join(root, "exp29")
    os.makedirs(out, exist_ok=True)
    manifest = {}
    for name, rel in PANELS.items():
        rows = [r for line in open(os.path.join(root, rel)) for r in cyclic(json.loads(line))]
        path = os.path.join(out, name + ".jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        manifest[name] = {"source": rel, "rows": len(rows), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data")
    print(json.dumps(build(p.parse_args().root), indent=1))
