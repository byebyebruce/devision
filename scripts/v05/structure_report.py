"""Close exp79 using its prewritten panels and saved predictions; never load a model."""
import json
from collections import defaultdict
from pathlib import Path

from devision.train.compare import load_records, paired_metric


def occurrences(rows):
    # Some historical dev files contain repeated question ids; retain each
    # occurrence and its picture rather than silently dropping duplicates.
    seen = defaultdict(int)
    out = []
    for r in rows:
        key = (r["sample_id"], r["qid"])
        out.append(dict(r, sample_id=f'{r["sample_id"]}#occ{seen[key]}'))
        seen[key] += 1
    return out


def main():
    natural = {r["id"] for r in map(json.loads, Path("data/n2/dev_sqa.jsonl").open())
               if r.get("subject") == "natural science"}
    panels = {}
    combined = [[], []]
    for name in ("dev_count", "dev_tallyqa", "dev_sqa", "dev_v7w", "dev_mix", "dev_spatial", "dev_coco_exist"):
        a = load_records(f"runs/exp78-res256/eval/{name}.details.jsonl")
        b = load_records(f"runs/exp79-tok256/eval/{name}.details.jsonl")
        if name == "dev_sqa":
            a = [r for r in a if r["sample_id"] in natural]
            b = [r for r in b if r["sample_id"] in natural]
        a, b = occurrences(a), occurrences(b)
        panels[name] = paired_metric(a, b, seed=79)
        if name in ("dev_count", "dev_tallyqa", "dev_sqa", "dev_v7w"):
            combined[0].extend(a)
            combined[1].extend(b)
    p1 = paired_metric(*combined, seed=79)
    harmful = [name for name in ("dev_mix", "dev_spatial", "dev_coco_exist")
               if panels[name]["difference"] <= -.02 and panels[name]["ci95"][1] < 0]
    report = {"baseline": "runs/exp78-res256", "candidate": "runs/exp79-tok256", "panels": panels,
              "P1": p1, "P1_pass": p1["ci95"][0] > 0, "P2_harmful": harmful, "P2_pass": not harmful}
    report["decision"] = "consider_full_run" if report["P1_pass"] and report["P2_pass"] else "do_not_expand_this_setting"
    out = Path("runs/exp79/comparison.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
