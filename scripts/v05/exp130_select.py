"""exp130 (docs/experiments/2026-10-28-exp130-v4-interp.md): pick the interpolation alpha from dev results only.

    uv run python scripts/v05/exp130_select.py runs/exp130-interp

Reads <root>/<model>/devsel/<set>.json for model in v04, a1.0 (= exp129) and a<alpha>; writes <root>/select.json.
Rule (written before any result): new-use dev sets keep >= 85% of exp129's gain over v0.4 each, and the yes/no
pair (dev_coco_exist + dev_vqa_yesno accuracy) beats exp129 by >= 0.5 pp in total; take the smallest such alpha.
"""
import json
import os
import sys

NEW = ["gui_dev_state", "gui_shop_dev_state", "gui_dev", "ego_dev"]
YESNO = ["dev_coco_exist", "dev_vqa_yesno"]
ALPHAS = ["0.9", "0.75", "0.6"]


def acc(root, model, s):
    return json.load(open(os.path.join(root, model, "devsel", s + ".json")))["accuracy_all"]


def main():
    root = sys.argv[1]
    v4 = {s: acc(root, "v04", s) for s in NEW + YESNO}
    top = {s: acc(root, "a1.0", s) for s in NEW + YESNO}
    rows, chosen = [], None
    for a in sorted(ALPHAS, key=float):
        m = {s: acc(root, "a" + a, s) for s in NEW + YESNO}
        keep = {s: (m[s] - v4[s]) / (top[s] - v4[s]) if top[s] > v4[s] else None for s in NEW}
        ok_new = all(k is not None and k >= 0.85 for k in keep.values())
        yn = sum(m[s] - top[s] for s in YESNO)
        ok = ok_new and yn >= 0.005
        rows.append({"alpha": float(a), "acc": m, "gain_kept": keep, "yesno_vs_exp129": yn, "eligible": ok})
        if ok and chosen is None:
            chosen = a
    out = {"v04": v4, "exp129": top, "alphas": rows, "chosen": chosen}
    json.dump(out, open(os.path.join(root, "select.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("CHOSEN", chosen or "none")


if __name__ == "__main__":
    main()
