"""r24 training mix (docs/experiments/2026-10-29-r24-v05-plan.md): r23's training file + the board-gap families that
passed criterion 1 in exp137 / exp138 (scripts/v05/judge_board.py) + A-OKVQA replay (exp137 lost 1.8 points on
bench_lv_aokvqa against r23; r23's file has only 451 A-OKVQA rows), and a temperature-fit set that covers yes/no
questions (r23 / exp137 fitted on 512 chart questions with 11 yes/no ones).

Every training row (base, replay and families) is filtered by the Decision Index board exclusion index (sha256 /
pixels / near-duplicate hash of the picture, and board source ids); the dropped counts go to the MANIFEST.

  train.jsonl   base (filtered) + --families at the exp137 / exp138 amounts + --aokvqa replay rows
  calib.jsonl   exp81 chart calib (512) + v2 dev yes/no, exist, VQA choice and GQA slices; pictures that are in the
                training mix or in a test_* / bench_* set of --config are dropped
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp138_mix import board_filter, by_image_sample, load, sha  # noqa: E402
from v05_board_common import BoardIndex  # noqa: E402

# family -> (rows, at most per picture); None = all rows. The amounts exp137 / exp138 tested.
AMOUNTS = {"caption": (20000, 1), "kie": (None, 0), "infographic": (None, 0), "figure": (12000, 2),
           "depth": (10000, 2), "som": (None, 0), "drive": (6000, 2), "hateful": (None, 0)}
CALIB = [("v05/exp81/calib.jsonl", None), ("v2/dev_vqa_yesno.jsonl", 300), ("v2/dev_coco_exist.jsonl", 150),
         ("v2/dev_vqa_choice.jsonl", 200), ("v2/dev_gqa.jsonl", 200)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--base", default="v05/exp132/train.jsonl")
    ap.add_argument("--board", default="v05/board")
    ap.add_argument("--exclusion", default="v05/board_exclusion.json")
    ap.add_argument("--families", required=True, help="comma-separated families that passed criterion 1")
    ap.add_argument("--aokvqa", type=int, default=4000)
    ap.add_argument("--aokvqa-pool", default="v4/aokvqa.jsonl")
    ap.add_argument("--out", default="v05/r24")
    ap.add_argument("--config", default="configs/r24-v05.yaml", help="its test_* / bench_* pictures stay out of calib")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=24)
    a = ap.parse_args()
    root, rng = Path(a.root), random.Random(a.seed)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    index = BoardIndex.load(str(root / a.exclusion))
    fams = [f for f in a.families.split(",") if f]
    unknown = set(fams) - set(AMOUNTS)
    if unknown:
        raise SystemExit("unknown families: %s" % sorted(unknown))

    base_all = load(root / a.base)
    ids = {r["id"] for r in base_all}
    base_images = {r["image"] for r in base_all}
    pool = [r for r in load(root / a.aokvqa_pool) if r["id"] not in ids and r["image"] not in base_images]
    replay = by_image_sample(pool, a.aokvqa, 1, rng)
    add = {"aokvqa_replay": replay}
    for fam in fams:
        rows = load(root / a.board / fam / "train.jsonl")
        n, per = AMOUNTS[fam]
        add[fam] = rows if n is None else by_image_sample(rows, n, per, rng)

    parts = {"base": base_all, **add}
    kept, filtered = {}, {}
    for name, rows in parts.items():
        kept[name], filtered[name] = board_filter(rows, root, index, a.workers)
        print(name, "board filter:", filtered[name], flush=True)

    rows: list[dict] = []
    seen: set[str] = set()
    for name, part in kept.items():
        dup = [r["id"] for r in part if r["id"] in seen]
        if dup:
            raise SystemExit("%s: %d duplicate ids" % (name, len(dup)))
        seen.update(r["id"] for r in part)
        rows.extend(part)
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tests = []
    for name, spec in yaml.safe_load(open(a.config))["evaluate"]["sets"].items():
        if name.startswith(("test_", "bench_")):
            path = spec["path"] if isinstance(spec, dict) else spec
            tests.append(str(Path(path).relative_to(a.root)))
    cfg = yaml.safe_load(open(a.config))["evaluate"]
    aliases = json.load(open(cfg["image_identity"])) if cfg.get("image_identity") else {}
    pic = lambda r: aliases.get(r.get("image_id", ""), r.get("image_id", ""))  # noqa: E731  same identity as pipeline
    test_rows = [r for t in tests for r in load(root / t)]
    avoid = {r["image"] for r in rows} | {r["image"] for r in test_rows}
    avoid_pics = {pic(r) for r in test_rows} - {""}
    calib, calib_parts = [], {}
    for path, n in CALIB:
        src = [r for r in load(root / path) if r["image"] not in avoid and pic(r) not in avoid_pics]
        take = src if n is None else rng.sample(src, min(n, len(src)))
        calib_parts[path] = len(take)
        calib.extend(take)
    qtypes = Counter(q["type"] for r in calib for q in r["questions"].values())
    if qtypes["noul"] < 300:
        raise SystemExit("calib has only %d yes/no questions" % qtypes["noul"])
    with open(out / "calib.jsonl", "w") as f:
        for r in calib:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "base": {"path": a.base, "rows": len(base_all), "sha256": sha(root / a.base)},
        "board_filtered": filtered,
        "exclusion_sha256": sha(root / a.exclusion),
        "families": fams,
        "board_manifests": {f: sha(root / a.board / f / "MANIFEST.json") for f in fams},
        "added": {k: {"rows": len(kept[k]), "images": len({r["image"] for r in kept[k]})} for k in add},
        "aokvqa_pool": {"path": a.aokvqa_pool, "available": len(pool)},
        "rows": len(rows), "seed": a.seed,
        "sources": Counter(r["source"] for r in rows),
        "train_sha256": sha(out / "train.jsonl"),
        "calib": {"parts": calib_parts, "avoided_test_sets": len(tests), "types": qtypes, "sha256": sha(out / "calib.jsonl")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("board_filtered", "added", "rows", "calib")}, indent=1))


if __name__ == "__main__":
    main()
