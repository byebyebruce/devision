"""exp119 training mix: exp117 (science replay) with the GUI rows reworked after exp118.

exp118 showed the screen-state rewrite makes AITW (phone) questions need the screen (test_gui_state AITW real
56.6 vs mismatched 34.8) but not Mind2Web (35.6 vs 35.3): desktop pages are 1024x800, at 256 letterbox the text
is ~3px. So here:
- AITW-general rows of exp117 -> screen-state variant (same rows as exp118, same conversion);
- + --shopping AITW web_shopping rows (new, from data/v05/gui/train.jsonl), screen-state variant;
- Mind2Web rows stay in the original format (history kept) — the state variant gives no visual signal at 256,
  and the original format kept test_gui Mind2Web at 78 (exp118 dropped it to 66).
Everything else is exp117's mix. Also writes the AITW-shopping test / dev (screen-state, episode pool from the
same file) as new eval sets.
"""
import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("exp118", Path(__file__).with_name("exp118_gui_state.py"))
assert spec and spec.loader
e118 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e118)


def write(p: Path, rows: list[dict]) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=119)
    ap.add_argument("--shopping", type=int, default=12000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp119"
    out.mkdir(parents=True, exist_ok=True)

    gui_train = e118.load(root / "v05/gui/train.jsonl")
    pool = e118.golds_by_episode(gui_train)
    rows, n = [], Counter()
    for r in e118.load(root / "v05/exp117/train.jsonl"):
        if r.get("kind") == "gui-next-action" and r["source"] == "aitw-general":
            r, _ = e118.convert(r, pool, 2, rng)
            n["aitw-general-state"] += 1
        elif r.get("kind") == "gui-next-action":
            n[r["source"] + "-original"] += 1
        rows.append(r)
    # sample shopping by episode so the training steps of one episode come together
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in gui_train:
        if r["source"] == "aitw-shopping":
            by_ep[r["episode"]].append(r)
    eps = sorted(by_ep)
    rng.shuffle(eps)
    shop = []
    for ep in eps:
        if len(shop) >= a.shopping:
            break
        shop.extend(by_ep[ep])
    shop = shop[: a.shopping]
    for r in shop:
        c, _ = e118.convert(r, pool, 2, rng)
        rows.append(c)
    n["aitw-shopping-state"] = len(shop)
    rng.shuffle(rows)
    write(out / "train.jsonl", rows)

    files = {}
    for split in ("test", "dev"):
        src = [r for r in e118.load(root / f"v05/gui/{split}.jsonl") if r["source"] == "aitw-shopping"]
        p = e118.golds_by_episode(src)
        conv = [e118.convert(r, p, 2, rng)[0] for r in src]
        if split == "dev":
            conv = rng.sample(conv, min(256, len(conv)))
        dst = out / f"gui_shop_{split}_state.jsonl"
        write(dst, conv)
        files[dst.name] = {"rows": len(conv), "sha256": e118.sha(dst),
                           "word_overlap_baseline": {"original": e118.overlap_baseline(src),
                                                     "state": e118.overlap_baseline(conv)}}
    manifest = {"built_by": "scripts/data/v05/exp119_mix.py", "seed": a.seed,
                "train_from": "v05/exp117/train.jsonl", "train_from_sha256": e118.sha(root / "v05/exp117/train.jsonl"),
                "train": {"rows": len(rows), "gui": dict(n), "sha256": e118.sha(out / "train.jsonl")},
                "shopping_episodes": len({r["episode"] for r in shop}), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
