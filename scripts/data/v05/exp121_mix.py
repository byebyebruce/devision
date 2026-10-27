"""exp121 training mix: exp120 (exp119 + depth replay + ego 24k) with more phone GUI screen-state rows.

exp119 showed the AITW screen-state questions need the screen (test_gui_shop_state real 59.8 vs mismatched 31.1)
and that 12k shopping rows lifted unseen-website shopping by +10.2 [+6.7, +13.7] over exp118 without hurting old
abilities. Here, on top of exp120's file (unchanged):
- + --general AITW-general rows not yet used, screen-state variant;
- + --shopping AITW web_shopping rows from episodes not yet used (whole episodes), screen-state variant.
Mind2Web stays at exp120's 6k original-format rows (unreadable at 256). Eval sets are exp119's.
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
    ap.add_argument("--seed", type=int, default=121)
    ap.add_argument("--general", type=int, default=10000)
    ap.add_argument("--shopping", type=int, default=14000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp121"
    out.mkdir(parents=True, exist_ok=True)

    base = e118.load(root / "v05/exp120/train.jsonl")
    used_ids = {r["id"] for r in base}
    used_eps = {r["episode"] for r in base if r.get("kind") == "gui-next-action" and "episode" in r}
    gui_train = e118.load(root / "v05/gui/train.jsonl")
    pool = e118.golds_by_episode(gui_train)

    added, n = [], Counter()
    for source, want in (("aitw-general", a.general), ("aitw-shopping", a.shopping)):
        by_ep: dict[str, list[dict]] = defaultdict(list)
        for r in gui_train:
            if r["source"] == source and r["id"] not in used_ids and r["episode"] not in used_eps:
                by_ep[r["episode"]].append(r)
        eps = sorted(by_ep)
        rng.shuffle(eps)
        take: list[dict] = []
        for ep in eps:
            if len(take) >= want:
                break
            take.extend(by_ep[ep])
        take = take[:want]
        assert len(take) == want, (source, len(take))
        for r in take:
            added.append(e118.convert(r, pool, 2, rng)[0])
        n[source + "-state"] = len(take)
        n[source + "-episodes"] = len({r["episode"] for r in take})
    rows = base + added
    assert len({r["id"] for r in added}) == len(added) and not used_ids & {r["id"] for r in added}
    rng.shuffle(rows)
    write(out / "train.jsonl", rows)
    gui = Counter(r["source"] for r in rows if r.get("kind") == "gui-next-action")
    manifest = {"built_by": "scripts/data/v05/exp121_mix.py", "seed": a.seed,
                "train_from": "v05/exp120/train.jsonl", "train_from_sha256": e118.sha(root / "v05/exp120/train.jsonl"),
                "added": dict(n), "gui_rows_total": dict(gui),
                "train": {"rows": len(rows), "sha256": e118.sha(out / "train.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
