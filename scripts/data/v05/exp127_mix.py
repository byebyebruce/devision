"""exp127 training mix: r22 data (= exp124 file, unchanged) + the rest of the egocentric pool + more AITW-general.

Owner direction 2026-10-27: scale the real phone/web GUI and first-person data. exp120 showed +12k EPIC-VISOR lifted
test_ego +3.6; the train pool (train participants only) still has ~19.6k unused rows. AITW-general (many apps; the
test_gui_state AITW part is unseen apps) has ~19k unused rows. Here, on top of data/v05/exp124/train.jsonl:
- + --ego remaining EPIC-VISOR rows not in the base (all of them by default), whatever kind;
- + --general unused AITW-general rows, grouped by episode (only ~2.6k sit in wholly unused episodes, so remaining steps
  of partly used episodes are allowed; all are train-split apps), screen-state variant (exp118 convert).
Shopping stays at 26k (exp121: more shopping cost MME / MMBench ~1pp). Mind2Web unchanged (unreadable at 256).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=127)
    ap.add_argument("--ego", type=int, default=0, help="0 = every unused train-participant row")
    ap.add_argument("--general", type=int, default=9000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp127"
    out.mkdir(parents=True, exist_ok=True)

    base = e118.load(root / "v05/exp124/train.jsonl")
    used_ids = {r["id"] for r in base}

    ego = [r for r in e118.load(root / "v05/ego/train.jsonl") if r["id"] not in used_ids]
    rng.shuffle(ego)
    if a.ego:
        ego = ego[: a.ego]

    gui_train = e118.load(root / "v05/gui/train.jsonl")
    pool = e118.golds_by_episode(gui_train)
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in gui_train:
        if r["source"] == "aitw-general" and r["id"] not in used_ids:
            by_ep[r["episode"]].append(r)
    eps = sorted(by_ep)
    rng.shuffle(eps)
    take: list[dict] = []
    for ep in eps:
        if len(take) >= a.general:
            break
        take.extend(by_ep[ep])
    take = take[: a.general]
    assert len(take) == a.general, len(take)
    general = [e118.convert(r, pool, 2, rng)[0] for r in take]

    added = ego + general
    assert len({r["id"] for r in added}) == len(added) and not used_ids & {r["id"] for r in added}
    rows = base + added
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {"built_by": "scripts/data/v05/exp127_mix.py", "seed": a.seed,
                "train_from": "v05/exp124/train.jsonl", "train_from_sha256": e118.sha(root / "v05/exp124/train.jsonl"),
                "added": {"epic-visor": dict(Counter(r["kind"] for r in ego)),
                          "aitw-general-state": len(general),
                          "aitw-general-episodes": len({r["episode"] for r in take})},
                "rows_by_source": dict(Counter(r["source"] for r in rows)),
                "train": {"rows": len(rows), "sha256": e118.sha(out / "train.jsonl")}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
