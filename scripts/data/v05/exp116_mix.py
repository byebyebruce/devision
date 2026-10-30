"""exp116 training mix: exp115 (v0.4 replay + chart + ScreenQA UI + counting) + real GUI next-action + egocentric.

GUI (data/v05/gui, AITW + Multimodal-Mind2Web) and ego (data/v05/ego, EPIC-KITCHENS VISOR) are read only;
their dev / test files are copied into data/v05/exp116/ (with sha256 in MANIFEST.json) so later rebuilds of
those directories cannot change what exp116 and its baselines were evaluated on.
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--gui", type=int, default=12000)
    ap.add_argument("--ego", type=int, default=12000)
    ap.add_argument("--dev", type=int, default=512, help="per-domain monitoring subset size")
    ap.add_argument("--seed", type=int, default=116)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    out = root / "v05/exp116"
    out.mkdir(parents=True, exist_ok=True)

    base = load(root / "v05/exp115/train.jsonl")
    # GUI: up to half web (Mind2Web, scarcer), phone (AITW, any subset) fills the rest.
    gui = load(root / "v05/gui/train.jsonl")
    web = [r for r in gui if r["source"] == "mind2web"]
    phone = [r for r in gui if r["source"] != "mind2web"]
    gui_pick = rng.sample(web, min(len(web), a.gui // 2))
    gui_pick += rng.sample(phone, min(len(phone), a.gui - len(gui_pick)))
    # Ego: uniform over the three kinds.
    ego = load(root / "v05/ego/train.jsonl")
    by_kind = defaultdict(list)
    for r in ego:
        by_kind[r["kind"]].append(r)
    ego_pick = []
    for k in sorted(by_kind):
        ego_pick += rng.sample(by_kind[k], min(len(by_kind[k]), a.ego // len(by_kind)))

    rows = base + gui_pick + ego_pick
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    files = {}
    for name, src in [("gui_test", "v05/gui/test.jsonl"), ("gui_dev", "v05/gui/dev.jsonl"),
                      ("ego_test", "v05/ego/test.jsonl"), ("ego_dev", "v05/ego/dev.jsonl"),
                      ("egothink", "v05/ego/bench_egothink.jsonl")]:
        dst = out / f"{name}.jsonl"
        shutil.copyfile(root / src, dst)
        files[name] = {"from": src, "rows": sum(1 for _ in open(dst)), "sha256": sha(dst)}
    for name in ("gui_dev", "ego_dev"):
        dev = load(out / f"{name}.jsonl")
        sub = rng.sample(dev, min(a.dev, len(dev)))
        dst = out / f"{name}_mon.jsonl"
        with open(dst, "w") as f:
            for r in sub:
                f.write(json.dumps(r) + "\n")
        files[f"{name}_mon"] = {"from": f"{name}.jsonl", "rows": len(sub), "sha256": sha(dst)}

    count = defaultdict(int)
    for r in gui_pick + ego_pick:
        count[r["source"] + "/" + r["kind"]] += 1
    manifest = {"built_by": "scripts/data/v05/exp116_mix.py", "seed": a.seed, "rows": len(rows),
                "exp115": len(base), "gui": len(gui_pick), "ego": len(ego_pick), "added_by_source_kind": dict(count),
                "gui_manifest_sha256": sha(root / "v05/gui/MANIFEST.json"),
                "ego_manifest_sha256": sha(root / "v05/ego/MANIFEST.json"), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
