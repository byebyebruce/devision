"""exp118 data: GUI next-action questions that need the screen ("screen-state" variant).

exp116 showed the original GUI questions are answerable from text: with a wrong screenshot the model still
scores 78.9% (real 80.6%), because goal + previous actions pick the gold among same-screen OCR / HTML
elements. Here each GUI question is rewritten so that text alone cannot tell which step we are at:
- the action history is dropped (no "Previous actions"; state_text becomes empty);
- up to --episode-distractors options are replaced by the gold actions of *other steps of the same
  episode* (all goal-relevant; only the screen shows whether e.g. the search box is already open);
  the rest of the options stay the original same-screen distractors, so the option count is unchanged.
Same ids and images as the source rows. The exp116 training mix is copied with its GUI rows replaced
(single change vs exp116); the frozen exp116 GUI dev / test are converted the same way, with the episode
pool taken from the same file (dev / test episodes never overlap with train).
"""
import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def golds_by_episode(rows: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("kind") == "gui-next-action":
            p = r["gold"]["q"]["probabilities"]
            out[r["episode"]].add(max(p, key=lambda k: p[k]))
    return out


def convert(r: dict, pool: dict[str, set[str]], k: int, rng: random.Random) -> tuple[dict, int]:
    q = r["questions"]["q"]
    p = r["gold"]["q"]["probabilities"]
    gold = max(p, key=lambda o: p[o])
    opts = list(q["criteria"])
    others = [o for o in opts if o != gold]
    cand = sorted(a for a in pool[r["episode"]] if a != gold and a not in opts)
    ep = rng.sample(cand, min(k, len(cand), len(others)))
    keep = rng.sample(others, len(others) - len(ep))
    new = [gold] + keep + ep
    rng.shuffle(new)
    out = dict(r)
    out["state_text"] = ""
    out["questions"] = {"q": {**q, "criteria": {o: None for o in new}}}
    out["gold"] = {"q": {"probabilities": {o: float(o == gold) for o in new}}}
    out["n_episode_distractors"] = len(ep)
    out["variant"] = "screen-state"
    return out, len(ep)


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def overlap_baseline(rows: list[dict]) -> float:
    hit = 0
    for r in rows:
        q = r["questions"]["q"]
        p = r["gold"]["q"]["probabilities"]
        ctx = words(q["instructions"] + " " + r.get("state_text", ""))
        best = max(q["criteria"], key=lambda o: len(words(o) & ctx))
        hit += p[best] == 1.0
    return round(hit / len(rows), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--episode-distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=118)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    src, out = root / "v05/exp116", root / "v05/exp118"
    out.mkdir(parents=True, exist_ok=True)

    train_pool = golds_by_episode(load(root / "v05/gui/train.jsonl"))
    stats: dict = {}
    rows = []
    n_ep = defaultdict(int)
    for r in load(src / "train.jsonl"):
        if r.get("kind") == "gui-next-action":
            r, k = convert(r, train_pool, a.episode_distractors, rng)
            n_ep[k] += 1
        rows.append(r)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["train"] = {"rows": len(rows), "gui_rows_by_episode_distractors": dict(n_ep)}

    files = {}
    for name in ("gui_test", "gui_dev", "gui_dev_mon"):
        orig = load(src / f"{name}.jsonl")
        pool = golds_by_episode(load(src / ("gui_dev.jsonl" if name == "gui_dev_mon" else f"{name}.jsonl")))
        conv, n_ep = [], defaultdict(int)
        for r in orig:
            c, k = convert(r, pool, a.episode_distractors, rng)
            conv.append(c)
            n_ep[k] += 1
        dst = out / f"{name}_state.jsonl"
        with open(dst, "w") as f:
            for r in conv:
                f.write(json.dumps(r) + "\n")
        bysrc = defaultdict(list)
        for o, c in zip(orig, conv):
            bysrc[o["source"]].append((o, c))
        files[f"{name}_state"] = {
            "from": f"v05/exp116/{name}.jsonl", "rows": len(conv), "sha256": sha(dst),
            "by_episode_distractors": dict(n_ep),
            "word_overlap_baseline": {s: {"original": overlap_baseline([o for o, _ in v]),
                                          "state": overlap_baseline([c for _, c in v])} for s, v in bysrc.items()},
        }
    manifest = {"built_by": "scripts/data/v05/exp118_gui_state.py", "seed": a.seed,
                "episode_distractors": a.episode_distractors, "train_from": "v05/exp116/train.jsonl",
                "train_from_sha256": sha(src / "train.jsonl"), **stats, "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
