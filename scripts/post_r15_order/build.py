"""Build the authorised post-r15 order A/B experiment without modifying its source pools.

    uv run --no-sync python scripts/post_r15_order/build.py

8,000 distinct base questions -> 16,000 exposures per arm. A repeats the original;
B changes only the criteria enumeration of the second choice view. Both share batches.
"""
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path("data")
OUT = ROOT / "post-r15-order"
EXT = ("clevr", "snli_ve", "figureqa", "iconqa", "vision_flan", "visonlyqa",
       "clevr_math", "superclevr", "mapqa", "spatialsense")
MONITORS = {
    "val": "data/r13b/val.jsonl", "ext": "data/ext/dev_ext.jsonl",
    "science": "data/r13b/panel_science.jsonl", "exist": "data/r13b/panel_exist.jsonl",
    "flip": "data/v6/dev_lr_pairs.jsonl", "count_wide": "data/r13b/count_wide.jsonl",
    "count_narrow": "data/v11/dev_count_transfer.jsonl",
    "count_wide_reverse": "data/post-r15-order/count_wide_reverse.jsonl",
    "count_narrow_reverse": "data/post-r15-order/count_narrow_reverse.jsonl",
}
NUMBERS = {s: i for i, s in enumerate("zero one two three four five six seven eight nine ten".split())}
NUMBERS["none"] = 0


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, data):
    with Path(path).open("w") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(s, aliases):
    key = s["image_id"].removesuffix(":flip")
    return aliases.get(key, key)


def flatten(ss):
    out = []
    for s in ss:
        for k, q in s["questions"].items():
            out.append(dict(s, questions={"q": q}, gold={"q": s["gold"][k]}, original_qid=k))
    return out


def question_key(s, aliases):
    q = s["questions"]["q"]
    return (canonical(s, aliases), s["image_id"].endswith(":flip"), q["type"],
            q["instructions"].strip(), json.dumps(s.get("state_text", ""), sort_keys=True))


def number(s):
    text = str(s).lower().strip()
    return int(text) if text.isdigit() else NUMBERS.get(text)


def top(s):
    p = s["gold"]["q"]["probabilities"]
    return max(p, key=p.get)


def is_count(s):
    q = s["questions"]["q"]
    return (q["type"] == "choice" and bool(re.match(r"(?i)how many\b", q["instructions"]))
            and all(number(x) is not None for x in q["criteria"]))


def views(s, arm, rng):
    """Stable IDs/group/qid in both arms; keep labels/descriptions attached to option keys."""
    out = []
    for view in range(2):
        r = copy.deepcopy(s)
        r["id"] = s["id"] + ":view%d" % view
        r["view"] = view
        q = r["questions"]["q"]
        if arm == "b" and view and q["type"] == "choice":
            original = list(q["criteria"])
            keys = original[:]
            while keys == original:
                rng.shuffle(keys)
            q["criteria"] = {k: q["criteria"][k] for k in keys}
        r.pop("answer_position", None)
        out.append(r)
    return out


def evaluation_files():
    paths = set()
    missing = set()
    for file in sorted(Path("configs").glob("*.yaml")):
        cfg = yaml.safe_load(file.read_text())
        candidates = []
        for st in cfg.get("stages") or []:
            candidates += [st.get("val")] + list((st.get("eval") or {}).values())
        ev = cfg.get("evaluate") or {}
        candidates += [ev.get("calibration_fit")] + list((ev.get("sets") or {}).values())
        candidates += ev.get("history") or []
        for item in candidates:
            p = item.get("path") if isinstance(item, dict) else item
            if p and p.endswith(".jsonl") and not p.startswith(str(OUT)):
                (paths if Path(p).exists() else missing).add(p)
    return sorted(paths), sorted(missing)


def image_hash(rel):
    with Image.open(ROOT / rel) as im:
        pixels = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
        return sum(int(x) << (63 - i) for i, x in enumerate(bits))


class Selector:
    def __init__(self, aliases, blocked_ids, blocked_files, rejected_hashes):
        self.aliases, self.blocked_ids, self.blocked_files = aliases, blocked_ids, blocked_files
        self.rejected_hashes = rejected_hashes
        self.used = set()
        self.rng = random.Random(1515)

    def allowed(self, s):
        return (canonical(s, self.aliases) not in self.blocked_ids
                and s["image"] not in self.blocked_files and s["image"] not in self.rejected_hashes
                and question_key(s, self.aliases) not in self.used)

    def take(self, pool, n, block, stratum=lambda s: s["source"]):
        bins = defaultdict(list)
        seen = set()
        for s in sorted(pool, key=lambda s: s["id"]):
            key = question_key(s, self.aliases)
            if key not in seen and self.allowed(s):
                bins[stratum(s)].append(s)
                seen.add(key)
        for values in bins.values():
            self.rng.shuffle(values)
        keys = sorted(bins)
        self.rng.shuffle(keys)
        picked, images = [], set()
        # First prefer different images, then allow additional questions from an image.
        for unique in (True, False):
            while len(picked) < n:
                progress = False
                for key in keys:
                    candidates = bins[key]
                    idx = next((i for i, s in enumerate(candidates) if self.allowed(s)
                                and (not unique or canonical(s, self.aliases) not in images)), None)
                    if idx is None:
                        continue
                    s = candidates.pop(idx)
                    self.used.add(question_key(s, self.aliases))
                    images.add(canonical(s, self.aliases))
                    picked.append(dict(s, block=block))
                    progress = True
                    if len(picked) == n:
                        break
                if not progress:
                    break
        if len(picked) != n:
            raise ValueError("%s: only %d / %d distinct eligible questions" % (block, len(picked), n))
        return picked


def choose(pool, r15, sel):
    out = []
    # Mirror families first so general replay cannot consume one member of a chosen pair.
    families = defaultdict(list)
    for s in r15:
        if s.get("block") == "mirror":
            families[s["id"].removeprefix("flip:")].append(s)
    pairs = [ss for _, ss in sorted(families.items()) if len(ss) == 2 and all(sel.allowed(s) for s in ss)]
    sel.rng.shuffle(pairs)
    chosen = []
    for source, n in (("coco-relation", 120), ("coco-position", 80)):
        part = [ss for ss in pairs if ss[0]["source"] == source][:n]
        if len(part) != n:
            raise ValueError("insufficient mirror pairs: %s" % source)
        for ss in part:
            group = "mirror:" + ss[0]["id"].removeprefix("flip:")
            for s in ss:
                sel.used.add(question_key(s, sel.aliases))
                chosen.append(dict(s, block="mirror", order_group=group))
    out += chosen
    for bucket, lo, hi in (("0-2", 0, 2), ("3-5", 3, 5), ("6-10", 6, 10)):
        part = [s for s in r15 if s.get("block") == "count_balanced" and s.get("candidates") == "0..10"
                and lo <= number(top(s)) <= hi]
        out += sel.take(part, 200, "count_wide")
    out += sel.take([s for s in r15 if s.get("block") == "count_balanced" and s.get("candidates") == "run"],
                    500, "count_narrow")
    out += sel.take([s for s in r15 if s.get("block") == "count_original"], 500, "count_original")
    for answer in ("true", "false"):
        out += sel.take([s for s in pool if s["source"] == "vsr" and top(s) == answer],
                        600, "vsr", lambda s: s.get("category", ""))
    out += sel.take([s for s in pool if s["source"] == "aokvqa"], 800, "aokvqa")
    ex = defaultdict(lambda: defaultdict(list))
    for s in r15:
        if s.get("block") == "existence" and sel.allowed(s):
            ex[s["questions"]["q"]["instructions"]][top(s)].append(s)
    slots = [cat for cat in sorted(ex) for _ in range(min(len(ex[cat]["true"]), len(ex[cat]["false"])))]
    if len(slots) < 600:
        raise ValueError("insufficient balanced existence pairs")
    for cat, n in sorted(Counter(sel.rng.sample(slots, 600)).items()):
        for answer in ("true", "false"):
            out += sel.take(ex[cat][answer], n, "existence")
    for source, n in (("scienceqa", 400), ("ai2d", 200), ("tqa", 200)):
        out += sel.take([s for s in pool if s["source"] == source], n, "science")
    for sources, n in ((("vqav2-yesno",), 200), (("gqa-v2",), 200), (("vqav2-choice",), 100),
                       (("v7w",), 100), (("coco-relation", "vg-relation"), 100), (("coco-size",), 100)):
        out += sel.take([s for s in pool if s["source"] in sources and not is_count(s)], n, "general")
    for source in EXT:
        out += sel.take([s for s in r15 if s.get("block") == "ext" and s["source"] == source], 120, "ext")
    assert len(out) == 8000 and len(sel.used) == 8000
    for s in out:
        s["original_id"] = s["id"]
        key = json.dumps(question_key(s, sel.aliases), ensure_ascii=False)
        s["id"] = "post-r15:" + hashlib.sha256(key.encode()).hexdigest()[:24]
        s["group"] = s.pop("order_group", s["id"])
        s["image_id"] = canonical(s, sel.aliases) + (":flip" if s["image_id"].endswith(":flip") else "")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "MANIFEST.json").exists():
        raise SystemExit("dataset already frozen; do not overwrite it")
    aliases = json.loads((ROOT / "v5/image_identity.json").read_text())
    paths, missing = evaluation_files()
    for p in MONITORS.values():
        if not p.startswith(str(OUT)) and p not in paths:
            paths.append(p)
    eval_rows = [s for p in paths for s in rows(p)]
    blocked_ids = {canonical(s, aliases) for s in eval_rows}
    blocked_files = {s["image"] for s in eval_rows}
    print("exclusion: %d files, %d canonical images" % (len(paths), len(blocked_ids)), flush=True)
    # Conservative near-duplicate exclusion for photos with opaque identities; diagrams retain ID exclusion.
    cache_file = OUT / "eval_image_hashes.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    known = json.loads((ROOT / "r13/identity_hashes.json").read_text())["hashes"]
    cache.update({p: int(v) for p, v in known.items() if p in blocked_files})
    missing_images = sorted(p for p in blocked_files if not (ROOT / p).is_file())
    for name, p in MONITORS.items():
        if not p.startswith(str(OUT)):
            for s in rows(p):
                if not (ROOT / s["image"]).is_file():
                    raise FileNotFoundError("required monitor %s: %s" % (name, s["image"]))
    photos = sorted({s["image"] for s in eval_rows if s.get("source") not in
                     ("scienceqa", "ai2d", "tqa", "figureqa", "iconqa", "clevr", "clevr_math", "superclevr", "mapqa", "cosyn")
                     and (ROOT / s["image"]).is_file()})
    todo = [p for p in photos if p not in cache]
    print("hashing %d evaluation photos (cached %d)" % (len(todo), len(cache)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for rel, value in zip(todo, ex.map(image_hash, todo)):
            cache[rel] = value
    cache_file.write_text(json.dumps(cache))
    values = np.array([cache[p] for p in photos], dtype=np.uint64)
    pool, r15 = flatten(rows(ROOT / "ext/train_mix.jsonl")), flatten(rows(ROOT / "r14b/train.jsonl"))
    rejected, near_matches = set(), {}
    train_hashes = {}
    for attempt in range(10):
        base = choose(pool, r15, Selector(aliases, blocked_ids, blocked_files, rejected))
        opaque = sorted({s["image"] for s in base if s["source"] in ("aokvqa", "tallyqa", "pixmo_count", "vision_flan")})
        todo = [p for p in opaque if p not in train_hashes]
        with ThreadPoolExecutor(max_workers=4) as ex:
            train_hashes.update(zip(todo, ex.map(image_hash, todo)))
        new = set()
        for rel in opaque:
            dist = np.bitwise_count(values ^ np.uint64(train_hashes[rel]))
            indices = np.flatnonzero(dist <= 2)
            if len(indices):
                new.add(rel)
                near_matches[rel] = [{"evaluation_image": photos[int(i)], "bits": int(dist[i])} for i in indices[:5]]
        if not new:
            break
        rejected.update(new)
        print("exclude %d near-photo matches and rebuild (pass %d)" % (len(new), attempt + 1), flush=True)
    else:
        raise ValueError("near-duplicate exclusions did not converge")
    for s in base:
        if not (ROOT / s["image"]).is_file():
            raise FileNotFoundError(s["image"])
    random.Random(1516).shuffle(base)
    write(OUT / "base.jsonl", base)
    for arm in ("a", "b"):
        rng = random.Random(1517)
        write(OUT / (arm + ".jsonl"), [v for s in base for v in views(s, arm, rng)])
    for name in ("count_wide", "count_narrow"):
        rev = copy.deepcopy(rows(MONITORS[name]))
        for s in rev:
            for q in s["questions"].values():
                q["criteria"] = dict(reversed(list(q["criteria"].items())))
        write(OUT / (name + "_reverse.jsonl"), rev)
    manifest = {
        "base_questions": len(base), "exposures_per_arm": 2 * len(base),
        "unique_original_images": len({canonical(s, aliases) for s in base}),
        "blocks": dict(Counter(s["block"] for s in base)), "sources": dict(Counter(s["source"] for s in base)),
        "vsr_relations": dict(Counter(s.get("category", "") for s in base if s["block"] == "vsr")),
        "excluded_near_matches": near_matches, "exclusion_files": {p: sha(p) for p in paths},
        "historical_missing_references": missing,
        "historical_missing_images": missing_images,
        "identity_limit": "Canonical COCO/VG + mirror + path; dHash <=2 for selected opaque natural-photo sources. "
                          "Not an exhaustive perceptual audit of all other sources or pretraining.",
        "input_sha256": {p: sha(p) for p in ("data/ext/train_mix.jsonl", "data/r14b/train.jsonl", "data/v5/image_identity.json")},
        "output_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*.jsonl"))}, "monitor_files": MONITORS,
        "construction_seeds": {"selection": 1515, "base_order": 1516, "permutation": 1517},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("base_questions", "exposures_per_arm", "unique_original_images", "blocks")}, indent=2))


if __name__ == "__main__":
    main()
