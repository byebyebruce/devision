"""Build exp80's fixed ChartQA choice protocol, disjoint by decoded image pixels."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image

SEED = 80
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink", "gray", "grey"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(answer):
    text = " ".join(str(answer).strip().lower().split())
    unit = "%" if text.endswith("%") else ""
    number = text.removesuffix("%").replace(",", "")
    try:
        d = Decimal(number)
        if d.is_finite():
            return "number:" + str(d.normalize()) + unit
    except InvalidOperation:
        pass
    return "text:" + ("gray" if text == "grey" else text)


def pool_key(question, answer):
    text = " ".join(answer.lower().split())
    if text in ("yes", "no"):
        return "yesno"
    if text in COLORS:
        return "color"
    if canonical(answer).startswith("number:"):
        return "year" if re.fullmatch(r"(?:19|20)\d\d", text) else "number"
    return "text:" + " ".join(re.findall(r"[a-z]+", question.lower())[:4])


def convert(row, pools, sid, image, image_id, frequencies=None):
    answers = row["label"]
    if len(answers) != 1 or not str(answers[0]).strip():
        return None
    answer = str(answers[0]).strip()
    key = pool_key(row["query"], answer)
    base = dict(id=sid, source="chartqa", kind=key, image=image, image_id=image_id,
                upstream="human" if row["human_or_machine"] == 0 else "machine")
    if key == "yesno":
        p = {"false": float(answer.lower() == "no"), "true": float(answer.lower() == "yes")}
        q = {"type": "noul", "instructions": row["query"]}
    else:
        pool = pools.get(key, {})
        candidates = sorted(v for k, v in pool.items() if k not in {canonical(a) for a in answers})
        if frequencies is not None:
            counts = frequencies[key]
            frequency = max(1, counts[canonical(answer)])
            for ratio in (2, 4):
                matched = [v for v in candidates if frequency / ratio <= counts[canonical(v)] <= frequency * ratio]
                if len(matched) >= 3:
                    candidates = matched
                    break
            else:
                return None
        if len(candidates) < 3:
            return None
        rng = random.Random(int(digest(sid.encode())[:16], 16) ^ SEED)
        opts = [answer] + rng.sample(candidates, 3)
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": row["query"], "criteria": dict.fromkeys(opts)}
        p = {v: float(v == answer) for v in opts}
    return dict(base, questions={"q": q}, gold={"q": {"probabilities": p}})


def pixels(raw):
    with Image.open(io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
        data = ("%dx%d:" % rgb.size).encode() + rgb.tobytes()
        return digest(data), hashlib.sha1(data).hexdigest()


def rank(value):
    return digest((str(SEED) + ":" + value).encode())


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"questions": len(rows), "pictures": len({r["image_id"] for r in rows}),
            "sha256": digest(path.read_bytes()), "sources": dict(Counter(r["source"] for r in rows))}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", default="runs/v05-prep/chartqa_source.json")
    p.add_argument("--root", default="data")
    p.add_argument("--output", default="exp80v2")
    a = p.parse_args()
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "n2"))
    from held import load
    held = load(a.root)
    print("Existing heldout pixel hashes:", len(held.pixels), flush=True)
    root = Path(a.root)
    out = root / "v05" / a.output
    if (out / "MANIFEST.json").exists():
        raise SystemExit("Frozen data already exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    images = root / "v05" / "exp80" / "images"
    images.mkdir(exist_ok=True)
    source = json.loads(Path(a.source).read_text())
    tables, cache = {}, {}
    for split, pattern in (("train", "train-*"), ("val", "val-*")):
        rows = []
        for f in sorted((Path(source["snapshot"]) / "data").glob(pattern)):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    raw = row.pop("image")["bytes"]
                    key = digest(raw)
                    if key not in cache:
                        pixel, old_pixel = pixels(raw)
                        rel = images / (pixel + ".png")
                        if not rel.exists():
                            with Image.open(io.BytesIO(raw)) as im:
                                im.convert("RGB").save(rel)
                        cache[key] = (pixel, old_pixel, str(rel.relative_to(root)))
                    pixel, old_pixel, rel = cache[key]
                    rows.append(dict(row, _pixel=pixel, _old=old_pixel, _image=rel))
            print(split, f.name, len(rows), "rows", flush=True)
        tables[split] = rows
    val_pics = {r["_pixel"] for r in tables["val"]}
    eligible = [r for r in tables["train"] if r["_pixel"] not in val_pics and r["_old"] not in held.pixels]
    calib_pics = {r["_pixel"] for r in eligible if int(rank(r["_pixel"])[:8], 16) % 20 == 0}
    train_raw = [r for r in eligible if r["_pixel"] not in calib_pics]
    pools = defaultdict(dict)
    frequencies = defaultdict(Counter)
    for r in train_raw:
        for answer in r["label"]:
            pools[pool_key(r["query"], answer)].setdefault(canonical(answer), answer.strip())
            frequencies[pool_key(r["query"], answer)][canonical(answer)] += 1
    converted = {}
    for split, rows in (("train", train_raw), ("calib", [r for r in eligible if r["_pixel"] in calib_pics]),
                        ("dev", tables["val"])):
        result = []
        for i, r in enumerate(rows):
            s = convert(r, pools, f"chartqa:{split}:{i}", r["_image"], "chartqa:" + r["_pixel"], frequencies)
            if s:
                result.append(s)
        result.sort(key=lambda r: rank(r["id"]))
        converted[split] = result
    if len(converted["train"]) < 8000 or len(converted["calib"]) < 100:
        raise SystemExit("Insufficient distinct examples for prewritten protocol")
    train = converted["train"][:8000]
    calibration = converted["calib"][:512]
    dev = converted["dev"][:1000]
    # Freeze an independent picture slice of the existing old-task development data.
    old = [json.loads(line) for line in (root / "v5/dev_mix.jsonl").open()]
    by_source = defaultdict(list)
    for r in old:
        by_source[r["source"]].append(r)
    protect = [r for rows in by_source.values() for r in sorted(rows, key=lambda r: rank(r["id"]))[:64]]
    excluded = val_pics | calib_pics
    replay, seen = [], set()
    with (root / "n2/train_mix_fix.jsonl").open() as f:
        raw_replay = [json.loads(line) for line in f]
    raw_replay.sort(key=lambda r: rank(r["id"]))
    for r in raw_replay:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        # Existing n2 split provides old-task exclusion; check the new chart split explicitly.
        try:
            raw = (root / r["image"]).read_bytes()
            h, _ = pixels(raw)
        except OSError:
            continue
        if h in excluded:
            continue
        replay.append(r)
        if len(replay) == 8000:
            break
    if len(replay) != 8000:
        raise SystemExit("Insufficient replay")
    mixed = train + replay
    random.Random(SEED).shuffle(mixed)
    manifest = {"protocol": "exp80-v2-frequency-matched", "source": source, "seed": SEED, "raw": {k: len(v) for k, v in tables.items()},
                "eligible": len(eligible), "converted": {k: len(v) for k, v in converted.items()},
                "external_held_pixel_hashes": len(held.pixels),
                "exclusion_cache_sha256": digest((root / "n2/held.json").read_bytes()),
                "source_parquets": {f.name: digest(f.read_bytes()) for f in (Path(source["snapshot"]) / "data").glob("*.parquet")},
                "files": {}}
    for name, rows in {"train": mixed, "fit": train[:256], "calib": calibration, "chart_dev": dev, "protect": protect}.items():
        manifest["files"][name] = write_rows(out / (name + ".jsonl"), rows)
    train_ids = {r["image_id"] for r in train}
    assert not train_ids & {r["image_id"] for r in calibration + dev}
    positions = Counter()
    for r in dev:
        probs = r["gold"]["q"]["probabilities"]
        positions[list(probs).index(max(probs, key=probs.get))] += 1
    manifest["dev_gold_positions"] = dict(positions)
    manifest["limits"] = ["Locally constructed choices, not official open-answer scoring", "Pixel equality exclusion only",
                          "ImageJevBench public image exclusion pending access", "validation is development, test untouched"]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
