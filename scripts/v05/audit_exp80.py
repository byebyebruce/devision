"""Freeze image isolation and simple nonvisual controls before exp80 training."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from chart_data import pixels


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main():
    root = Path("data")
    data = root / "v05/exp80v2"
    rows = {n: read(data / (n + ".jsonl")) for n in ("train", "calib", "chart_dev", "protect")}
    rows["examples"] = read(root / "v05/external/jev_examples.jsonl")
    rows["realworld"] = read(root / "v05/external/realworld_native.jsonl")
    files = sorted({r["image"] for rs in rows.values() for r in rs})
    def pixel(path):
        return path, pixels((root / path).read_bytes())[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(pixel, files))
    ids = {n: {hashes[r["image"]] for r in rs} for n, rs in rows.items()}
    overlaps = {n: len(ids["train"] & v) for n, v in ids.items() if n != "train"}
    priors = defaultdict(Counter)
    for r in rows["train"]:
        q = r["questions"]["q"]
        probs = r["gold"]["q"]["probabilities"]
        if r["source"] == "chartqa":
            priors[r["kind"]][max(probs, key=probs.get)] += 1
        assert abs(sum(probs.values()) - 1) < 1e-7
        assert all(0 <= v <= 1 for v in probs.values())
        assert set(probs) == (set(q["criteria"]) if q["type"] == "choice" else {"false", "true"})
    hits, random_sum, pos = 0, 0.0, Counter()
    for r in rows["chart_dev"]:
        probs = r["gold"]["q"]["probabilities"]
        pred = max(sorted(probs), key=lambda o: priors[r["kind"]][o])
        hits += probs[pred]
        random_sum += 1 / len(probs)
        pos[list(probs).index(max(probs, key=probs.get))] += 1
    result = {"images_hashed": len(files), "train_overlap_by_decoded_pixels": overlaps,
              "label_prior_accuracy": hits / len(rows["chart_dev"]),
              "random_accuracy": random_sum / len(rows["chart_dev"]),
              "majority_position_accuracy": max(pos.values()) / len(rows["chart_dev"]),
              "train_questions": len(rows["train"]),
              "manifest_sha256": hashlib.sha256((data / "MANIFEST.json").read_bytes()).hexdigest(),
              "limits": ["Pixel equality does not exclude near-duplicates", "Official full ImageJevBench not available",
                         "Earlier base model training exposure to newly fetched tests not audited"]}
    result["pass"] = not any(overlaps.values()) and result["train_questions"] == 16000
    Path("runs/v05-prep/data_audit_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit("Image isolation or training budget failed")


if __name__ == "__main__":
    main()
